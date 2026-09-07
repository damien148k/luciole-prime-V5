"""
FileWatcher — Surveillance du filesystem via watchdog PollingObserver.

Utilise le `PollingObserver` (et non l'`Observer` natif basé sur inotify)
pour garantir la fiabilité sur les volumes Docker bind-mount, CIFS et NFS
où les événements inotify ne sont pas propagés de manière fiable.

Mécanisme de debounce :
Un éditeur qui sauvegarde un fichier peut générer 3 à 5 événements
(create temp → write → rename → modify). Le debounce attend DEBOUNCE_SECONDS
secondes après le dernier événement pour un path donné avant d'enqueue un job.
Cela fusionne les rafales d'événements en un seul job.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger
from watchdog.events import (
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers.polling import PollingObserver

from .constants import (
    DEFAULT_EXCLUDED_DIRS,
    IGNORED_EXTENSIONS,
    IGNORED_FILENAME_PREFIXES,
    JOB_ACTION_DELETE,
    JOB_ACTION_MOVE,
    JOB_ACTION_UPSERT,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from .index_routing import derive_index_name
from .models import WatchedPath, WatcherConfig

if TYPE_CHECKING:
    from .queue import JobQueue


class _WatcherEventHandler(FileSystemEventHandler):
    """
    Gestionnaire d'événements watchdog interne.

    Délègue immédiatement au FileWatcher parent pour le filtrage et le debounce.
    Ne contient aucune logique métier.
    """

    def __init__(self, watcher: "FileWatcher", watched_path: WatchedPath) -> None:
        super().__init__()
        self._watcher = watcher
        self._watched_path = watched_path

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._on_file_event(
                event.src_path, JOB_ACTION_UPSERT,
                watched_path=self._watched_path,
            )

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._on_file_event(
                event.src_path, JOB_ACTION_UPSERT,
                watched_path=self._watched_path,
            )

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._on_file_event(
                event.src_path, JOB_ACTION_DELETE,
                watched_path=self._watched_path,
            )

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._watcher._on_file_event(
                event.dest_path, JOB_ACTION_MOVE,
                old_path=event.src_path,
                watched_path=self._watched_path,
            )


class FileWatcher:
    """
    Observe les chemins configurés et enqueue des jobs d'indexation.

    Tourne dans son propre thread (démarré par `start()`).
    N'effectue aucun traitement lourd — uniquement du filtrage et du debounce.
    """

    def __init__(
        self,
        config: WatcherConfig,
        queue: "JobQueue",
        db_conn: sqlite3.Connection,
    ) -> None:
        """
        Args:
            config: Configuration du watcher.
            queue: File d'attente des jobs.
            db_conn: Connexion SQLite pour journaliser les événements.
        """
        self._config = config
        self._queue = queue
        self._db_conn = db_conn

        self._observer: Optional[PollingObserver] = None
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._debounce_lock = threading.Lock()
        self._running = False

        # Extensions autorisées (fusion config + défaut)
        self._allowed_extensions: frozenset[str] = self._build_allowed_extensions()
        self._excluded_dirs: frozenset[str] = (
            DEFAULT_EXCLUDED_DIRS
            | frozenset(self._config.excluded_dirs)
        )

    def start(self) -> None:
        """
        Démarre l'observation des chemins configurés.

        Crée un PollingObserver et planifie un handler par chemin surveillé.
        """
        if self._running:
            logger.warning("FileWatcher déjà démarré — ignoré")
            return

        self._observer = PollingObserver(timeout=self._config.polling_interval)

        scheduled_count = 0
        for wp in self._config.watched_paths:
            path = Path(wp.path)
            if not path.exists() or not path.is_dir():
                logger.warning(f"Chemin ignoré au démarrage (introuvable) : {wp.path}")
                continue

            handler = _WatcherEventHandler(self, wp)
            self._observer.schedule(handler, str(path), recursive=wp.recursive)
            scheduled_count += 1
            logger.info(
                f"Surveillance démarrée : {wp.path} "
                f"(récursif={wp.recursive}, index={wp.index_name or self._config.default_index_name})"
            )

        if scheduled_count == 0:
            logger.warning("Aucun chemin surveillé valide — le watcher ne fait rien")

        self._observer.start()
        self._running = True
        logger.info(f"FileWatcher démarré ({scheduled_count} chemin(s))")

    def stop(self) -> None:
        """
        Arrête l'observateur et annule les timers de debounce en cours.
        """
        self._running = False

        # Annuler tous les timers debounce actifs
        with self._debounce_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=10)
            self._observer = None

        logger.info("FileWatcher arrêté")

    @property
    def is_running(self) -> bool:
        """Indique si le watcher est actif."""
        return self._running

    # ─────────────────────────────────────────────────────────────────────
    # Traitement des événements
    # ─────────────────────────────────────────────────────────────────────

    def _on_file_event(
        self,
        path: str,
        action: str,
        old_path: Optional[str] = None,
        watched_path: Optional[WatchedPath] = None,
    ) -> None:
        """
        Point d'entrée pour tous les événements filesystem.

        Filtre, journalise, puis planifie un job via debounce.
        """
        normalized = str(Path(path).resolve())
        reason = self._should_ignore(normalized)

        # Journaliser l'événement brut (ignoré ou non)
        self._log_event(
            event_type=action,
            file_path=normalized,
            old_path=str(Path(old_path).resolve()) if old_path else None,
            ignored=(reason is not None),
            reason=reason,
        )

        if reason:
            logger.debug(f"Événement ignoré [{reason}] : {normalized}")
            return

        # Auto-derive l'index_name depuis le premier sous-dossier sous le
        # watched_path. Permet le multi-projet (chavenay/, client-x/, etc.)
        # sans modifier la config à chaque nouveau dossier.
        index_name = derive_index_name(
            file_path=normalized,
            watched_paths=self._config.watched_paths,
            default_index_name=self._config.default_index_name,
        )
        logger.debug(f"Index résolu : {normalized} → index_name={index_name}")

        self._schedule_debounced(
            path=normalized,
            action=action,
            old_path=str(Path(old_path).resolve()) if old_path else None,
            index_name=index_name,
        )

    def _should_ignore(self, path: str) -> Optional[str]:
        """
        Détermine si un chemin doit être ignoré, et pourquoi.

        Returns:
            Raison (str) si à ignorer, None sinon.
        """
        p = Path(path)

        # Fichiers temporaires par extension
        ext = p.suffix.lower()
        if ext in IGNORED_EXTENSIONS:
            return f"extension_temporaire:{ext}"

        # Extensions non autorisées
        if ext and ext not in self._allowed_extensions:
            return f"extension_non_supportee:{ext}"

        # Préfixes de noms temporaires
        for prefix in IGNORED_FILENAME_PREFIXES:
            if p.name.startswith(prefix):
                return f"prefixe_temporaire:{prefix}"

        # Répertoires exclus dans le chemin
        parts = set(p.parts)
        for excluded in self._excluded_dirs:
            if excluded in parts:
                return f"repertoire_exclu:{excluded}"

        # Extensions explicitement exclues dans la config
        if ext in frozenset(self._config.excluded_extensions):
            return f"extension_exclue_config:{ext}"

        # Taille max (seulement si le fichier existe encore)
        try:
            if p.exists() and p.stat().st_size > self._config.max_file_size_mb * 1024 * 1024:
                return f"fichier_trop_grand:{p.stat().st_size}"
        except OSError:
            pass

        return None

    def _schedule_debounced(
        self,
        path: str,
        action: str,
        old_path: Optional[str],
        index_name: str,
    ) -> None:
        """
        Planifie l'enqueue du job avec un délai de debounce.

        Si un timer existe déjà pour ce path, il est annulé et remplacé.
        Cela garantit qu'un seul job est créé même si plusieurs événements
        arrivent rapidement pour le même fichier.
        """
        with self._debounce_lock:
            existing = self._debounce_timers.pop(path, None)
            if existing:
                existing.cancel()

            timer = threading.Timer(
                self._config.debounce_seconds,
                self._fire_job,
                kwargs={
                    "path": path,
                    "action": action,
                    "old_path": old_path,
                    "index_name": index_name,
                },
            )
            self._debounce_timers[path] = timer
            timer.start()

        logger.debug(
            f"Debounce planifié : {action} {path} "
            f"(délai={self._config.debounce_seconds}s)"
        )

    def _fire_job(
        self,
        path: str,
        action: str,
        old_path: Optional[str],
        index_name: str,
    ) -> None:
        """
        Callback déclenché après le délai de debounce.
        Enqueue le job dans la JobQueue.
        """
        with self._debounce_lock:
            self._debounce_timers.pop(path, None)

        if not self._running:
            return

        try:
            job_id = self._queue.enqueue(
                file_path=path,
                action=action,
                old_path=old_path,
                source="watcher",
            )
            # Mettre à jour le job_id dans le dernier événement journalisé
            self._db_conn.execute(
                """
                UPDATE watcher_events
                SET job_id = :job_id, debounced = 1
                WHERE file_path = :path AND job_id IS NULL
                ORDER BY id DESC LIMIT 1
                """,
                {"job_id": job_id, "path": path},
            )
            self._db_conn.commit()
            logger.info(f"Job enqueued après debounce : {action} {path} → job_id={job_id}")

        except Exception as exc:
            logger.error(f"Échec enqueue après debounce : {path} : {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Journalisation
    # ─────────────────────────────────────────────────────────────────────

    def _log_event(
        self,
        event_type: str,
        file_path: str,
        old_path: Optional[str],
        ignored: bool,
        reason: Optional[str],
    ) -> None:
        """Insère un événement dans la table watcher_events."""
        try:
            self._db_conn.execute(
                """
                INSERT INTO watcher_events
                    (event_type, file_path, old_path, timestamp, ignored, reason)
                VALUES
                    (:event_type, :file_path, :old_path, :timestamp, :ignored, :reason)
                """,
                {
                    "event_type": event_type,
                    "file_path": file_path,
                    "old_path": old_path,
                    "timestamp": datetime.utcnow().isoformat(),
                    "ignored": 1 if ignored else 0,
                    "reason": reason,
                },
            )
            self._db_conn.commit()
        except Exception as exc:
            logger.debug(f"Impossible de journaliser l'événement : {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_allowed_extensions(self) -> frozenset[str]:
        """Construit l'ensemble des extensions autorisées depuis la config."""
        if self._config.allowed_extensions:
            return frozenset(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in self._config.allowed_extensions
            )
        return SUPPORTED_DOCUMENT_EXTENSIONS

    def get_pending_debounce_count(self) -> int:
        """Retourne le nombre de timers debounce actifs."""
        with self._debounce_lock:
            return len(self._debounce_timers)
