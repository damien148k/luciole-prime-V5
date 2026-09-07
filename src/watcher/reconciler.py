"""
Reconciler — Scan périodique de réconciliation filesystem / StateStore.

Garantit la cohérence entre ce qui est sur le disque et ce qui est indexé,
même si des événements watchdog ont été manqués (redémarrage conteneur,
problème inotify sur volume réseau, etc.).

Le scan compare :
1. Les fichiers présents sur disque dans les chemins surveillés
2. L'ensemble des documents indexés dans le StateStore

Et enqueue :
- Un job upsert pour tout fichier présent mais non indexé (ou modifié)
- Un job delete pour tout document indexé dont le fichier est absent
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from loguru import logger

from .constants import (
    IGNORED_EXTENSIONS,
    IGNORED_FILENAME_PREFIXES,
    JOB_ACTION_DELETE,
    JOB_ACTION_UPSERT,
    JOB_SOURCE_RESCAN,
    JOB_SOURCE_STARTUP,
    MAX_SCAN_FILES,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from .hashing import quick_hash
from .models import ReconcileReport, WatcherConfig
from .queue import JobQueue
from .state import StateStore


class Reconciler:
    """
    Compare l'état du filesystem aux documents indexés et enqueue les diffs.

    Le scan est conçu pour être léger : il ne lit pas le contenu des fichiers
    (pas de SHA-256 lors du scan), seulement les métadonnées FS (stat).
    La vérification du content_hash est déléguée au worker lors de l'upsert.
    """

    def __init__(
        self,
        config: WatcherConfig,
        queue: JobQueue,
        state: StateStore,
    ) -> None:
        """
        Args:
            config: Configuration du watcher.
            queue: File d'attente pour les jobs générés.
            state: StateStore pour lire les documents indexés.
        """
        self._config = config
        self._queue = queue
        self._state = state

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_report: Optional[ReconcileReport] = None

        self._allowed_extensions: frozenset[str] = self._build_allowed_extensions()

    # ─────────────────────────────────────────────────────────────────────
    # Cycle de vie
    # ─────────────────────────────────────────────────────────────────────

    def start_periodic(self, interval: Optional[int] = None) -> None:
        """
        Démarre le thread de réconciliation périodique.

        Args:
            interval: Intervalle en secondes (utilise la config si None).
        """
        if self._thread and self._thread.is_alive():
            logger.warning("Reconciler déjà démarré")
            return

        self._effective_interval = interval or self._config.reconcile_interval
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_periodic,
            name="luciole-reconciler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Reconciler démarré (intervalle={self._effective_interval}s)"
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Arrête le thread de réconciliation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Reconciler arrêté")

    @property
    def is_running(self) -> bool:
        """Indique si le reconciler est actif."""
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_report(self) -> Optional[ReconcileReport]:
        """Retourne le rapport du dernier scan."""
        return self._last_report

    # ─────────────────────────────────────────────────────────────────────
    # Boucle périodique
    # ─────────────────────────────────────────────────────────────────────

    def _run_periodic(self) -> None:
        """
        Boucle principale : scan immédiat au démarrage, puis périodique.
        """
        logger.debug("Reconciler : première exécution au démarrage")
        self._safe_run_scan(source=JOB_SOURCE_STARTUP)

        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self._effective_interval):
                break
            self._safe_run_scan(source=JOB_SOURCE_RESCAN)

        logger.debug("Reconciler : boucle terminée")

    def _safe_run_scan(self, source: str = JOB_SOURCE_RESCAN) -> None:
        """Exécute un scan en capturant toutes les exceptions."""
        try:
            report = self.run_full_scan(source=source)
            self._last_report = report
            logger.info(
                f"Réconciliation terminée : "
                f"+{report.new_files_enqueued} nouveaux, "
                f"~{report.modified_files_enqueued} modifiés, "
                f"-{report.deleted_files_enqueued} supprimés"
            )
        except Exception as exc:
            logger.error(f"Reconciler : erreur lors du scan : {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Scan principal
    # ─────────────────────────────────────────────────────────────────────

    def run_full_scan(self, source: str = JOB_SOURCE_RESCAN) -> ReconcileReport:
        """
        Effectue un scan complet de réconciliation.

        Étapes :
        1. Collecter les fichiers présents sur disque
        2. Comparer avec le StateStore
        3. Enqueue les diffs (upsert + delete)

        Args:
            source: Origine des jobs générés ('rescan', 'startup', 'manual').

        Returns:
            ReconcileReport avec les statistiques du scan.
        """
        report = ReconcileReport()
        logger.info(f"Réconciliation démarrée (source={source})")

        # 1. Collecter les fichiers sur disque
        disk_files = self._collect_disk_files()
        report.files_on_disk = len(disk_files)

        # 2. Récupérer les chemins indexés depuis le StateStore
        indexed_paths = self._state.get_all_indexed_paths()
        report.files_in_store = len(indexed_paths)

        disk_paths = set(disk_files.keys())

        # 3. Fichiers présents sur disque mais pas (ou plus) indexés
        for path_str, qh in disk_files.items():
            if path_str not in indexed_paths:
                self._queue.enqueue(
                    file_path=path_str,
                    action=JOB_ACTION_UPSERT,
                    source=source,
                    priority=-1,  # Priorité basse : le watcher temps-réel prime
                )
                report.new_files_enqueued += 1
            else:
                # Fichier connu : vérifier si modifié (quick_hash)
                doc = self._state.get_document_by_path(path_str)
                if doc and doc.quick_hash and doc.quick_hash != qh:
                    self._queue.enqueue(
                        file_path=path_str,
                        action=JOB_ACTION_UPSERT,
                        source=source,
                        priority=-1,
                    )
                    report.modified_files_enqueued += 1

        # 4. Documents indexés mais fichiers absents du disque
        for path_str in indexed_paths:
            if path_str not in disk_paths:
                self._queue.enqueue(
                    file_path=path_str,
                    action=JOB_ACTION_DELETE,
                    source=source,
                    priority=-1,
                )
                report.deleted_files_enqueued += 1

        report.finish()
        return report

    # ─────────────────────────────────────────────────────────────────────
    # Collecte des fichiers sur disque
    # ─────────────────────────────────────────────────────────────────────

    def _collect_disk_files(self) -> dict[str, str]:
        """
        Liste tous les fichiers éligibles dans les chemins surveillés.

        Returns:
            Dict {chemin_absolu: quick_hash}
        """
        files: dict[str, str] = {}
        total = 0

        for wp in self._config.watched_paths:
            root = Path(wp.path)
            if not root.exists():
                logger.warning(f"Reconciler : chemin introuvable : {wp.path}")
                continue

            pattern = "**/*" if wp.recursive else "*"
            for file_path in root.glob(pattern):
                if not file_path.is_file():
                    continue
                if total >= MAX_SCAN_FILES:
                    logger.warning(
                        f"Reconciler : limite de {MAX_SCAN_FILES} fichiers atteinte"
                    )
                    break
                if not self._is_eligible(file_path):
                    continue

                path_str = str(file_path.resolve())
                try:
                    qh = quick_hash(file_path)
                    files[path_str] = qh
                    total += 1
                except OSError as exc:
                    logger.debug(f"Reconciler : fichier ignoré ({exc}) : {file_path}")

        logger.debug(f"Reconciler : {total} fichier(s) collecté(s) sur disque")
        return files

    def _is_eligible(self, path: Path) -> bool:
        """Vérifie si un fichier doit être considéré lors du scan."""
        ext = path.suffix.lower()

        # Extensions ignorées (temporaires, système)
        if ext in IGNORED_EXTENSIONS:
            return False

        # Extensions non autorisées
        if ext and ext not in self._allowed_extensions:
            return False

        # Préfixes de noms temporaires
        for prefix in IGNORED_FILENAME_PREFIXES:
            if path.name.startswith(prefix):
                return False

        # Taille max
        try:
            size = path.stat().st_size
            if size > self._config.max_file_size_mb * 1024 * 1024:
                return False
        except OSError:
            return False

        return True

    def _build_allowed_extensions(self) -> frozenset[str]:
        """Construit l'ensemble des extensions autorisées."""
        if self._config.allowed_extensions:
            return frozenset(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in self._config.allowed_extensions
            )
        return SUPPORTED_DOCUMENT_EXTENSIONS
