"""
IndexWorker — Thread de traitement des jobs d'indexation.

Le worker tourne dans un thread dédié et consomme la JobQueue en boucle.
Il n'effectue aucune observation du filesystem — c'est le rôle du FileWatcher.

Pour chaque job :
- upsert : vérifie la stabilité du fichier, compare le content_hash,
           supprime les anciens chunks par document_id, indexe via IngestionPipeline
- delete : supprime les chunks par document_id, soft-delete dans le StateStore
- move   : met à jour current_path dans le StateStore et les payloads vector stores

Note d'identité :
- `source_id` (UUID) est l'identifiant stable INTERNE au watcher (StateStore SQLite).
- `document_id` (= filename) est la clé d'identité dans les payloads Qdrant/OpenSearch,
  alignée avec l'ingestion (cf. rag-system/src/ingestion/chunker.py).
- Les opérations sur les vector stores passent par `document_id`, pas par `source_id`.

Le IngestionPipeline existant (`src.ingestion.pipeline`) est réutilisé tel quel.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from .cleanup import ChunkCleaner
from .constants import (
    DELETION_REASON_FILE_REMOVED,
    JOB_ACTION_DELETE,
    JOB_ACTION_MOVE,
    JOB_ACTION_UPSERT,
    STABILITY_CHECKS,
    STABILITY_INTERVAL,
    WORKER_POLL_INTERVAL,
)
from .db import get_connection, log_audit_error
from .exceptions import CleanupError, FileNotStableError, IndexingError
from .hashing import content_hash, quick_hash, wait_stable
from .index_routing import derive_index_name
from .models import DocumentState, DocumentStatus, Job, WatcherConfig
from .queue import JobQueue
from .state import StateStore


class IndexWorker:
    """
    Traite les jobs d'indexation depuis la JobQueue.

    Instancié par WatcherService. Possède ses propres connexions SQLite
    (StateStore et JobQueue distincts de ceux du FileWatcher).
    """

    def __init__(
        self,
        queue: JobQueue,
        state: StateStore,
        cleaner: ChunkCleaner,
        pipeline_factory: Callable,
        db_path: str,
        default_index_name: str = "documents",
        stability_checks: int = STABILITY_CHECKS,
        stability_interval: float = STABILITY_INTERVAL,
        poll_interval: float = WORKER_POLL_INTERVAL,
        config: Optional[WatcherConfig] = None,
    ) -> None:
        """
        Args:
            queue: File d'attente des jobs.
            state: StateStore pour la persistance des documents.
            cleaner: ChunkCleaner pour la suppression dans les vector stores.
            pipeline_factory: Callable() → IngestionPipeline (lazy init).
            db_path: Chemin vers watcher.db (pour log_audit_error).
            default_index_name: Nom d'index utilisé si aucune dérivation n'est possible
                (fichier hors watched_path, ou sans sous-dossier projet).
            stability_checks: Nb de vérifications de stabilité fichier.
            stability_interval: Intervalle entre vérifications (s).
            poll_interval: Fréquence de poll de la queue (s).
            config: Configuration du watcher (pour accéder à `watched_paths`).
                Si None, fallback systématique sur `default_index_name`.
        """
        self._queue = queue
        self._state = state
        self._cleaner = cleaner
        self._pipeline_factory = pipeline_factory
        self._db_path = db_path
        self._default_index_name = default_index_name
        self._stability_checks = stability_checks
        self._stability_interval = stability_interval
        self._poll_interval = poll_interval
        self._config = config

        self._pipeline = None          # Chargé au premier usage
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audit_conn = get_connection(db_path)

    # ─────────────────────────────────────────────────────────────────────
    # Cycle de vie
    # ─────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Démarre le thread de traitement des jobs."""
        if self._thread and self._thread.is_alive():
            logger.warning("IndexWorker déjà démarré")
            return

        recovered = self._queue.recover_in_progress()
        if recovered:
            logger.info(f"Worker : {recovered} job(s) récupéré(s) après crash")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="luciole-index-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("IndexWorker démarré")

    def stop(self, timeout: float = 30.0) -> None:
        """
        Arrête le thread de traitement de manière gracieuse.

        Args:
            timeout: Délai d'attente maximal pour la fin du job en cours (s).
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("IndexWorker : thread non terminé après timeout")
        logger.info("IndexWorker arrêté")

    @property
    def is_running(self) -> bool:
        """Indique si le worker est actif.

        Attention : un thread vivant ne garantit pas que le worker progresse.
        Voir `lock_errors` pour détecter une boucle qui tourne à vide sur une
        base verrouillée.
        """
        return bool(self._thread and self._thread.is_alive())

    @property
    def lock_errors(self) -> int:
        """Verrous SQLite subis d'affilée par la boucle de dépilement."""
        return self._queue.consecutive_lock_errors if self._queue else 0

    # ─────────────────────────────────────────────────────────────────────
    # Boucle principale
    # ─────────────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Boucle de traitement des jobs.

        Poll la queue toutes les `poll_interval` secondes.
        S'arrête proprement sur le signal `_stop_event`.
        """
        logger.debug("IndexWorker : boucle démarrée")

        # Pré-charger le pipeline d'ingestion au démarrage (évite la latence au 1er job).
        # Non bloquant : si le chargement échoue, le worker continue et réessaiera au 1er job.
        try:
            self._ensure_pipeline()
        except Exception as exc:
            logger.warning(
                f"IndexWorker : pré-chargement pipeline échoué ({exc}), "
                "sera retentée au premier job."
            )
            self._pipeline = None

        while not self._stop_event.is_set():
            job = self._queue.dequeue()

            if job is None:
                self._stop_event.wait(timeout=self._poll_interval)
                continue

            self._process_job(job)

        logger.debug("IndexWorker : boucle terminée")

    def _process_job(self, job: Job) -> None:
        """
        Dispatche un job vers le handler approprié.

        Gère les erreurs et met à jour le statut du job dans la queue.
        """
        logger.info(
            f"Traitement job : {job.job_id} | action={job.action} | {job.file_path}"
        )

        try:
            if job.action == JOB_ACTION_UPSERT:
                self._handle_upsert(job)
            elif job.action == JOB_ACTION_DELETE:
                self._handle_delete(job)
            elif job.action == JOB_ACTION_MOVE:
                self._handle_move(job)
            else:
                raise ValueError(f"Action inconnue : {job.action}")

            self._queue.complete(job.job_id)
            logger.info(f"Job terminé : {job.job_id}")

        except FileNotStableError as exc:
            self._handle_job_error(job, exc, "file_stability_error")
        except CleanupError as exc:
            self._handle_job_error(job, exc, "cleanup_error")
        except IndexingError as exc:
            self._handle_job_error(job, exc, "indexing_error")
        except Exception as exc:
            self._handle_job_error(job, exc, "unexpected_error")

    def _handle_job_error(self, job: Job, exc: Exception, error_type: str) -> None:
        """Journalise l'erreur et met à jour le statut du job."""
        tb = traceback.format_exc()
        msg = str(exc)
        logger.error(f"Erreur job {job.job_id} [{error_type}] : {msg}")

        log_audit_error(
            conn=self._audit_conn,
            error_type=error_type,
            error_message=msg,
            job_id=job.job_id,
            file_path=job.file_path,
            stack_trace=tb,
        )
        self._queue.fail(job.job_id, msg)

    # ─────────────────────────────────────────────────────────────────────
    # Handlers par action
    # ─────────────────────────────────────────────────────────────────────

    def _handle_upsert(self, job: Job) -> None:
        """
        Traite un job d'indexation (create ou modify).

        Flux :
        1. Vérifier la stabilité du fichier (copie en cours ?)
        2. Calculer quick_hash + content_hash
        3. Comparer au content_hash stocké → skip si identique
        4. Supprimer les anciens chunks par document_id (= filename)
        5. Indexer via IngestionPipeline
        6. Mettre à jour le StateStore
        """
        path = Path(job.file_path)

        if not path.exists():
            logger.warning(f"upsert : fichier disparu avant traitement : {path}")
            return

        # Vérifier que le fichier a fini d'être écrit
        stable = wait_stable(
            path,
            checks=self._stability_checks,
            interval=self._stability_interval,
        )
        if not stable:
            raise FileNotStableError(
                f"Fichier instable après {self._stability_checks} vérifications : {path}"
            )

        qh = quick_hash(path)
        ch = content_hash(path)

        # Chercher un document existant pour ce path (ou créer un nouveau)
        doc = self._state.get_document_by_path(str(path))

        if doc is None:
            # Première découverte — générer un source_id stable
            resolved_index = self._resolve_index_name(str(path))
            doc = DocumentState(
                current_path=str(path),
                file_name=path.name,
                file_extension=path.suffix.lower(),
                index_name=resolved_index,
            )
            logger.info(
                f"Nouveau document découvert : {path} → "
                f"source_id={doc.source_id} | index={resolved_index}"
            )
        else:
            # Vérifier si le contenu a réellement changé
            if doc.content_hash == ch and doc.status == DocumentStatus.INDEXED:
                # Mettre à jour quick_hash (mtime/taille peuvent avoir changé)
                doc.quick_hash = qh
                doc.touch()
                self._state.upsert_document(doc)
                logger.debug(f"Contenu inchangé, skip réindexation : {path}")
                return

        # Supprimer les anciens chunks AVANT réindexation
        # Note : on filtre par document_id (= filename), aligné avec l'ingestion.
        if doc.content_hash:
            try:
                self._cleaner.delete_document_chunks(doc.file_name, doc.index_name)
            except CleanupError as exc:
                # Non bloquant pour un nouveau document (pas encore de chunks)
                if doc.version > 0:
                    raise
                logger.debug(f"Pas de chunks existants à supprimer : {exc}")

        # Indexer via le pipeline existant
        pipeline = self._ensure_pipeline()
        try:
            result = pipeline.ingest_file(
                file_path=str(path),
                skip_if_indexed=False,
                index_name=doc.index_name,
            )
        except Exception as exc:
            raise IndexingError(
                f"Échec indexation {path} : {exc}"
            ) from exc

        if result.get("status") == "error":
            raise IndexingError(
                f"Pipeline retourne erreur pour {path} : {result.get('error')}"
            )

        # Mettre à jour le StateStore
        chunk_count = result.get("chunks", 0)
        now = datetime.utcnow().isoformat()

        doc.file_size_bytes  = path.stat().st_size
        doc.quick_hash       = qh
        doc.content_hash     = ch
        doc.chunk_count      = chunk_count
        doc.status           = DocumentStatus.INDEXED
        doc.last_indexed_at  = now
        doc.last_error       = None
        doc.retry_count      = 0
        doc.version         += 1

        if not doc.first_indexed_at:
            doc.first_indexed_at = now

        self._state.upsert_document(doc)

        logger.info(
            f"Indexation OK : {path} | "
            f"source_id={doc.source_id} | document_id={doc.file_name} | "
            f"chunks={chunk_count} | v{doc.version}"
        )

    def _handle_delete(self, job: Job) -> None:
        """
        Traite un job de suppression.

        Flux :
        1. Récupérer le document depuis le StateStore
        2. Supprimer les chunks par document_id (= filename) dans Qdrant + OpenSearch
        3. Soft-delete dans le StateStore (tombstone) via source_id (clé interne)

        Fallback : si le document est inconnu du StateStore (cas d'une
        désynchronisation, p.ex. ingestion faite avec watcher arrêté),
        on tente quand même de purger les vector stores en utilisant le
        basename du fichier supprimé.
        """
        doc = self._state.get_document_by_path(job.file_path)

        if doc is None:
            # Fallback : purge "best effort" par filename, même sans entry DB.
            # Couvre le cas où l'indexation a été faite hors watcher.
            file_name = Path(job.file_path).name
            logger.warning(
                f"delete : document inconnu du StateStore, "
                f"purge fallback par filename : {file_name}"
            )
            fallback_index = self._resolve_index_name(job.file_path)
            try:
                self._cleaner.delete_document_chunks(
                    file_name, fallback_index
                )
            except CleanupError as exc:
                logger.error(f"Purge fallback échouée pour {file_name} : {exc}")
                raise
            return

        # Supprimer les chunks dans les vector stores via document_id (= filename)
        try:
            self._cleaner.delete_document_chunks(doc.file_name, doc.index_name)
        except CleanupError:
            raise

        # Soft-delete dans le StateStore (clé interne = source_id)
        self._state.mark_deleted(
            source_id=doc.source_id,
            reason=DELETION_REASON_FILE_REMOVED,
        )

        logger.info(
            f"Suppression OK : {job.file_path} | "
            f"source_id={doc.source_id} | document_id={doc.file_name}"
        )

    def _handle_move(self, job: Job) -> None:
        """
        Traite un job de déplacement/renommage.

        Flux :
        1. Récupérer le document depuis le StateStore (par old_path)
        2. Mettre à jour current_path dans le StateStore
        3. Mettre à jour file_path/file_name dans les payloads Qdrant/OpenSearch
        4. Aucun ré-embedding : le contenu n'a pas changé

        Si le document est inconnu du StateStore (move d'un fichier non encore
        indexé), le traite comme un upsert sur le nouveau chemin.

        Note : si le filename change lors du move, le `document_id` dans les
        payloads des vector stores ne correspondra plus au nouveau nom. Le
        update_file_path met à jour file_path et file_name mais PAS document_id
        (qui reste l'identifiant historique de l'indexation). Un futur delete
        utilisera doc.file_name côté StateStore — donc on met à jour file_name
        dans le StateStore mais on garde aussi un alias vers l'ancien document_id
        si nécessaire (TODO si on observe des problèmes).
        """
        old_path = job.old_path or job.file_path
        new_path = job.file_path

        doc = self._state.get_document_by_path(old_path)

        if doc is None:
            # Fichier déplacé mais jamais indexé → traiter comme un nouveau fichier
            logger.info(
                f"move : source inconnu, traitement comme upsert : {old_path} → {new_path}"
            )
            self._handle_upsert(job)
            return

        # `document_id` dans les payloads = filename d'origine (au moment de
        # l'indexation). On l'utilise comme clé pour retrouver les chunks.
        original_document_id = doc.file_name

        # Mettre à jour le chemin dans le StateStore (source_id inchangé)
        self._state.move_document(
            source_id=doc.source_id,
            new_path=new_path,
        )

        # Mettre à jour les métadonnées d'affichage dans les vector stores
        # (le filtre se fait sur l'ancien document_id pour retrouver les points)
        self._cleaner.update_file_path(
            document_id=original_document_id,
            new_path=new_path,
            index_name=doc.index_name,
        )

        logger.info(
            f"Move OK : {old_path} → {new_path} | "
            f"source_id={doc.source_id} | document_id={original_document_id}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline d'ingestion
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_index_name(self, file_path: str) -> str:
        """
        Dérive l'index_name pour un fichier donné à partir de son chemin.

        Utilise `derive_index_name()` qui :
        - extrait le 1er sous-dossier sous le watched_path comme nom d'index
        - sanitise (lowercase, [a-zA-Z0-9_-], 64 chars max)
        - fallback sur watched_path.index_name puis self._default_index_name

        Si `self._config` est None (ancien appel sans config), fallback direct
        sur le default_index_name pour rester rétrocompatible.
        """
        if self._config is None:
            return self._default_index_name
        return derive_index_name(
            file_path=file_path,
            watched_paths=self._config.watched_paths,
            default_index_name=self._default_index_name,
        )

    def _ensure_pipeline(self):
        """
        Retourne le pipeline d'ingestion, en le créant si nécessaire.

        Le pipeline est chargé au premier accès (lazy init) pour ne pas
        bloquer le démarrage du watcher si les modèles d'embedding sont lents.
        """
        if self._pipeline is None:
            logger.info("IndexWorker : chargement du pipeline d'ingestion...")
            try:
                self._pipeline = self._pipeline_factory()
                logger.info("IndexWorker : pipeline chargé")
            except Exception as exc:
                logger.error(f"Impossible de charger le pipeline : {exc}")
                raise IndexingError(f"Échec initialisation pipeline : {exc}") from exc
        return self._pipeline