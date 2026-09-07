"""
WatcherService — Orchestrateur principal du service watcher.

Initialise et coordonne tous les composants :
- FileWatcher (observer)
- JobQueue (queue SQLite)
- StateStore (état des documents)
- ChunkCleaner (suppression Qdrant/OpenSearch)
- IndexWorker (traitement des jobs)
- Reconciler (scan périodique)

Une seule instance de WatcherService est créée au démarrage de l'application.
Elle expose des méthodes de contrôle utilisables par l'API FastAPI.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from loguru import logger

from .cleanup import ChunkCleaner
from .config import load_watcher_config
from .constants import LOCK_ERRORS_WARN_THRESHOLD
from .db import init_db
from .models import WatcherConfig
from .observer import FileWatcher
from .queue import JobQueue
from .reconciler import Reconciler
from .state import StateStore
from .worker import IndexWorker


class WatcherService:
    """
    Point d'entrée unique du service watcher.

    Instanciée au démarrage de l'application FastAPI (lifespan).
    Tous les composants sont créés ici et partagent la même configuration.
    """

    def __init__(
        self,
        config: Optional[WatcherConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: Instance WatcherConfig directe (tests, injection).
            config_path: Chemin vers settings.yaml (prioritaire si config=None).
        """
        self._config = config or load_watcher_config(config_path)
        self._started_at: Optional[str] = None
        self._rescan_count: int = 0

        # Composants initialisés dans start()
        self._db_conn = None
        self._queue: Optional[JobQueue] = None
        self._state: Optional[StateStore] = None
        self._cleaner: Optional[ChunkCleaner] = None
        self._watcher: Optional[FileWatcher] = None
        self._worker: Optional[IndexWorker] = None
        self._reconciler: Optional[Reconciler] = None

    # ─────────────────────────────────────────────────────────────────────
    # Cycle de vie
    # ─────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Démarre tous les composants du service watcher.

        Appelé depuis le lifespan FastAPI ou en standalone.
        """
        if not self._config.enabled:
            logger.info("WatcherService désactivé (WATCHER_ENABLED=false)")
            return

        logger.info("WatcherService : démarrage...")

        # 1. Initialiser la base SQLite
        conn = init_db(self._config.db_path)
        self._db_conn = conn

        # 2. Instancier les composants de données
        self._queue = JobQueue(
            db_path=self._config.db_path,
            retry_max_attempts=self._config.retry_max_attempts,
            retry_backoff_base=self._config.retry_backoff_base,
        )
        self._state = StateStore(db_path=self._config.db_path)

        # 3. Initialiser les clients vector stores
        qdrant = self._build_qdrant_client()
        opensearch = self._build_opensearch_client()
        self._cleaner = ChunkCleaner(qdrant=qdrant, opensearch=opensearch)

        # 4. Factory du pipeline d'ingestion (lazy — chargé au premier job)
        pipeline_factory = self._build_pipeline_factory()

        # 5. IndexWorker
        self._worker = IndexWorker(
            queue=JobQueue(db_path=self._config.db_path),
            state=StateStore(db_path=self._config.db_path),
            cleaner=self._cleaner,
            pipeline_factory=pipeline_factory,
            db_path=self._config.db_path,
            default_index_name=self._config.default_index_name,
            stability_checks=self._config.stability_checks,
            stability_interval=self._config.stability_interval,
            poll_interval=self._config.worker_poll_interval,
            config=self._config,
        )

        # 6. FileWatcher
        self._watcher = FileWatcher(
            config=self._config,
            queue=self._queue,
            db_conn=self._db_conn,
        )

        # 7. Reconciler
        self._reconciler = Reconciler(
            config=self._config,
            queue=JobQueue(db_path=self._config.db_path),
            state=StateStore(db_path=self._config.db_path),
        )

        # 8. Démarrer les threads
        self._worker.start()
        self._watcher.start()
        self._reconciler.start_periodic()

        self._started_at = datetime.utcnow().isoformat()
        logger.info("WatcherService démarré")

    async def stop(self) -> None:
        """
        Arrête tous les composants de manière gracieuse.

        Appelé depuis le lifespan FastAPI (shutdown) ou en standalone.
        """
        logger.info("WatcherService : arrêt en cours...")

        if self._watcher:
            self._watcher.stop()
        if self._worker:
            self._worker.stop()
        if self._reconciler:
            self._reconciler.stop()

        if self._state:
            self._state.close()
        if self._queue:
            self._queue.close()
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass

        logger.info("WatcherService arrêté")

    @property
    def is_running(self) -> bool:
        """Indique si le service est opérationnel."""
        return bool(
            self._started_at
            and self._worker
            and self._worker.is_running
        )

    # ─────────────────────────────────────────────────────────────────────
    # Contrôle via l'API
    # ─────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Retourne l'état complet du service (pour GET /api/watcher/status).
        """
        if not self._started_at:
            return {"running": False, "message": "Service non démarré"}

        uptime = 0
        if self._started_at:
            started = datetime.fromisoformat(self._started_at)
            uptime = int((datetime.utcnow() - started).total_seconds())

        queue_counts = self._queue.get_counts_by_status() if self._queue else {}
        doc_counts = self._state.get_documents_count_by_status() if self._state else {}

        # Un worker dont le thread est vivant mais qui ne parvient plus à
        # écrire doit être visible ici, pas seulement dans les logs.
        lock_errors = self._worker.lock_errors if self._worker else 0
        degraded_reason = None
        if lock_errors >= LOCK_ERRORS_WARN_THRESHOLD:
            degraded_reason = (
                f"worker bloqué : {lock_errors} verrous SQLite consécutifs"
            )

        return {
            "running": self.is_running,
            "degraded": degraded_reason is not None,
            "degraded_reason": degraded_reason,
            "started_at": self._started_at,
            "uptime_seconds": uptime,
            "watched_paths": [
                {
                    "path": wp.path,
                    "recursive": wp.recursive,
                    "index_name": wp.index_name or self._config.default_index_name,
                }
                for wp in self._config.watched_paths
            ],
            "components": {
                "watcher": self._watcher.is_running if self._watcher else False,
                "worker": self._worker.is_running if self._worker else False,
                "reconciler": self._reconciler.is_running if self._reconciler else False,
            },
            "queue": {
                "pending": queue_counts.get("pending", 0),
                "in_progress": queue_counts.get("in_progress", 0),
                "failed": queue_counts.get("failed", 0),
                "dead": queue_counts.get("dead", 0),
                "completed": queue_counts.get("completed", 0),
                "lock_errors": lock_errors,
            },
            "documents": {
                "indexed": doc_counts.get("indexed", 0),
                "pending": doc_counts.get("pending", 0),
                "error": doc_counts.get("error", 0),
                "deleted": doc_counts.get("deleted", 0),
            },
            "debounce_active": (
                self._watcher.get_pending_debounce_count()
                if self._watcher
                else 0
            ),
            "reconciler": {
                "last_report": (
                    {
                        "run_id": self._reconciler.last_report.run_id,
                        "finished_at": self._reconciler.last_report.finished_at,
                        "total_jobs_created": self._reconciler.last_report.total_jobs_created,
                    }
                    if self._reconciler and self._reconciler.last_report
                    else None
                )
            },
        }

    def trigger_rescan(self, path: Optional[str] = None) -> str:
        """
        Lance un scan manuel de réconciliation.

        Args:
            path: Si fourni, limite le scan à ce chemin (non implémenté en V1,
                  le scan couvre toujours tous les chemins configurés).

        Returns:
            run_id du rapport de réconciliation.
        """
        if not self._reconciler:
            raise RuntimeError("Reconciler non initialisé")

        from .constants import JOB_SOURCE_MANUAL
        report = self._reconciler.run_full_scan(source=JOB_SOURCE_MANUAL)
        self._rescan_count += 1

        logger.info(
            f"Rescan manuel : run_id={report.run_id} | "
            f"jobs créés={report.total_jobs_created}"
        )
        return report.run_id

    # ─────────────────────────────────────────────────────────────────────
    # Accès aux composants (pour l'API)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def queue(self) -> Optional[JobQueue]:
        """Accès à la JobQueue pour les endpoints API."""
        return self._queue

    @property
    def state(self) -> Optional[StateStore]:
        """Accès au StateStore pour les endpoints API."""
        return self._state

    @property
    def config(self) -> WatcherConfig:
        """Accès à la configuration."""
        return self._config

    # ─────────────────────────────────────────────────────────────────────
    # Builders internes
    # ─────────────────────────────────────────────────────────────────────

    def _build_qdrant_client(self):
        """Crée le client Qdrant depuis les variables d'environnement ou la config."""
        from qdrant_client import QdrantClient

        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_url:
            logger.debug(f"Qdrant : connexion via QDRANT_URL={qdrant_url}")
            return QdrantClient(url=qdrant_url)

        import yaml
        config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
        try:
            with open(config_path, encoding="utf-8") as fh:
                full = yaml.safe_load(fh) or {}
            host = full.get("qdrant", {}).get("host", "localhost")
            port = full.get("qdrant", {}).get("port", 6333)
        except Exception:
            host, port = "localhost", 6333

        logger.debug(f"Qdrant : connexion via config ({host}:{port})")
        return QdrantClient(host=host, port=port)

    def _build_opensearch_client(self):
        """Crée le client OpenSearch depuis les variables d'environnement ou la config."""
        from opensearchpy import OpenSearch

        opensearch_url = os.environ.get("OPENSEARCH_URL")
        if opensearch_url:
            from urllib.parse import urlparse
            parsed = urlparse(opensearch_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 9200
        else:
            import yaml
            config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
            try:
                with open(config_path, encoding="utf-8") as fh:
                    full = yaml.safe_load(fh) or {}
                host = full.get("opensearch", {}).get("host", "localhost")
                port = full.get("opensearch", {}).get("port", 9200)
            except Exception:
                host, port = "localhost", 9200

        logger.debug(f"OpenSearch : connexion via {host}:{port}")
        return OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
        )

    def _build_pipeline_factory(self):
        """
        Retourne une factory callable qui crée un IngestionPipeline.

        L'instanciation est différée (lazy) pour ne pas bloquer le démarrage
        si les modèles d'embedding sont lents à charger.
        """
        config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
        index_name = self._config.default_index_name

        def factory():
            from src.ingestion.pipeline import IngestionPipeline
            return IngestionPipeline(
                config_path=config_path,
                index_name=index_name,
                enable_tracking=False,  # Le watcher gère son propre state store
            )

        return factory
