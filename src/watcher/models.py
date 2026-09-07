"""
Modèles de données du service watcher.

Tous les modèles sont des dataclasses ou Pydantic BaseModel selon leur usage :
- Pydantic pour la configuration (validation au démarrage)
- Dataclasses pour les objets de données internes (performance)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Énumérations
# ─────────────────────────────────────────────────────────────────────────────


class DocumentStatus(str, Enum):
    """Statut d'un document dans le StateStore."""
    PENDING = "pending"
    INDEXED = "indexed"
    ERROR   = "error"
    DELETED = "deleted"


class JobStatus(str, Enum):
    """Statut d'un job dans la JobQueue."""
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    DEAD        = "dead"


class JobAction(str, Enum):
    """Action portée par un job."""
    UPSERT = "upsert"
    DELETE = "delete"
    MOVE   = "move"


class JobSource(str, Enum):
    """Origine d'un job."""
    WATCHER = "watcher"
    RESCAN  = "rescan"
    MANUAL  = "manual"
    STARTUP = "startup"


class DeletionReason(str, Enum):
    """Raison d'une suppression de document."""
    FILE_REMOVED = "file_removed"
    MANUAL       = "manual"
    RECONCILE    = "reconcile"


class WatcherEventType(str, Enum):
    """Type d'événement filesystem."""
    CREATED  = "created"
    MODIFIED = "modified"
    MOVED    = "moved"
    DELETED  = "deleted"


# ─────────────────────────────────────────────────────────────────────────────
# Objets de données internes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DocumentState:
    """
    Représente l'état d'un document dans le StateStore.

    `source_id` est l'identifiant stable généré à la première découverte.
    Il ne change jamais, même après un rename ou un déplacement.
    `current_path` est l'attribut modifiable qui reflète la localisation actuelle.
    """

    # Identité stable — généré une seule fois
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Localisation actuelle (modifiable par un move/rename)
    current_path: str = ""
    file_name: str = ""
    file_extension: str = ""

    # Métadonnées fichier
    file_size_bytes: int = 0
    quick_hash: str = ""      # hash rapide : taille + mtime
    content_hash: str = ""    # SHA-256 du contenu (décision d'indexation)

    # Indexation
    chunk_count: int = 0
    index_name: str = "documents"
    status: DocumentStatus = DocumentStatus.PENDING

    # Soft-delete
    deleted_at: Optional[str] = None
    deletion_reason: Optional[str] = None

    # Audit et reprise
    first_indexed_at: Optional[str] = None
    last_indexed_at: Optional[str] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    version: int = 0

    # Timestamps internes
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_deleted(self) -> bool:
        """Indique si le document est en soft-delete."""
        return self.status == DocumentStatus.DELETED

    def touch(self) -> None:
        """Met à jour `updated_at` après toute modification."""
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class Job:
    """
    Représente un job dans la file d'attente d'indexation.
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    action: str = JobAction.UPSERT.value
    old_path: Optional[str] = None
    status: str = JobStatus.PENDING.value
    priority: int = 0
    source: str = JobSource.WATCHER.value
    attempts: int = 0
    max_attempts: int = 3
    next_retry_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class WatcherEvent:
    """
    Représente un événement filesystem journalisé.
    """
    event_type: str
    file_path: str
    old_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    debounced: bool = False
    job_id: Optional[str] = None
    ignored: bool = False
    reason: Optional[str] = None


@dataclass
class ReconcileReport:
    """
    Rapport produit par un scan de réconciliation.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: Optional[str] = None
    files_on_disk: int = 0
    files_in_store: int = 0
    new_files_enqueued: int = 0
    deleted_files_enqueued: int = 0
    modified_files_enqueued: int = 0
    errors: list[str] = field(default_factory=list)

    def finish(self) -> None:
        self.finished_at = datetime.utcnow().isoformat()

    @property
    def total_jobs_created(self) -> int:
        return self.new_files_enqueued + self.deleted_files_enqueued + self.modified_files_enqueued


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Pydantic (validation au démarrage)
# ─────────────────────────────────────────────────────────────────────────────


class WatchedPath(BaseModel):
    """
    Définit un chemin à surveiller, avec ses options propres.
    """
    path: str
    recursive: bool = True
    index_name: Optional[str] = None     # Si None, utilise l'index par défaut

    @field_validator("path")
    @classmethod
    def path_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le chemin surveillé ne peut pas être vide.")
        return v.strip()


class WatcherConfig(BaseModel):
    """
    Configuration complète du service watcher.
    Chargée depuis la section `watcher:` de settings.yaml.
    """
    enabled: bool = True
    watched_paths: list[WatchedPath] = []

    # Filesystem
    polling_interval: float = 5.0
    debounce_seconds: float = 3.0
    stability_checks: int = 3
    stability_interval: float = 2.0

    # Worker
    max_worker_threads: int = 1
    worker_poll_interval: float = 1.0

    # Réconciliation
    reconcile_interval: int = 300
    reconcile_on_startup: bool = True

    # Filtres
    allowed_extensions: list[str] = []      # Si vide, utilise SUPPORTED_DOCUMENT_EXTENSIONS
    excluded_dirs: list[str] = []
    excluded_extensions: list[str] = []
    max_file_size_mb: int = 9999

    # Retry
    retry_max_attempts: int = 3
    retry_backoff_base: float = 60.0

    # Base de données
    db_path: str = "/app/backups/watcher/watcher.db"

    # API
    api_port: int = 8090

    # Index par défaut (fallback si WatchedPath.index_name est None)
    default_index_name: str = "documents"
