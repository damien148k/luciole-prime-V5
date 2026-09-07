"""
StateStore — Persistance de l'état des documents indexés.

Toutes les opérations CRUD sur la table `documents` passent par ce module.
L'identifiant principal est `source_id` (UUID stable).
`current_path` est l'attribut modifiable mis à jour lors des moves/renames.

Les suppressions sont des soft-deletes : la ligne reste en base avec
`status='deleted'`, `deleted_at` et `deletion_reason` pour l'audit.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from loguru import logger

from .db import get_connection
from .models import DocumentState, DocumentStatus


class StateStore:
    """
    Gère la persistance et la consultation de l'état des documents.

    Maintient une connexion SQLite par instance (une instance par thread).
    """

    def __init__(self, db_path: str) -> None:
        """
        Args:
            db_path: Chemin vers watcher.db (doit exister — appeler init_db() avant).
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection = get_connection(db_path)
        logger.debug(f"StateStore connecté : {db_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Lecture
    # ─────────────────────────────────────────────────────────────────────

    def get_document_by_source_id(self, source_id: str) -> Optional[DocumentState]:
        """
        Récupère un document par son identifiant stable.

        Args:
            source_id: UUID du document.

        Returns:
            DocumentState ou None si introuvable.
        """
        row = self._conn.execute(
            "SELECT * FROM documents WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return _row_to_doc(row) if row else None

    def get_document_by_path(self, path: str) -> Optional[DocumentState]:
        """
        Récupère un document par son chemin actuel.

        Ignore les documents soft-deleted (status='deleted').

        Args:
            path: Chemin absolu normalisé du fichier.

        Returns:
            DocumentState ou None si introuvable.
        """
        row = self._conn.execute(
            "SELECT * FROM documents WHERE current_path = ? AND status != 'deleted'",
            (path,),
        ).fetchone()
        return _row_to_doc(row) if row else None

    def get_documents_by_status(self, status: str) -> list[DocumentState]:
        """
        Retourne tous les documents ayant le statut donné.

        Args:
            status: Valeur de DocumentStatus (ex. 'indexed', 'error').

        Returns:
            Liste de DocumentState.
        """
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE status = ? ORDER BY updated_at DESC",
            (status,),
        ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_all_indexed_paths(self) -> set[str]:
        """
        Retourne l'ensemble des chemins actuellement indexés (status='indexed').

        Utilisé par le Reconciler pour détecter les fichiers supprimés ou nouveaux.

        Returns:
            Ensemble de chemins absolus.
        """
        rows = self._conn.execute(
            "SELECT current_path FROM documents WHERE status = 'indexed'",
        ).fetchall()
        return {row["current_path"] for row in rows}

    def get_documents_count_by_status(self) -> dict[str, int]:
        """
        Retourne le nombre de documents par statut.

        Returns:
            Dict {statut: nombre}.
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM documents GROUP BY status",
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def search_documents(
        self,
        search: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DocumentState]:
        """
        Recherche paginée dans les documents.

        Args:
            search: Filtre sur file_name (sous-chaîne, insensible à la casse).
            status: Filtre sur le statut.
            limit: Nombre maximum de résultats.
            offset: Décalage pour la pagination.

        Returns:
            Liste de DocumentState.
        """
        conditions = []
        params: list = []

        if search:
            conditions.append("LOWER(file_name) LIKE LOWER(?)")
            params.append(f"%{search}%")
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT * FROM documents {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_doc(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────
    # Écriture
    # ─────────────────────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentState) -> None:
        """
        Crée ou met à jour un document dans le StateStore.

        Utilise un INSERT OR REPLACE basé sur source_id (UNIQUE).
        Met automatiquement à jour `updated_at`.

        Args:
            doc: DocumentState à persister.
        """
        doc.touch()
        self._conn.execute(
            """
            INSERT INTO documents (
                source_id, current_path, file_name, file_extension,
                file_size_bytes, quick_hash, content_hash,
                chunk_count, index_name, status,
                deleted_at, deletion_reason,
                first_indexed_at, last_indexed_at,
                last_error, retry_count, version,
                created_at, updated_at
            ) VALUES (
                :source_id, :current_path, :file_name, :file_extension,
                :file_size_bytes, :quick_hash, :content_hash,
                :chunk_count, :index_name, :status,
                :deleted_at, :deletion_reason,
                :first_indexed_at, :last_indexed_at,
                :last_error, :retry_count, :version,
                :created_at, :updated_at
            )
            ON CONFLICT(source_id) DO UPDATE SET
                current_path     = excluded.current_path,
                file_name        = excluded.file_name,
                file_extension   = excluded.file_extension,
                file_size_bytes  = excluded.file_size_bytes,
                quick_hash       = excluded.quick_hash,
                content_hash     = excluded.content_hash,
                chunk_count      = excluded.chunk_count,
                index_name       = excluded.index_name,
                status           = excluded.status,
                deleted_at       = excluded.deleted_at,
                deletion_reason  = excluded.deletion_reason,
                first_indexed_at = excluded.first_indexed_at,
                last_indexed_at  = excluded.last_indexed_at,
                last_error       = excluded.last_error,
                retry_count      = excluded.retry_count,
                version          = excluded.version,
                updated_at       = excluded.updated_at
            """,
            {
                "source_id":        doc.source_id,
                "current_path":     doc.current_path,
                "file_name":        doc.file_name,
                "file_extension":   doc.file_extension,
                "file_size_bytes":  doc.file_size_bytes,
                "quick_hash":       doc.quick_hash,
                "content_hash":     doc.content_hash,
                "chunk_count":      doc.chunk_count,
                "index_name":       doc.index_name,
                "status":           doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                "deleted_at":       doc.deleted_at,
                "deletion_reason":  doc.deletion_reason,
                "first_indexed_at": doc.first_indexed_at,
                "last_indexed_at":  doc.last_indexed_at,
                "last_error":       doc.last_error,
                "retry_count":      doc.retry_count,
                "version":          doc.version,
                "created_at":       doc.created_at,
                "updated_at":       doc.updated_at,
            },
        )
        self._conn.commit()

    def mark_deleted(
        self,
        source_id: str,
        reason: str = "file_removed",
    ) -> None:
        """
        Effectue un soft-delete sur un document (tombstone).

        La ligne reste en base pour audit et réconciliation.
        Les chunks dans Qdrant/OpenSearch doivent être supprimés séparément
        par le worker (via ChunkCleaner) avant d'appeler cette méthode.

        Args:
            source_id: UUID du document à supprimer.
            reason: Raison de la suppression (DeletionReason).
        """
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            UPDATE documents
            SET status = 'deleted',
                deleted_at = :deleted_at,
                deletion_reason = :reason,
                updated_at = :now
            WHERE source_id = :source_id
            """,
            {"source_id": source_id, "deleted_at": now, "reason": reason, "now": now},
        )
        self._conn.commit()
        logger.debug(f"Soft-delete appliqué : source_id={source_id}, raison={reason}")

    def move_document(self, source_id: str, new_path: str) -> None:
        """
        Met à jour le chemin courant d'un document après un move/rename.

        Le `source_id` et l'historique sont préservés intégralement.

        Args:
            source_id: UUID du document.
            new_path: Nouveau chemin absolu normalisé.
        """
        now = datetime.utcnow().isoformat()
        from pathlib import Path as PPath
        self._conn.execute(
            """
            UPDATE documents
            SET current_path = :new_path,
                file_name = :file_name,
                updated_at = :now
            WHERE source_id = :source_id
            """,
            {
                "source_id": source_id,
                "new_path": new_path,
                "file_name": PPath(new_path).name,
                "now": now,
            },
        )
        self._conn.commit()
        logger.debug(f"Move appliqué : source_id={source_id} → {new_path}")

    def mark_error(self, source_id: str, error_message: str) -> None:
        """
        Marque un document en erreur et incrémente son compteur de tentatives.

        Args:
            source_id: UUID du document.
            error_message: Message d'erreur à persister.
        """
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            UPDATE documents
            SET status = 'error',
                last_error = :error,
                retry_count = retry_count + 1,
                updated_at = :now
            WHERE source_id = :source_id
            """,
            {"source_id": source_id, "error": error_message, "now": now},
        )
        self._conn.commit()

    def close(self) -> None:
        """Ferme la connexion SQLite."""
        try:
            self._conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────


def _row_to_doc(row: sqlite3.Row) -> DocumentState:
    """Convertit une ligne SQLite en DocumentState."""
    status_val = row["status"]
    try:
        status = DocumentStatus(status_val)
    except ValueError:
        status = DocumentStatus.PENDING

    return DocumentState(
        source_id=row["source_id"],
        current_path=row["current_path"],
        file_name=row["file_name"],
        file_extension=row["file_extension"] or "",
        file_size_bytes=row["file_size_bytes"] or 0,
        quick_hash=row["quick_hash"] or "",
        content_hash=row["content_hash"] or "",
        chunk_count=row["chunk_count"] or 0,
        index_name=row["index_name"] or "documents",
        status=status,
        deleted_at=row["deleted_at"],
        deletion_reason=row["deletion_reason"],
        first_indexed_at=row["first_indexed_at"],
        last_indexed_at=row["last_indexed_at"],
        last_error=row["last_error"],
        retry_count=row["retry_count"] or 0,
        version=row["version"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
