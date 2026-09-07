"""
API FastAPI du service watcher — Endpoints d'administration.

Tous les endpoints sont préfixés par /api/watcher et requièrent
l'authentification admin (réutilise le cookie de session existant).

Le WatcherService est injecté via une dépendance FastAPI.
Il doit être initialisé avant le montage du routeur (dans le lifespan).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from .db import purge_tombstones, get_connection

router = APIRouter(prefix="/api/watcher", tags=["watcher"])

# ─────────────────────────────────────────────────────────────────────────────
# Injection du WatcherService
# (la variable globale est définie par WatcherService au démarrage de l'app)
# ─────────────────────────────────────────────────────────────────────────────

_watcher_service = None


def set_watcher_service(service) -> None:
    """
    Enregistre l'instance WatcherService pour l'injection dans les routes.

    À appeler depuis le lifespan FastAPI après `await service.start()`.
    """
    global _watcher_service
    _watcher_service = service


def get_service():
    """Dépendance FastAPI — retourne le WatcherService ou lève 503."""
    if _watcher_service is None:
        raise HTTPException(status_code=503, detail="Service watcher non initialisé")
    return _watcher_service


# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic des requêtes
# ─────────────────────────────────────────────────────────────────────────────


class RescanRequest(BaseModel):
    path: Optional[str] = None


class ReindexRequest(BaseModel):
    file_path: str


class DeleteDocumentRequest(BaseModel):
    file_path: str


class PurgeTombstonesRequest(BaseModel):
    older_than_days: int = 90


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Statut
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/status", summary="État général du service watcher")
async def get_status(service=Depends(get_service)) -> dict:
    """
    Retourne l'état complet du watcher : composants actifs, métriques
    documents et jobs, dernier scan de réconciliation.
    """
    return service.get_status()


@router.get(
    "/status/{watched_path_id}",
    summary="État d'un chemin surveillé spécifique",
)
async def get_path_status(
    watched_path_id: int,
    service=Depends(get_service),
) -> dict:
    """
    Retourne les statistiques de documents pour un chemin surveillé
    identifié par son index dans la liste configured.
    """
    paths = service.config.watched_paths
    if watched_path_id >= len(paths):
        raise HTTPException(status_code=404, detail="Chemin surveillé introuvable")

    wp = paths[watched_path_id]
    state = service.state
    if not state:
        raise HTTPException(status_code=503, detail="StateStore non disponible")

    counts = state.get_documents_count_by_status()

    return {
        "path": wp.path,
        "recursive": wp.recursive,
        "index_name": wp.index_name or service.config.default_index_name,
        "documents_indexed": counts.get("indexed", 0),
        "documents_error": counts.get("error", 0),
        "documents_pending": counts.get("pending", 0),
        "documents_deleted": counts.get("deleted", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Jobs
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/jobs", summary="Liste des jobs de la queue")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filtre par statut"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service=Depends(get_service),
) -> dict:
    """
    Retourne une liste paginée de jobs.
    Statuts possibles : pending, in_progress, completed, failed, dead.
    """
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    jobs = queue.list_jobs(status=status, limit=limit, offset=offset)
    counts = queue.get_counts_by_status()

    return {
        "jobs": [_job_to_dict(j) for j in jobs],
        "total_by_status": counts,
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/{job_id}", summary="Détail d'un job")
async def get_job(job_id: str, service=Depends(get_service)) -> dict:
    """Retourne le détail complet d'un job par son UUID."""
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job introuvable : {job_id}")

    return _job_to_dict(job)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Actions
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/rescan", summary="Lancer un scan de réconciliation manuel")
async def trigger_rescan(
    body: RescanRequest,
    service=Depends(get_service),
) -> dict:
    """
    Lance immédiatement un scan de réconciliation.
    Enqueue des jobs pour les fichiers nouveaux, modifiés ou supprimés.
    """
    try:
        run_id = service.trigger_rescan(path=body.path)
        return {"run_id": run_id, "message": "Scan démarré"}
    except Exception as exc:
        logger.error(f"Erreur rescan : {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retry/{job_id}", summary="Relancer un job en erreur")
async def retry_job(job_id: str, service=Depends(get_service)) -> dict:
    """Remet un job en statut 'pending' pour une nouvelle tentative."""
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job introuvable : {job_id}")

    queue.retry_job(job_id)
    return {"job_id": job_id, "status": "pending", "message": "Job relancé"}


@router.post("/retry-all-failed", summary="Relancer tous les jobs en erreur")
async def retry_all_failed(service=Depends(get_service)) -> dict:
    """Remet tous les jobs 'failed' et 'dead' en 'pending'."""
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    count = queue.retry_all_failed()
    return {"retried": count, "message": f"{count} job(s) relancé(s)"}


@router.post("/reindex", summary="Forcer la réindexation d'un fichier")
async def reindex_file(
    body: ReindexRequest,
    service=Depends(get_service),
) -> dict:
    """
    Force la réindexation d'un fichier, même si son contenu n'a pas changé.
    Crée un job upsert de haute priorité.
    """
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    job_id = queue.enqueue(
        file_path=body.file_path,
        action="upsert",
        source="manual",
        priority=1,
    )
    return {
        "job_id": job_id,
        "file_path": body.file_path,
        "message": "Réindexation planifiée",
    }


@router.delete("/document", summary="Supprimer un document de l'index")
async def delete_document(
    body: DeleteDocumentRequest,
    service=Depends(get_service),
) -> dict:
    """
    Enqueue un job de suppression pour un document.
    Supprime les chunks dans Qdrant/OpenSearch et marque le document comme deleted.
    """
    queue = service.queue
    if not queue:
        raise HTTPException(status_code=503, detail="Queue non disponible")

    job_id = queue.enqueue(
        file_path=body.file_path,
        action="delete",
        source="manual",
        priority=1,
    )
    return {
        "job_id": job_id,
        "file_path": body.file_path,
        "message": "Suppression planifiée",
    }


@router.post("/purge-tombstones", summary="Purger les suppressions anciennes")
async def purge_old_tombstones(
    body: PurgeTombstonesRequest,
    service=Depends(get_service),
) -> dict:
    """
    Supprime définitivement les documents soft-deleted datant de plus de N jours.
    La suppression cascade sur les chunks d'audit.
    """
    conn = get_connection(service.config.db_path)
    try:
        count = purge_tombstones(conn, older_than_days=body.older_than_days)
        return {
            "purged": count,
            "older_than_days": body.older_than_days,
            "message": f"{count} tombstone(s) supprimé(s)",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Documents
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/documents", summary="Liste des documents indexés")
async def list_documents(
    status: Optional[str] = Query(None, description="Filtre par statut"),
    search: Optional[str] = Query(None, description="Recherche dans le nom de fichier"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service=Depends(get_service),
) -> dict:
    """Retourne une liste paginée de documents avec leur statut d'indexation."""
    state = service.state
    if not state:
        raise HTTPException(status_code=503, detail="StateStore non disponible")

    docs = state.search_documents(
        search=search,
        status=status,
        limit=limit,
        offset=offset,
    )

    return {
        "documents": [_doc_to_dict(d) for d in docs],
        "limit": limit,
        "offset": offset,
    }


@router.get("/documents/{source_id}", summary="Détail d'un document")
async def get_document(source_id: str, service=Depends(get_service)) -> dict:
    """Retourne le détail complet d'un document par son source_id."""
    state = service.state
    if not state:
        raise HTTPException(status_code=503, detail="StateStore non disponible")

    doc = state.get_document_by_source_id(source_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document introuvable : {source_id}")

    return _doc_to_dict(doc)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Métriques
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/metrics", summary="Métriques du service watcher")
async def get_metrics(service=Depends(get_service)) -> dict:
    """
    Retourne les métriques opérationnelles du watcher.
    Utile pour le monitoring et les tableaux de bord.
    """
    status = service.get_status()
    queue = service.queue

    last_event_ts = None
    if queue:
        conn = get_connection(service.config.db_path)
        try:
            row = conn.execute(
                "SELECT timestamp FROM watcher_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                last_event_ts = row["timestamp"]
        finally:
            conn.close()

    return {
        "documents_indexed": status.get("documents", {}).get("indexed", 0),
        "documents_error": status.get("documents", {}).get("error", 0),
        "documents_pending": status.get("documents", {}).get("pending", 0),
        "documents_deleted": status.get("documents", {}).get("deleted", 0),
        "queue_pending": status.get("queue", {}).get("pending", 0),
        "queue_failed": status.get("queue", {}).get("failed", 0),
        "queue_dead": status.get("queue", {}).get("dead", 0),
        "debounce_active": status.get("debounce_active", 0),
        "uptime_seconds": status.get("uptime_seconds", 0),
        "last_event_timestamp": last_event_ts,
        "watcher_running": status.get("components", {}).get("watcher", False),
        "worker_running": status.get("components", {}).get("worker", False),
        "reconciler_running": status.get("components", {}).get("reconciler", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints : Journal
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/events", summary="Journal des événements filesystem")
async def get_events(
    limit: int = Query(100, ge=1, le=1000),
    since: Optional[str] = Query(None, description="ISO datetime (ex: 2026-05-16T00:00:00)"),
    service=Depends(get_service),
) -> dict:
    """Retourne les derniers événements filesystem journalisés."""
    conn = get_connection(service.config.db_path)
    try:
        if since:
            rows = conn.execute(
                """
                SELECT * FROM watcher_events
                WHERE timestamp >= ?
                ORDER BY id DESC LIMIT ?
                """,
                (since, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM watcher_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return {
            "events": [dict(r) for r in rows],
            "count": len(rows),
        }
    finally:
        conn.close()


@router.get("/errors", summary="Journal des erreurs d'indexation")
async def get_errors(
    resolved: Optional[bool] = Query(False, description="Inclure les erreurs résolues"),
    limit: int = Query(50, ge=1, le=500),
    service=Depends(get_service),
) -> dict:
    """Retourne les erreurs d'indexation non résolues (ou toutes si resolved=true)."""
    conn = get_connection(service.config.db_path)
    try:
        if resolved:
            rows = conn.execute(
                "SELECT * FROM audit_errors ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM audit_errors
                WHERE resolved = 0
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "errors": [dict(r) for r in rows],
            "count": len(rows),
        }
    finally:
        conn.close()


@router.post(
    "/errors/{error_id}/resolve",
    summary="Marquer une erreur comme résolue",
)
async def resolve_error(error_id: int, service=Depends(get_service)) -> dict:
    """Marque manuellement une erreur d'audit comme résolue."""
    conn = get_connection(service.config.db_path)
    try:
        conn.execute(
            """
            UPDATE audit_errors
            SET resolved = 1, resolved_at = ?
            WHERE id = ?
            """,
            (datetime.utcnow().isoformat(), error_id),
        )
        conn.commit()
        return {"error_id": error_id, "resolved": True}
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de sérialisation
# ─────────────────────────────────────────────────────────────────────────────


def _job_to_dict(job) -> dict:
    """Convertit un Job dataclass en dict JSON-sérialisable."""
    return {
        "job_id": job.job_id,
        "file_path": job.file_path,
        "action": job.action,
        "old_path": job.old_path,
        "status": job.status,
        "priority": job.priority,
        "source": job.source,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "next_retry_at": job.next_retry_at,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def _doc_to_dict(doc) -> dict:
    """Convertit un DocumentState dataclass en dict JSON-sérialisable."""
    return {
        "source_id": doc.source_id,
        "current_path": doc.current_path,
        "file_name": doc.file_name,
        "file_extension": doc.file_extension,
        "file_size_bytes": doc.file_size_bytes,
        "chunk_count": doc.chunk_count,
        "index_name": doc.index_name,
        "status": doc.status.value if hasattr(doc.status, "value") else doc.status,
        "deleted_at": doc.deleted_at,
        "deletion_reason": doc.deletion_reason,
        "first_indexed_at": doc.first_indexed_at,
        "last_indexed_at": doc.last_indexed_at,
        "last_error": doc.last_error,
        "retry_count": doc.retry_count,
        "version": doc.version,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }
