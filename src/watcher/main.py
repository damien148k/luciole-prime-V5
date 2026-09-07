"""
Point d'entrée FastAPI du service watcher standalone.

Ce module expose une application FastAPI minimale dédiée au watcher.
Elle peut être lancée en conteneur séparé ou intégrée dans l'admin-ui.

Démarrage :
    uvicorn src.watcher.main:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from .api import router, set_watcher_service
from .config import load_watcher_config
from .service import WatcherService

# Instance globale du WatcherService
_service: WatcherService | None = None


def _journal_fichier() -> None:
    """Duplique le journal dans un fichier lisible par l'interface admin.

    L'admin doit pouvoir montrer ce que fait le watcher. Passer par le
    socket Docker donnerait a un conteneur d'interface le controle du
    demon : un fichier depose dans backups/, deja monte par le watcher en
    ecriture et lisible ailleurs, suffit et n'accorde aucun privilege.

    Rotation a 10 Mo, trois fichiers conserves : meme ordre de grandeur
    que la rotation Docker configuree dans docker-compose.yml.
    """
    chemin = os.environ.get("WATCHER_LOG_PATH", "/app/backups/watcher/watcher.log")
    try:
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        logger.add(chemin, rotation="10 MB", retention=3, encoding="utf-8",
                   enqueue=True, level=os.environ.get("LOG_LEVEL", "INFO"))
        logger.info(f"Journal du watcher ecrit dans {chemin}")
    except Exception as e:  # noqa: BLE001
        # Un journal indisponible ne doit jamais empecher l'ingestion.
        logger.warning(f"Journal fichier indisponible ({e})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie FastAPI.

    Démarre le WatcherService au démarrage de l'application
    et l'arrête proprement à l'extinction.
    """
    global _service

    _journal_fichier()
    config = load_watcher_config()
    _service = WatcherService(config=config)

    await _service.start()
    set_watcher_service(_service)

    logger.info("Watcher API prête")
    yield

    logger.info("Arrêt du Watcher Service...")
    await _service.stop()


app = FastAPI(
    title="Luciole Prime — Watcher API",
    description="API d'administration du service de surveillance de fichiers RAG",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/watcher/docs",
    redoc_url="/api/watcher/redoc",
)

app.include_router(router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Vérifie que le service est actif."""
    return {
        "status": "ok",
        "service": "watcher",
        "running": _service.is_running if _service else False,
    }
