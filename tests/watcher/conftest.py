"""
Fixtures partagées entre tous les tests du watcher.
"""

from pathlib import Path

import pytest

# Assurer que les imports src.* fonctionnent depuis le répertoire rag-system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.watcher.db import init_db
from src.watcher.models import WatcherConfig, WatchedPath
from src.watcher.queue import JobQueue
from src.watcher.state import StateStore


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Répertoire temporaire pour les tests filesystem."""
    return tmp_path


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Chemin vers une base SQLite temporaire initialisée."""
    path = str(tmp_path / "watcher_test.db")
    init_db(path)
    return path


@pytest.fixture
def queue(db_path: str) -> JobQueue:
    """JobQueue connectée à la base de test."""
    q = JobQueue(db_path=db_path, retry_max_attempts=3, retry_backoff_base=1.0)
    yield q
    q.close()


@pytest.fixture
def state(db_path: str) -> StateStore:
    """StateStore connecté à la base de test."""
    s = StateStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def watcher_config(tmp_dir: Path, db_path: str) -> WatcherConfig:
    """Configuration watcher minimale pour les tests."""
    return WatcherConfig(
        enabled=True,
        watched_paths=[
            WatchedPath(path=str(tmp_dir), recursive=True)
        ],
        polling_interval=1.0,
        debounce_seconds=0.1,   # Très court pour les tests
        stability_checks=2,
        stability_interval=0.1,
        reconcile_interval=9999,
        reconcile_on_startup=False,
        db_path=db_path,
        default_index_name="test_documents",
    )
