"""
Tests unitaires — observer.py

Vérifie : filtrage des événements, debounce, détection des extensions.
Les tests utilisent des mocks — pas de watchdog réel.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.watcher.models import WatcherConfig, WatchedPath
from src.watcher.observer import FileWatcher
from src.watcher.queue import JobQueue


@pytest.fixture
def mock_queue() -> MagicMock:
    """Mock de JobQueue."""
    q = MagicMock(spec=JobQueue)
    q.enqueue.return_value = "job-id-test"
    return q


@pytest.fixture
def mock_db_conn() -> MagicMock:
    """Mock de connexion SQLite."""
    conn = MagicMock()
    conn.execute.return_value = MagicMock()
    return conn


@pytest.fixture
def watcher(watcher_config: WatcherConfig, mock_queue: MagicMock, mock_db_conn: MagicMock) -> FileWatcher:
    """FileWatcher avec queue et connexion mockées, marqué comme démarré."""
    fw = FileWatcher(
        config=watcher_config,
        queue=mock_queue,
        db_conn=mock_db_conn,
    )
    # Simuler un watcher démarré sans lancer le vrai PollingObserver
    fw._running = True
    return fw


class TestShouldIgnore:
    def test_extension_supportee_non_ignoree(self, watcher: FileWatcher, tmp_path: Path) -> None:
        f = tmp_path / "rapport.pdf"
        f.write_bytes(b"contenu")
        result = watcher._should_ignore(str(f))
        assert result is None

    def test_extension_temporaire_ignoree(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "fichier.tmp"))
        assert result is not None
        assert "temporaire" in result

    def test_extension_non_supportee_ignoree(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "fichier.xyz"))
        assert result is not None

    def test_prefixe_word_temporaire_ignore(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "~$document.docx"))
        assert result is not None
        assert "temporaire" in result

    def test_prefixe_libreoffice_ignore(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / ".~lock.document.odt#"))
        assert result is not None

    def test_repertoire_exclu_ignore(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "__pycache__" / "module.py"))
        assert result is not None

    def test_fichier_md_supporte(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "README.md"))
        assert result is None

    def test_fichier_txt_supporte(self, watcher: FileWatcher, tmp_path: Path) -> None:
        result = watcher._should_ignore(str(tmp_path / "notes.txt"))
        assert result is None


class TestDebounce:
    def test_debounce_enqueue_apres_delai(
        self,
        watcher: FileWatcher,
        mock_queue: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Un seul enqueue est émis après le délai de debounce."""
        path = str(tmp_path / "doc.pdf")

        watcher._schedule_debounced(path=path, action="upsert", old_path=None, index_name="docs")
        time.sleep(0.3)  # Délai = 0.1s en test, attendre qu'il se déclenche

        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args.kwargs["file_path"] == path
        assert call_args.kwargs["action"] == "upsert"

    def test_debounce_fusionne_evenements_rapides(
        self,
        watcher: FileWatcher,
        mock_queue: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Plusieurs événements rapides → un seul job."""
        path = str(tmp_path / "doc.pdf")

        for _ in range(5):
            watcher._schedule_debounced(path=path, action="upsert", old_path=None, index_name="docs")
            time.sleep(0.02)

        time.sleep(0.3)  # Attendre l'expiration du dernier timer

        assert mock_queue.enqueue.call_count == 1

    def test_debounce_deux_paths_distincts(
        self,
        watcher: FileWatcher,
        mock_queue: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Deux fichiers différents → deux jobs distincts."""
        path1 = str(tmp_path / "doc1.pdf")
        path2 = str(tmp_path / "doc2.pdf")

        watcher._schedule_debounced(path=path1, action="upsert", old_path=None, index_name="docs")
        watcher._schedule_debounced(path=path2, action="upsert", old_path=None, index_name="docs")
        time.sleep(0.3)

        assert mock_queue.enqueue.call_count == 2

    def test_stop_annule_timers_actifs(
        self,
        watcher: FileWatcher,
        mock_queue: MagicMock,
        tmp_path: Path,
    ) -> None:
        """stop() annule les timers en cours → aucun enqueue."""
        path = str(tmp_path / "doc.pdf")
        watcher._schedule_debounced(path=path, action="upsert", old_path=None, index_name="docs")
        watcher.stop()  # Annule avant expiration

        time.sleep(0.3)
        mock_queue.enqueue.assert_not_called()


class TestBuildAllowedExtensions:
    def test_extensions_par_defaut_quand_config_vide(
        self,
        watcher: FileWatcher,
    ) -> None:
        exts = watcher._allowed_extensions
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".md" in exts

    def test_extensions_configurees_surchargent_les_defauts(
        self,
        tmp_path: Path,
        db_path: str,
        mock_queue: MagicMock,
        mock_db_conn: MagicMock,
    ) -> None:
        config = WatcherConfig(
            watched_paths=[WatchedPath(path=str(tmp_path))],
            allowed_extensions=["pdf", ".docx"],
            db_path=db_path,
        )
        fw = FileWatcher(config=config, queue=mock_queue, db_conn=mock_db_conn)
        assert ".pdf" in fw._allowed_extensions
        assert ".docx" in fw._allowed_extensions
        assert ".txt" not in fw._allowed_extensions
