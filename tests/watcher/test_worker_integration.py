"""
Tests d'intégration — worker.py

Vérifie les cycles complets : create → index, modify → réindexation,
delete → suppression, move → mise à jour path, skip si contenu identique.

Utilise des mocks pour le pipeline et les vector stores.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.watcher.cleanup import ChunkCleaner
from src.watcher.models import DocumentStatus, Job
from src.watcher.queue import JobQueue
from src.watcher.state import StateStore
from src.watcher.worker import IndexWorker


def make_mock_pipeline(chunks: int = 5) -> MagicMock:
    """Crée un mock d'IngestionPipeline."""
    pipeline = MagicMock()
    pipeline.ingest_file.return_value = {
        "status": "success",
        "chunks": chunks,
        "file": "mock",
    }
    return pipeline


def make_mock_cleaner() -> MagicMock:
    """Crée un mock de ChunkCleaner."""
    cleaner = MagicMock(spec=ChunkCleaner)
    cleaner.delete_document_chunks.return_value = 0
    return cleaner


@pytest.fixture
def pipeline() -> MagicMock:
    return make_mock_pipeline()


@pytest.fixture
def cleaner() -> MagicMock:
    return make_mock_cleaner()


@pytest.fixture
def worker(queue: JobQueue, state: StateStore, pipeline, cleaner, db_path: str) -> IndexWorker:
    return IndexWorker(
        queue=JobQueue(db_path=db_path),  # connexion séparée pour le worker
        state=StateStore(db_path=db_path),
        cleaner=cleaner,
        pipeline_factory=lambda: pipeline,
        db_path=db_path,
        default_index_name="test_documents",
        stability_checks=1,
        stability_interval=0.01,
    )


class TestHandleUpsert:
    def test_nouveau_fichier_indexe(
        self,
        worker: IndexWorker,
        state: StateStore,
        pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "rapport.pdf"
        f.write_bytes(b"contenu du rapport")

        job = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job)

        pipeline.ingest_file.assert_called_once()
        doc = state.get_document_by_path(str(f))
        assert doc is not None
        assert doc.status == DocumentStatus.INDEXED
        assert doc.chunk_count == 5
        assert doc.version == 1
        assert doc.source_id is not None

    def test_contenu_identique_skip_reindexation(
        self,
        worker: IndexWorker,
        state: StateStore,
        pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"contenu stable")

        # Premier indexation
        job = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job)
        pipeline.ingest_file.reset_mock()

        # Deuxième appel avec le même contenu → skip
        worker._handle_upsert(job)
        pipeline.ingest_file.assert_not_called()

    def test_contenu_modifie_reinsdexe(
        self,
        worker: IndexWorker,
        state: StateStore,
        pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"version 1")

        job = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job)
        source_id_v1 = state.get_document_by_path(str(f)).source_id
        pipeline.ingest_file.reset_mock()

        # Modifier le contenu
        f.write_bytes(b"version 2 modifiee")
        worker._handle_upsert(job)

        pipeline.ingest_file.assert_called_once()
        doc = state.get_document_by_path(str(f))
        assert doc.version == 2
        assert doc.source_id == source_id_v1  # source_id INCHANGÉ

    def test_source_id_stable_apres_reindexation(
        self,
        worker: IndexWorker,
        state: StateStore,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "stable.pdf"
        f.write_bytes(b"contenu v1")
        job = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job)
        sid1 = state.get_document_by_path(str(f)).source_id

        f.write_bytes(b"contenu v2 different")
        worker._handle_upsert(job)
        sid2 = state.get_document_by_path(str(f)).source_id

        assert sid1 == sid2

    def test_ancien_chunks_supprimes_avant_reindexation(
        self,
        worker: IndexWorker,
        cleaner: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"v1")
        job = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job)
        cleaner.delete_document_chunks.reset_mock()

        f.write_bytes(b"v2 contenu modifie")
        worker._handle_upsert(job)

        cleaner.delete_document_chunks.assert_called_once()

    def test_fichier_absent_ne_crash_pas(
        self,
        worker: IndexWorker,
        pipeline: MagicMock,
    ) -> None:
        job = Job(file_path="/data/inexistant.pdf", action="upsert")
        worker._handle_upsert(job)
        pipeline.ingest_file.assert_not_called()


class TestHandleDelete:
    def test_supprime_chunks_et_soft_delete(
        self,
        worker: IndexWorker,
        state: StateStore,
        cleaner: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"contenu")

        # Indexer d'abord
        job_upsert = Job(file_path=str(f), action="upsert")
        worker._handle_upsert(job_upsert)
        source_id = state.get_document_by_path(str(f)).source_id
        file_name = state.get_document_by_path(str(f)).file_name
        cleaner.delete_document_chunks.reset_mock()

        # Supprimer
        job_delete = Job(file_path=str(f), action="delete")
        worker._handle_delete(job_delete)

        cleaner.delete_document_chunks.assert_called_once_with(file_name, "test_documents")
        doc = state.get_document_by_source_id(source_id)
        assert doc.status == DocumentStatus.DELETED
        assert doc.deleted_at is not None

    def test_delete_fichier_inconnu_ne_crash_pas(
        self,
        worker: IndexWorker,
        cleaner: MagicMock,
    ) -> None:
        # Document inconnu du StateStore : pas de crash, et purge fallback
        # "best effort" par basename dans les vector stores (voir _handle_delete).
        job = Job(file_path="/data/jamais_vu.pdf", action="delete")
        worker._handle_delete(job)
        cleaner.delete_document_chunks.assert_called_once()
        assert cleaner.delete_document_chunks.call_args.args[0] == "jamais_vu.pdf"


class TestHandleMove:
    def test_move_met_a_jour_path_sans_reembedding(
        self,
        worker: IndexWorker,
        state: StateStore,
        cleaner: MagicMock,
        pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        old = tmp_path / "ancien.pdf"
        new = tmp_path / "nouveau.pdf"
        old.write_bytes(b"contenu")

        job_upsert = Job(file_path=str(old), action="upsert")
        worker._handle_upsert(job_upsert)
        source_id = state.get_document_by_path(str(old)).source_id
        pipeline.ingest_file.reset_mock()

        job_move = Job(file_path=str(new), action="move", old_path=str(old))
        worker._handle_move(job_move)

        # Pas de ré-embedding
        pipeline.ingest_file.assert_not_called()

        # Path mis à jour dans le StateStore
        doc = state.get_document_by_source_id(source_id)
        assert doc.current_path == str(new)
        assert doc.source_id == source_id

    def test_move_appelle_update_path_dans_cleaners(
        self,
        worker: IndexWorker,
        cleaner: MagicMock,
        tmp_path: Path,
    ) -> None:
        old = tmp_path / "a.pdf"
        new = tmp_path / "b.pdf"
        old.write_bytes(b"contenu")

        worker._handle_upsert(Job(file_path=str(old), action="upsert"))
        cleaner.update_file_path.reset_mock()

        worker._handle_move(Job(file_path=str(new), action="move", old_path=str(old)))
        cleaner.update_file_path.assert_called_once()
