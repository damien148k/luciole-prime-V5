"""
Tests d'intégration — reconciler.py

Vérifie : détection fichiers nouveaux, supprimés, modifiés.
"""

from pathlib import Path

import pytest

from src.watcher.models import DocumentState, DocumentStatus, WatcherConfig
from src.watcher.queue import JobQueue
from src.watcher.reconciler import Reconciler
from src.watcher.state import StateStore


@pytest.fixture
def reconciler(
    watcher_config: WatcherConfig,
    queue: JobQueue,
    state: StateStore,
) -> Reconciler:
    return Reconciler(config=watcher_config, queue=queue, state=state)


class TestRunFullScan:
    def test_nouveau_fichier_enqueue_upsert(
        self,
        reconciler: Reconciler,
        queue: JobQueue,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "nouveau.pdf"
        f.write_bytes(b"contenu")

        report = reconciler.run_full_scan()

        assert report.new_files_enqueued == 1
        assert report.deleted_files_enqueued == 0
        jobs = queue.list_jobs(status="pending")
        assert any(j.file_path == str(f.resolve()) for j in jobs)

    def test_fichier_indexe_absent_enqueue_delete(
        self,
        reconciler: Reconciler,
        queue: JobQueue,
        state: StateStore,
    ) -> None:
        # Simuler un fichier indexé qui n'existe plus
        doc = DocumentState(
            current_path="/data/disparu.pdf",
            file_name="disparu.pdf",
            file_extension=".pdf",
            index_name="test_documents",
            status=DocumentStatus.INDEXED,
        )
        state.upsert_document(doc)

        report = reconciler.run_full_scan()

        assert report.deleted_files_enqueued == 1
        jobs = queue.list_jobs(status="pending")
        assert any(j.file_path == "/data/disparu.pdf" and j.action == "delete" for j in jobs)

    def test_fichier_inchange_pas_enqueue(
        self,
        reconciler: Reconciler,
        queue: JobQueue,
        state: StateStore,
        tmp_path: Path,
    ) -> None:
        from src.watcher.hashing import quick_hash

        f = tmp_path / "stable.pdf"
        f.write_bytes(b"contenu stable")

        # Simuler un document indexé avec le bon quick_hash
        doc = DocumentState(
            current_path=str(f.resolve()),
            file_name=f.name,
            file_extension=".pdf",
            index_name="test_documents",
            status=DocumentStatus.INDEXED,
            quick_hash=quick_hash(f),
        )
        state.upsert_document(doc)

        report = reconciler.run_full_scan()

        assert report.modified_files_enqueued == 0
        assert report.new_files_enqueued == 0

    def test_fichier_modifie_enqueue_upsert(
        self,
        reconciler: Reconciler,
        queue: JobQueue,
        state: StateStore,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "modifie.pdf"
        f.write_bytes(b"contenu v1")

        # Simuler un document indexé avec un ancien quick_hash
        doc = DocumentState(
            current_path=str(f.resolve()),
            file_name=f.name,
            file_extension=".pdf",
            index_name="test_documents",
            status=DocumentStatus.INDEXED,
            quick_hash="ancien_hash_invalide",
        )
        state.upsert_document(doc)

        report = reconciler.run_full_scan()

        assert report.modified_files_enqueued == 1

    def test_rapport_contient_totaux_corrects(
        self,
        reconciler: Reconciler,
        state: StateStore,
        tmp_path: Path,
    ) -> None:
        # 2 nouveaux fichiers
        (tmp_path / "a.pdf").write_bytes(b"a")
        (tmp_path / "b.docx").write_bytes(b"b")

        # 1 fichier supprimé (dans le store, absent du disque)
        doc = DocumentState(
            current_path="/ghost/missing.pdf",
            file_name="missing.pdf",
            file_extension=".pdf",
            index_name="test_documents",
            status=DocumentStatus.INDEXED,
        )
        state.upsert_document(doc)

        report = reconciler.run_full_scan()

        assert report.new_files_enqueued == 2
        assert report.deleted_files_enqueued == 1
        assert report.total_jobs_created == 3
        assert report.finished_at is not None

    def test_fichier_tmp_ignore_lors_du_scan(
        self,
        reconciler: Reconciler,
        queue: JobQueue,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "fichier.tmp").write_bytes(b"temporaire")
        (tmp_path / "~$word.docx").write_bytes(b"lock")

        report = reconciler.run_full_scan()

        assert report.new_files_enqueued == 0
