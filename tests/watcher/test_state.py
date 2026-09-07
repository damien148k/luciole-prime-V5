"""
Tests unitaires — state.py

Vérifie : CRUD DocumentState, move, soft-delete, search
"""

from pathlib import Path


from src.watcher.models import DocumentState, DocumentStatus
from src.watcher.state import StateStore


def make_doc(path: str = "/data/rapport.pdf", index: str = "documents") -> DocumentState:
    """Crée un DocumentState minimal pour les tests."""
    return DocumentState(
        current_path=path,
        file_name=Path(path).name,
        file_extension=Path(path).suffix.lower(),
        index_name=index,
        content_hash="abc123",
        quick_hash="qh001",
    )


class TestUpsertAndGet:
    def test_upsert_puis_get_par_source_id(self, state: StateStore) -> None:
        doc = make_doc()
        state.upsert_document(doc)

        result = state.get_document_by_source_id(doc.source_id)
        assert result is not None
        assert result.source_id == doc.source_id
        assert result.current_path == doc.current_path

    def test_upsert_puis_get_par_path(self, state: StateStore) -> None:
        doc = make_doc("/data/test.docx")
        state.upsert_document(doc)

        result = state.get_document_by_path("/data/test.docx")
        assert result is not None
        assert result.file_name == "test.docx"

    def test_get_path_inexistant_retourne_none(self, state: StateStore) -> None:
        result = state.get_document_by_path("/data/inexistant.pdf")
        assert result is None

    def test_upsert_met_a_jour_sans_changer_source_id(self, state: StateStore) -> None:
        doc = make_doc()
        state.upsert_document(doc)
        original_source_id = doc.source_id

        # Mise à jour du contenu
        doc.content_hash = "nouveau_hash"
        doc.chunk_count = 42
        doc.status = DocumentStatus.INDEXED
        state.upsert_document(doc)

        updated = state.get_document_by_source_id(original_source_id)
        assert updated.content_hash == "nouveau_hash"
        assert updated.chunk_count == 42
        assert updated.source_id == original_source_id

    def test_updated_at_change_a_chaque_upsert(self, state: StateStore) -> None:
        doc = make_doc()
        state.upsert_document(doc)
        first_updated = state.get_document_by_source_id(doc.source_id).updated_at

        import time; time.sleep(0.01)
        doc.content_hash = "nouveau"
        state.upsert_document(doc)
        second_updated = state.get_document_by_source_id(doc.source_id).updated_at

        assert second_updated >= first_updated


class TestMoveDocument:
    def test_move_met_a_jour_current_path(self, state: StateStore) -> None:
        doc = make_doc("/data/ancien.pdf")
        state.upsert_document(doc)

        state.move_document(doc.source_id, "/data/nouveau.pdf")

        moved = state.get_document_by_source_id(doc.source_id)
        assert moved.current_path == "/data/nouveau.pdf"
        assert moved.file_name == "nouveau.pdf"

    def test_move_preserve_source_id(self, state: StateStore) -> None:
        doc = make_doc("/data/original.docx")
        state.upsert_document(doc)
        source_id = doc.source_id

        state.move_document(source_id, "/data/renomme.docx")

        result = state.get_document_by_source_id(source_id)
        assert result is not None
        assert result.source_id == source_id

    def test_ancien_path_devient_introuvable(self, state: StateStore) -> None:
        doc = make_doc("/data/avant.pdf")
        state.upsert_document(doc)
        state.move_document(doc.source_id, "/data/apres.pdf")

        result = state.get_document_by_path("/data/avant.pdf")
        assert result is None


class TestSoftDelete:
    def test_mark_deleted_applique_tombstone(self, state: StateStore) -> None:
        doc = make_doc()
        state.upsert_document(doc)

        state.mark_deleted(doc.source_id, reason="file_removed")

        result = state.get_document_by_source_id(doc.source_id)
        assert result.status == DocumentStatus.DELETED
        assert result.deleted_at is not None
        assert result.deletion_reason == "file_removed"

    def test_document_deleted_invisible_par_path(self, state: StateStore) -> None:
        doc = make_doc("/data/supprime.pdf")
        state.upsert_document(doc)
        state.mark_deleted(doc.source_id)

        result = state.get_document_by_path("/data/supprime.pdf")
        assert result is None  # Ignoré car status='deleted'

    def test_document_deleted_toujours_visible_par_source_id(self, state: StateStore) -> None:
        doc = make_doc()
        state.upsert_document(doc)
        state.mark_deleted(doc.source_id)

        result = state.get_document_by_source_id(doc.source_id)
        assert result is not None
        assert result.is_deleted


class TestGetAllIndexedPaths:
    def test_retourne_seulement_les_indexed(self, state: StateStore) -> None:
        doc1 = make_doc("/data/a.pdf")
        doc1.status = DocumentStatus.INDEXED
        doc2 = make_doc("/data/b.pdf")
        doc2.status = DocumentStatus.ERROR

        state.upsert_document(doc1)
        state.upsert_document(doc2)

        paths = state.get_all_indexed_paths()
        assert "/data/a.pdf" in paths
        assert "/data/b.pdf" not in paths

    def test_deleted_non_inclus(self, state: StateStore) -> None:
        doc = make_doc("/data/supprime.pdf")
        doc.status = DocumentStatus.INDEXED
        state.upsert_document(doc)
        state.mark_deleted(doc.source_id)

        paths = state.get_all_indexed_paths()
        assert "/data/supprime.pdf" not in paths


class TestSearchDocuments:
    def test_recherche_par_nom(self, state: StateStore) -> None:
        doc = make_doc("/data/rapport_annuel.pdf")
        doc.status = DocumentStatus.INDEXED
        state.upsert_document(doc)

        results = state.search_documents(search="rapport")
        assert any(d.file_name == "rapport_annuel.pdf" for d in results)

    def test_recherche_insensible_casse(self, state: StateStore) -> None:
        doc = make_doc("/data/BILAN.docx")
        state.upsert_document(doc)

        results = state.search_documents(search="bilan")
        assert any(d.file_name == "BILAN.docx" for d in results)

    def test_filtre_par_statut(self, state: StateStore) -> None:
        doc_ok = make_doc("/data/ok.pdf")
        doc_ok.status = DocumentStatus.INDEXED
        doc_err = make_doc("/data/erreur.pdf")
        doc_err.status = DocumentStatus.ERROR

        state.upsert_document(doc_ok)
        state.upsert_document(doc_err)

        results = state.search_documents(status="indexed")
        assert all(d.status == DocumentStatus.INDEXED for d in results)
