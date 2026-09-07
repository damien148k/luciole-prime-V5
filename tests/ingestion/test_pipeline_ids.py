"""Identifiants Qdrant deterministes et pipeline sans etat (V4).

Ces tests n'ont besoin ni de GPU, ni de Qdrant, ni d'OpenSearch : les
clients et l'embedder sont remplaces par des doublures.
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings, DEFAULTS, _fusionner
from src.ingestion.pipeline import IngestionPipeline, point_id, document_hash


def test_point_id_stable_dans_un_autre_processus():
    """Le meme chunk_id donne le meme id, meme dans un processus separe
    (ce que hash() de V3 ne garantissait pas)."""
    ici = point_id("Tome_4.pdf_chunk_12")
    code = ("import sys; sys.path.insert(0, %r); "
            "from src.ingestion.pipeline import point_id; "
            "print(point_id('Tome_4.pdf_chunk_12'))" % str(Path(__file__).resolve().parents[2]))
    ailleurs = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, check=True).stdout.strip()
    assert ici == ailleurs
    assert point_id("autre") != ici


def test_document_hash_sha256(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("bonjour", encoding="utf-8")
    assert document_hash(str(f)) == (
        "2cb4b1431b84ec15d35ed83bb927e27e8967d75f4bcd9cc4b25c8d879ae23e18")


def _pipeline_factice(tmp_path):
    settings = Settings(_fusionner(DEFAULTS, {"qdrant": {"collection_name": "test"},
                                              "opensearch": {"index_name": "test"}}))
    embedder = MagicMock()
    embedder.embedding_dim = 4

    def embed_chunks(chunks):
        return [{
            "chunk_id": c.chunk_id, "document_id": c.document_id, "text": c.text,
            "text_with_context": c.text_with_context, "file_path": c.file_path,
            "file_name": c.file_name, "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": c.metadata,
        } for c in chunks]

    embedder.embed_chunks.side_effect = embed_chunks
    qdrant = MagicMock()
    qdrant.get_collections.return_value.collections = []
    opensearch = MagicMock()
    opensearch.indices.exists.return_value = False
    opensearch.bulk.return_value = {"errors": False, "items": []}
    return IngestionPipeline(settings=settings, qdrant=qdrant, opensearch=opensearch,
                             embedder=embedder), qdrant, opensearch


def test_reingestion_remplace_les_memes_points(tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("Premiere phrase. Deuxieme phrase. Troisieme phrase.", encoding="utf-8")
    pipeline, qdrant, opensearch = _pipeline_factice(tmp_path)

    r1 = pipeline.ingest_file(str(doc))
    ids_1 = [p.id for call in qdrant.upsert.call_args_list for p in call.kwargs["points"]]
    qdrant.upsert.reset_mock()
    r2 = pipeline.ingest_file(str(doc))
    ids_2 = [p.id for call in qdrant.upsert.call_args_list for p in call.kwargs["points"]]

    assert r1["status"] == r2["status"] == "success"
    assert r1["chunks"] == r2["chunks"] > 0
    assert ids_1 == ids_2, "une reingestion doit viser exactement les memes points"
    assert r1["document_hash"] == r2["document_hash"]
    # OpenSearch : _id = chunk_id texte, comme en V3
    actions = opensearch.bulk.call_args.kwargs["body"]
    assert actions[0]["index"]["_id"].endswith("_chunk_0")
    assert actions[1]["document_hash"] == r1["document_hash"]


def test_reglage_lu_sur_settings_pas_copie(tmp_path):
    """Le chunker recoit la valeur de settings au moment de la construction :
    aucune constante de taille n'est figee dans le pipeline."""
    settings = Settings(_fusionner(DEFAULTS, {"chunking": {"chunk_size": 123, "adaptive": False}}))
    embedder = MagicMock(); embedder.embedding_dim = 4
    qdrant = MagicMock(); qdrant.get_collections.return_value.collections = []
    opensearch = MagicMock(); opensearch.indices.exists.return_value = True
    p = IngestionPipeline(settings=settings, qdrant=qdrant, opensearch=opensearch, embedder=embedder)
    assert p.chunker.chunk_size == 123
