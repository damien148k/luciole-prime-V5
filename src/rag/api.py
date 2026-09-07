"""API du coeur RAG (L2).

  POST /api/rag/query      {question, custom_prompt?, history?, deep?, filters?}
  POST /api/reload-config  relit settings.yaml et prompts.yaml ; les
                           composants GPU (embedder, reranker) sont gardes
  GET  /api/health

Construction paresseuse : rien n'est charge tant qu'aucune requete
n'arrive. Les composants sont crees par `build_components()`, que les
tests remplacent par des doublures via `app.state.service`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.config import settings as settings_module
from src.config.prompts import reset_prompts_instance

app = FastAPI(title="Luciole Prime V4 — RAG", version="4.0.0")


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    custom_prompt: Optional[str] = None
    history: List[Message] = Field(default_factory=list)
    deep: bool = False
    filters: Optional[Dict] = None


def build_components():
    """Instancie la pile reelle (Qdrant, OpenSearch, modeles)."""
    from src.ingestion.embedder import Embedder
    from src.rag.bm25_search import BM25Search
    from src.rag.dense_search import DenseSearch
    from src.rag.hybrid import HybridSearch
    from src.rag.llm import LLMGenerator
    from src.rag.reranker import Reranker
    from src.rag.retriever import Retriever
    from src.rag.service import RagService

    s = settings_module.get_settings()
    emb = s.section("embedding")
    # EMBEDDER_DEVICE (env, ex. cpu) l'emporte sur embedding.device : sur une
    # carte 12 Go, l'encodage des seules requetes peut rester sur CPU pour
    # laisser la VRAM au LLM (le watcher, lui, garde embedding.device).
    emb_device = os.environ.get("EMBEDDER_DEVICE", "").strip().lower() or emb["device"]
    embedder = Embedder(model_name=emb["model"], device=emb_device)
    dense = DenseSearch(host=s.get("qdrant.host"), port=s.get("qdrant.port"),
                        collection_name=s.collection_name, embedder=embedder)
    bm25 = BM25Search(host=s.get("opensearch.host"), port=s.get("opensearch.port"),
                      index_name=s.opensearch_index)
    r = s.section("retrieval")
    hybrid = HybridSearch(bm25, dense, bm25_weight=r["bm25_weight"], dense_weight=r["dense_weight"],
                          rrf_k=r.get("rrf_k", 60), bm25_top_k=r["bm25_top_k"], dense_top_k=r["dense_top_k"])
    rr = s.section("reranker")
    reranker = Reranker(model_name=rr["model"], device=rr["device"], batch_size=rr.get("batch_size", 32))
    llm = LLMGenerator(s)
    retriever = Retriever(settings_module.get_settings, hybrid, llm, reranker)
    return RagService(settings_module.get_settings, retriever)


def _service():
    if getattr(app.state, "service", None) is None:
        app.state.service = build_components()
    return app.state.service


@app.get("/api/health")
def health():
    svc = getattr(app.state, "service", None)
    llm_ok = None
    if svc is not None:
        try:
            llm_ok = svc.retriever.llm_generator.health_check()
        except Exception:  # noqa: BLE001
            llm_ok = False
    return {"status": "ok", "loaded": svc is not None, "llm": llm_ok,
            "instance": settings_module.get_settings().get("instance.name")}


@app.post("/api/rag/query")
def rag_query(req: QueryRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history] or None
        return _service().query(req.question, custom_prompt=req.custom_prompt,
                                history=history, deep=req.deep, filters=req.filters)
    except Exception as e:  # noqa: BLE001
        logger.exception("rag_query")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload-config")
def reload_config():
    """Relit la configuration. Les reglages retrieval/query2 sont lus a
    chaque requete ; le generateur LLM est reconstruit ; embedder et
    reranker (GPU) sont conserves."""
    settings_module.reload()
    reset_prompts_instance()
    svc = getattr(app.state, "service", None)
    if svc is not None:
        from src.rag.llm import LLMGenerator
        svc.retriever.llm_generator = LLMGenerator(settings_module.get_settings())
    return {"status": "ok", "a_chaud": ["retrieval", "query2", "llm", "prompts"],
            "conserves": ["embedder", "reranker"],
            "redemarrage_requis_pour": ["embedding.model", "reranker.model", "watcher.watched_paths"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("RAG_PORT", 8000)))
