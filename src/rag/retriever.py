"""Recherche hybride + rerank + generation : la « route classique » de V3.

Reprise du mode chat de `DocumentAnalyzer` (les modes files/folder/cross
ne sont pas repris). L'interface attendue par `query2.IterativePipeline`
est conservee : `analyze()`, `hybrid_search`, `reranker`, `llm_generator`,
`_build_context()`, `LIMITS`, `fusion_top_k`, `rerank_top_n`.

Difference V4 : `fusion_top_k` et `rerank_top_n` ne sont pas des copies
faites a la construction, ce sont des lectures sur Settings a chaque
acces. Un reload de configuration est vu a la requete suivante.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from loguru import logger

from src.config.settings import Settings


class Retriever:
    LIMITS = {
        "quick": {"max_total_chunks": 20},
        "standard": {"max_total_chunks": 100},
        "deep": {"max_total_chunks": 500},
    }

    def __init__(self, settings_provider, hybrid_search, llm_generator, reranker=None):
        """settings_provider : callable sans argument renvoyant le Settings
        courant (par ex. `src.config.settings.get_settings`)."""
        self._settings = settings_provider
        self.hybrid_search = hybrid_search
        self.llm_generator = llm_generator
        self.reranker = reranker

    @property
    def settings(self) -> Settings:
        return self._settings()

    @property
    def fusion_top_k(self) -> int:
        return int(self.settings.get("retrieval.fusion_top_k", 30))

    @property
    def rerank_top_n(self) -> int:
        return int(self.settings.get("retrieval.rerank_top_n", 15))

    # ------------------------------------------------------------------
    def analyze(self, query: str, mode: str = "chat", scope: Dict = None,
                options: Dict = None, history: list = None, search_queries: list = None) -> Dict:
        debut = time.time()
        options = options or {}
        result = self._chat(
            query, options.get("custom_prompt"), history, search_queries,
            (scope or {}).get("metadata_filters"), options.get("retrieval_overrides"),
        )
        result["metadata"].update({
            "mode": "chat", "processing_time_ms": int((time.time() - debut) * 1000), "query": query,
        })
        return result

    def _chat(self, query, custom_prompt, history, search_queries, filters, overrides) -> Dict:
        ov = overrides or {}
        search_top_k = ov.get("search_top_k") or int(
            self.settings.get("retrieval.search_top_k", self.LIMITS["standard"]["max_total_chunks"]))
        fusion_k = ov.get("fusion_top_k") or self.fusion_top_k
        top_n = ov.get("rerank_top_n") or self.rerank_top_n
        bm25_k, dense_k = ov.get("bm25_top_k"), ov.get("dense_top_k")

        if search_queries and len(search_queries) > 1 and hasattr(self.hybrid_search, "search_multi"):
            results = self.hybrid_search.search_multi(search_queries, top_k=search_top_k, filters=filters,
                                                      bm25_top_k=bm25_k, dense_top_k=dense_k)
        else:
            results = self.hybrid_search.search(query, top_k=search_top_k, filters=filters,
                                                bm25_top_k=bm25_k, dense_top_k=dense_k)
        if self.reranker and results:
            results = self.reranker.rerank(query, results[:fusion_k], top_n=top_n)

        context = self._build_context(results, top_n=top_n)
        llm = self.llm_generator.generate(query, context, results, custom_prompt=custom_prompt, history=history)
        return {
            "result_type": "chat",
            "response": llm.get("response", ""),
            "sources": llm.get("sources", []),
            "search_results": results,
            "metadata": {
                "confidence": llm.get("confidence", 0),
                "model": llm.get("model", "unknown"),
                "custom_prompt_used": custom_prompt is not None,
                "history_used": bool(history),
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _etiquette_source(chunk: Dict) -> str:
        meta = chunk.get("metadata") or {}
        nom = chunk.get("file_name") or meta.get("file_name") or ""
        debut = meta.get("page_start", chunk.get("page_start"))
        fin = meta.get("page_end", chunk.get("page_end"))
        if debut is None:
            return f"[Source: {nom}]"
        if fin is None or fin == debut:
            return f"[Source: {nom}, page {debut}]"
        return f"[Source: {nom}, pages {debut} à {fin}]"

    def _build_context(self, chunks: List[Dict], top_n: Optional[int] = None) -> str:
        if not chunks:
            return ""
        top_n = top_n or self.rerank_top_n
        return "\n\n---\n\n".join(
            f"{self._etiquette_source(c)}\n{c.get('text', '')}" for c in chunks[:top_n]
        )
