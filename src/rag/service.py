"""Service RAG : assemble les composants et expose le contrat de reponse.

Contrat (docs/rag-ingestion-spec.md, section 7, complete) :

  {answer, citations[], verdict, confidence, trace_id, passages[], trace}

- verdict : SUFFISANT / PARTIEL / INSUFFISANT — projection du verdict de
  couverture de query2 (COUVERT / PARTIEL / NON_COUVERT), avec la regle
  V4 : si la reponse finale esquive encore apres la recherche B, le
  verdict est degrade a PARTIEL au minimum (le juge a pu se tromper, le
  texte final fait foi).
- passages : ordre conserve, etiquette « cible » sur les passages de la
  recherche B (defaut n°6).
"""

from __future__ import annotations

import uuid
from typing import Callable, Dict, List, Optional

from loguru import logger

from src.config.settings import Settings
from src.rag.passages import projeter
from src.rag.query2 import IterativePipeline, _reponse_esquive

VERDICTS = {"COUVERT": "SUFFISANT", "PARTIEL": "PARTIEL", "NON_COUVERT": "INSUFFISANT"}


class RagService:
    def __init__(self, settings_provider: Callable[[], Settings], retriever):
        self._settings = settings_provider
        self.retriever = retriever

    def query(self, question: str, custom_prompt: Optional[str] = None,
              history: Optional[List[Dict]] = None, deep: bool = False,
              filters: Optional[Dict] = None) -> Dict:
        settings = self._settings()
        pipeline = IterativePipeline(self.retriever, query2_config=settings.section("query2"))
        result = pipeline.run(query=question, custom_prompt=custom_prompt,
                              history=history, max_rounds=1, deep=deep)
        answer = result.get("response", "")
        trace = result.get("iterative", {}) or {}
        couverture = (trace.get("couverture") or {}).get("verdict", "COUVERT")
        verdict = VERDICTS.get(couverture, "SUFFISANT")
        esquive = _reponse_esquive(answer)
        if esquive and verdict == "SUFFISANT":
            verdict = "PARTIEL"
        confidence = {"SUFFISANT": 0.8, "PARTIEL": 0.5, "INSUFFISANT": 0.2}[verdict]
        if esquive:
            confidence = min(confidence, 0.4)
        trace_id = str(uuid.uuid4())
        logger.info(f"rag[{trace_id[:8]}] verdict={verdict} esquive={esquive} "
                    f"recherche_b={(trace.get('recherche_b') or {}).get('effectuee')}")
        return {
            "answer": answer,
            "citations": result.get("sources", []),
            "verdict": verdict,
            "confidence": confidence,
            "esquive": esquive,
            "trace_id": trace_id,
            "passages": projeter(result.get("search_results", [])),
            "trace": trace,
            "model": (result.get("metadata") or {}).get("model", ""),
        }
