"""Service RAG : assemble les composants et expose le contrat de reponse.

Contrat (docs/rag-ingestion-spec.md, section 7, complete) :

  {answer, citations[], verdict, confidence, controles, trace_id, passages[], trace}

- verdict : SUFFISANT / PARTIEL / INSUFFISANT — projection du verdict de
  couverture de query2 v3.1 (COUVERT / PARTIEL / NON_COUVERT). Le verdict
  est calcule par query2 a partir de gardes deterministes par bloc
  (esquive, troncature, degenerescence, pages citees hors lot) ; le
  service ne le recalcule pas et n'applique plus de detection d'esquive
  sur le texte concatene (les titres de section masquaient les blocs
  suivants). Sans trace exploitable, le verdict est PARTIEL, jamais
  SUFFISANT par defaut.
- confidence : derivee du verdict et des gardes (0.8 / 0.5 / 0.2, moins
  0.1 par garde declenchee, plancher 0.1). C'est un indicateur de forme,
  pas une mesure d'adequation a la demande : l'UI ne devrait pas
  l'afficher comme telle.
- controles : gardes agregees (esquive, tronque, degenere, hors_lot,
  repli_elargi, relance, synthese) pour l'UI et l'evaluation.
- citations : etiquettes des passages recuperes (union des questions),
  pas les pages citees par le modele ; les pages citees absentes du lot
  sont dans controles.hors_lot.
- passages : ordre conserve ; la repartition par question est dans
  trace.reponses[].etiquettes.
"""

from __future__ import annotations

import uuid
from typing import Callable, Dict, List, Optional

from loguru import logger

from src.config.settings import Settings
from src.rag.passages import projeter
from src.rag.query2 import Pipeline

VERDICTS = {"COUVERT": "SUFFISANT", "PARTIEL": "PARTIEL", "NON_COUVERT": "INSUFFISANT"}
CONFIANCE = {"SUFFISANT": 0.8, "PARTIEL": 0.5, "INSUFFISANT": 0.2}


class RagService:
    def __init__(self, settings_provider: Callable[[], Settings], retriever):
        self._settings = settings_provider
        self.retriever = retriever

    def query(self, question: str, custom_prompt: Optional[str] = None,
              history: Optional[List[Dict]] = None, deep: bool = False,
              filters: Optional[Dict] = None) -> Dict:
        settings = self._settings()
        pipeline = Pipeline(self.retriever, query2_config=settings.section("query2"))
        result = pipeline.run(query=question, custom_prompt=custom_prompt,
                              history=history, deep=deep)
        answer = result.get("response", "")
        trace = result.get("iterative", {}) or {}
        blocs = trace.get("reponses") or []

        couverture = (trace.get("couverture") or {}).get("verdict")
        if couverture not in VERDICTS:
            logger.warning(f"rag: verdict de couverture absent ou inconnu ({couverture!r}), PARTIEL")
            couverture = "PARTIEL"
        verdict = VERDICTS[couverture]

        gardes = trace.get("gardes") or {}
        controles = {
            "esquive": bool(gardes.get("esquive")),
            "tronque": bool(gardes.get("tronque")),
            "degenere": bool(gardes.get("degenere")),
            "hors_lot": list(gardes.get("hors_lot") or []),
            "repli_elargi": any(b.get("repli_elargi") for b in blocs),
            "relance": any(b.get("relance") for b in blocs),
            "synthese": bool((trace.get("synthese") or {}).get("effectuee")),
        }
        declenchees = sum(1 for k in ("esquive", "tronque", "degenere") if controles[k]) \
            + (1 if controles["hors_lot"] else 0)
        confidence = round(max(0.1, CONFIANCE[verdict] - 0.1 * declenchees), 2)

        trace_id = str(uuid.uuid4())
        logger.info(
            f"rag[{trace_id[:8]}] verdict={verdict} questions={len(blocs)} "
            f"origine={trace.get('origine_questions')} esquive={controles['esquive']} "
            f"tronque={controles['tronque']} degenere={controles['degenere']} "
            f"hors_lot={len(controles['hors_lot'])} repli={controles['repli_elargi']} "
            f"relance={controles['relance']} synthese={controles['synthese']}")
        return {
            "answer": answer,
            "citations": result.get("sources", []),
            "verdict": verdict,
            "confidence": confidence,
            "esquive": controles["esquive"],  # compatibilite UI
            "controles": controles,
            "trace_id": trace_id,
            "passages": projeter(result.get("search_results", [])),
            "trace": trace,
            "model": (result.get("metadata") or {}).get("model", ""),
        }
