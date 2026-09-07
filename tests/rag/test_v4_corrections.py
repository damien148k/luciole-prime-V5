"""Tests des correctifs V4 sur le coeur RAG (defauts 2, 5, 6) et du
contrat d'API, sans GPU ni moteurs de recherche."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config.settings import DEFAULTS, Settings, _fusionner
from src.rag import api as api_module
from src.rag.query2 import _reponse_esquive, sans_accents
from src.rag.reranker import Reranker
from src.rag.retriever import Retriever
from src.rag.service import RagService


# ---------------------------------------------------------------- defaut 2
class TestEsquiveAccents:
    def test_forme_accentuee_detectee(self):
        texte = "Les études ne précisent pas la distance aux habitations. " + "x " * 400
        assert _reponse_esquive(texte) is True

    def test_aucun_element_accentue(self):
        assert _reponse_esquive("Aucun élément du dossier n'aborde ce point. " + "y " * 400) is True

    def test_reponse_sourcee_qui_concede_a_la_fin(self):
        corps = "Pour rappel, le Tome 4 p. 12 indique la mesure ECO-E1. " * 20
        assert _reponse_esquive(corps + " Toutefois le dossier ne précise pas la date.") is False

    def test_apostrophe_typographique(self):
        assert "n'evoque" in sans_accents("n’évoque")


# ---------------------------------------------------------------- defaut 5
class TestRerankerSansEtat:
    def _reranker(self):
        modele = MagicMock()
        modele.predict.side_effect = lambda pairs, **_: [float(len(p[1])) for p in pairs]
        return Reranker(device="cpu", model=modele)

    def test_top_n_obligatoire(self):
        with pytest.raises(TypeError):
            self._reranker().rerank("q", [{"text": "a"}])  # noqa: E1120

    def test_top_n_vient_de_l_appelant(self):
        r = self._reranker()
        docs = [{"text": "a"}, {"text": "abc"}, {"text": "ab"}]
        assert [d["text"] for d in r.rerank("q", docs, top_n=2)] == ["abc", "ab"]


class TestRetrieverLitSettings:
    def test_reload_vu_a_la_requete_suivante(self):
        etat = {"s": Settings(_fusionner(DEFAULTS, {"retrieval": {"rerank_top_n": 3}}))}
        hybrid = MagicMock()
        hybrid.search.return_value = [{"text": f"p{i}", "file_name": "T.pdf", "metadata": {}} for i in range(10)]
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, res, top_n: res[:top_n]
        llm = MagicMock()
        llm.generate.return_value = {"response": "ok", "sources": [], "confidence": 0.8, "model": "m"}
        retr = Retriever(lambda: etat["s"], hybrid, llm, reranker)

        retr.analyze("q")
        assert reranker.rerank.call_args.kwargs["top_n"] == 3
        etat["s"] = Settings(_fusionner(DEFAULTS, {"retrieval": {"rerank_top_n": 7}}))
        retr.analyze("q")
        assert reranker.rerank.call_args.kwargs["top_n"] == 7


# ---------------------------------------------------------------- defaut 6 + contrat
def _service_factice(verdict_json: str, reponse_a: str, reponse_b: str = "B"):
    settings = Settings(_fusionner(DEFAULTS, {"query2": {"catalogue_couverture": False,
                                                          "garde_contradiction": True}}))
    hybrid = MagicMock()
    hybrid.search.side_effect = lambda q, **kw: [
        {"chunk_id": f"{q[:6]}-{i}", "text": f"passage {q} {i} " * 5, "file_name": "Tome_4.pdf",
         "metadata": {"page_start": i, "page_end": i}} for i in range(8)]
    hybrid.bm25_search = None
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, res, top_n: res[:top_n]
    llm = MagicMock()
    llm.call_llm.side_effect = lambda system, prompt: (
        verdict_json if "JSON" in prompt else "le bridage des chiroptères")
    llm.generate.side_effect = [
        {"response": reponse_a, "sources": [], "confidence": 0.8, "model": "m"},
        {"response": reponse_b, "sources": [{"file_name": "Tome_4.pdf"}], "confidence": 0.8, "model": "m"},
    ]
    retr = Retriever(lambda: settings, hybrid, llm, reranker)
    return RagService(lambda: settings, retr)


class TestServiceContrat:
    def test_partiel_declenche_recherche_b_et_etiquette_cible(self):
        svc = _service_factice(
            '{"verdict": "PARTIEL", "manques": ["bridage"], "requetes": ["bridage chiropteres"]}',
            reponse_a="A", reponse_b="Pour rappel, le Tome 4 p. 3 detaille le bridage. " * 30)
        out = svc.query("La MRAe recommande de préciser les mesures de bridage.")
        assert out["verdict"] == "PARTIEL"
        assert out["trace"]["recherche_b"]["effectuee"] is True
        cibles = [p for p in out["passages"] if p.get("cible") == "bridage chiropteres"]
        assert len(cibles) == 5, "quota de 5 passages proteges, etiquetes par leur requete"
        assert out["passages"][0].get("cible"), "les passages cibles sont en tete, ordre conserve"
        for cle in ("answer", "citations", "verdict", "confidence", "trace_id", "passages"):
            assert cle in out

    def test_couvert_mais_esquive_degrade_le_verdict(self):
        svc = _service_factice('{"verdict": "COUVERT", "manques": [], "requetes": []}',
                               reponse_a="Le dossier ne précise pas ce point.",
                               reponse_b="Le dossier ne précise pas ce point.")
        out = svc.query("Remarque ?")
        assert out["esquive"] is True
        assert out["verdict"] == "PARTIEL"
        assert out["confidence"] <= 0.4


class TestApi:
    def test_contrat_http(self):
        api_module.app.state.service = _service_factice(
            '{"verdict": "COUVERT", "manques": [], "requetes": []}',
            reponse_a="Pour rappel, le Tome 4 p. 3 indique la mesure. " * 30)
        client = TestClient(api_module.app)
        r = client.post("/api/rag/query", json={"question": "Quelle mesure ?"})
        assert r.status_code == 200
        corps = r.json()
        assert corps["verdict"] == "SUFFISANT"
        assert corps["passages"][0]["page"] == 0
        assert client.get("/api/health").json()["loaded"] is True
        api_module.app.state.service = None
