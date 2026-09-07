"""UI de chat (couche 3) : traduction entre le front, ecrit pour la V3,
et le contrat V4 du coeur RAG. Aucun reseau, aucun GPU."""
import pytest
from fastapi.testclient import TestClient

from src.ui import chat_ui


class _Reponse:
    def __init__(self, charge, status=200, texte=""):
        self._charge, self.status_code, self.text = charge, status, texte or str(charge)

    def json(self):
        return self._charge


class _ClientFactice:
    """Doublure d'httpx.AsyncClient : enregistre l'appel sortant."""
    appels = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _ClientFactice.appels.append((url, json))
        return _ClientFactice.reponse_post

    async def get(self, url):
        _ClientFactice.appels.append((url, None))
        return _ClientFactice.reponse_get


REPONSE_V4 = {
    "answer": "Le bridage est detaille au Tome 4 p. 243.",
    "citations": [{"file_name": "Tome_4.pdf"}],
    "verdict": "SUFFISANT",
    "confidence": 0.8,
    "esquive": False,
    "trace_id": "abc-123",
    "passages": [{"file_name": "Tome_4.pdf", "page": 243, "text": "..."}],
    "trace": {"couverture": {"verdict": "COUVERT"}},
    "model": "qwen2.5:14b-instruct-q4_K_M",
}


@pytest.fixture
def client(monkeypatch):
    _ClientFactice.appels = []
    _ClientFactice.reponse_post = _Reponse(REPONSE_V4)
    _ClientFactice.reponse_get = _Reponse({"instance": "brissy"})
    monkeypatch.setattr(chat_ui.httpx, "AsyncClient", _ClientFactice)
    return TestClient(chat_ui.app)


class TestProxyQuery:
    def test_la_demande_est_traduite_au_contrat_v4(self, client):
        client.post("/api/query", json={"query": "Quel bridage ?", "deep_search": True,
                                        "history": [{"role": "user", "content": "bonjour"}]})
        url, charge = _ClientFactice.appels[-1]
        assert url.endswith("/api/rag/query"), "V4 : plus d'agent ni de /api/query2"
        assert charge["question"] == "Quel bridage ?", "query -> question"
        assert charge["deep"] is True, "deep_search -> deep"
        assert charge["history"] == [{"role": "user", "content": "bonjour"}]
        assert "index_name" not in charge and "top_k" not in charge, \
            "une instance V4 = un index ; l'entonnoir vient de settings.yaml"

    def test_la_reponse_est_rendue_au_format_attendu_par_le_front(self, client):
        d = client.post("/api/query", json={"query": "Quel bridage ?"}).json()
        assert d["response"] == REPONSE_V4["answer"], "answer -> response"
        assert d["sources"] == REPONSE_V4["citations"], "citations -> sources"
        assert d["passages"] == REPONSE_V4["passages"]
        assert d["query"] == "Quel bridage ?"
        assert isinstance(d["processing_time_ms"], int), "duree mesuree par l'UI"
        # Champs V4 transmis tels quels, disponibles sans nouvel appel.
        assert d["verdict"] == "SUFFISANT" and d["trace_id"] == "abc-123"

    def test_erreur_http_du_coeur_rag_affichee_sans_planter(self, client):
        _ClientFactice.reponse_post = _Reponse({}, status=500, texte="boom")
        d = client.post("/api/query", json={"query": "Quel bridage ?"}).json()
        assert d["error"] == "HTTP 500"
        assert "500" in d["response"], "le front affiche toujours quelque chose"

    def test_delai_depasse_rend_un_message_lisible(self, client, monkeypatch):
        async def _timeout(self, url, json=None):
            raise chat_ui.httpx.TimeoutException("trop long")
        monkeypatch.setattr(_ClientFactice, "post", _timeout)
        d = client.post("/api/query", json={"query": "Quel bridage ?"}).json()
        assert d["error"] == "timeout" and "delai" in d["response"]


class TestIndexes:
    def test_index_unique_nomme_par_l_instance(self, client):
        d = client.get("/api/indexes").json()
        assert d["indexes"] == ["brissy"] and d["default"] == "brissy"
        assert d["single_index_mode"] is True, "pas de selecteur d'index en V4"

    def test_coeur_rag_injoignable_garde_le_nom_local(self, client, monkeypatch):
        """Le chat doit s'afficher meme quand le coeur RAG est arrete."""
        async def _echec(self, url):
            raise RuntimeError("rag arrete")
        monkeypatch.setattr(_ClientFactice, "get", _echec)
        assert client.get("/api/indexes").json()["default"] == chat_ui.SERVICE_NAME


class TestFeedback:
    def test_sans_service_de_feedback_le_bouton_est_masque(self, client, monkeypatch):
        monkeypatch.setattr(chat_ui, "FEEDBACK_URL", "")
        assert client.get("/api/feedback/config").json() == {"enabled": False, "key_users": []}
