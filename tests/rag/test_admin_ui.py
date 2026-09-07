"""UI d'administration (couche 4) : supervision du watcher et RAGAS.

La V3 pilotait l'ingestion depuis cette page ; en V4 le watcher en est la
seule voie, l'admin la supervise. Aucun reseau, aucun GPU, aucun modele.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


class _Reponse:
    def __init__(self, charge, status=200):
        self._charge, self.status_code, self.text = charge, status, str(charge)

    def json(self):
        return self._charge


class _ClientFactice:
    appels = []
    reponses = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, methode, url, **k):
        _ClientFactice.appels.append((methode, url))
        return _ClientFactice.reponses.get(url, _Reponse({}))

    async def get(self, url, **k):
        _ClientFactice.appels.append(("GET", url))
        return _ClientFactice.reponses.get(url, _Reponse({}))


@pytest.fixture
def module(tmp_path, monkeypatch):
    journal = tmp_path / "watcher.log"
    journal.write_text("ligne une\nligne deux\nligne trois\n", encoding="utf-8")
    monkeypatch.setenv("WATCHER_LOG_PATH", str(journal))
    monkeypatch.setenv("INSTANCE_NAME", "brissy")
    monkeypatch.setenv("WATCHER_URL", "http://watcher:8090")
    import src.ui.admin_ui as adm
    adm = importlib.reload(adm)
    _ClientFactice.appels = []
    _ClientFactice.reponses = {}
    monkeypatch.setattr("httpx.AsyncClient", _ClientFactice)
    return adm


class TestSupervisionWatcher:
    def test_etat_transmis_depuis_le_watcher(self, module):
        _ClientFactice.reponses["http://watcher:8090/api/watcher/status"] = _Reponse(
            {"running": True, "documents": {"indexed": 9}})
        d = TestClient(module.app).get("/api/watcher/status").json()
        assert d["documents"]["indexed"] == 9

    def test_watcher_injoignable_dit_pourquoi(self, module, monkeypatch):
        async def _echec(self, methode, url, **k):
            raise RuntimeError("connexion refusee")
        monkeypatch.setattr(_ClientFactice, "request", _echec)
        d = TestClient(module.app).get("/api/watcher/status").json()
        assert d["error"] == "watcher injoignable" and "connexion" in d["detail"]

    def test_journal_lu_dans_le_fichier_pas_via_docker(self, module):
        """Le socket Docker donnerait a l'interface le controle du demon."""
        d = TestClient(module.app).get("/api/watcher/logs?lines=2").json()
        assert d["lines"] == ["ligne deux", "ligne trois"]

    def test_journal_absent_signale_sans_planter(self, module, monkeypatch):
        monkeypatch.setattr(module.os.path, "exists", lambda p: False)
        d = TestClient(module.app).get("/api/watcher/logs").json()
        assert d["error"] == "journal absent" and d["lines"] == []

    def test_l_ingestion_ne_se_commande_plus_depuis_l_admin(self, module):
        """En V4 le watcher est la seule voie d'entree des documents."""
        chemins = {r.path for r in module.app.routes if hasattr(r, "path")}
        assert "/api/ingest" not in chemins
        assert "/api/count-files" not in chemins


class TestStatistiques:
    def test_comptage_sans_charger_de_modele(self, module, tmp_path, monkeypatch):
        """La V3 instanciait le pipeline, donc bge-m3, pour compter."""
        reglages = tmp_path / "settings.yaml"
        reglages.write_text(
            "qdrant:\n  collection_name: brissy\nopensearch:\n  index_name: brissy\n"
            "embedding:\n  model: BAAI/bge-m3\nchunking:\n  chunk_size: 1024\n",
            encoding="utf-8")
        monkeypatch.setattr(module, "_find_settings_yaml", lambda: str(reglages))
        _ClientFactice.reponses["http://qdrant:6333/collections/brissy"] = _Reponse(
            {"result": {"points_count": 4151}})
        _ClientFactice.reponses["http://opensearch:9200/brissy/_count"] = _Reponse(
            {"count": 4151})
        d = TestClient(module.app).get("/api/stats").json()
        assert d["qdrant_vectors"] == 4151 and d["opensearch_documents"] == 4151
        assert d["chunk_size"] == 1024
        assert "error" not in d


class TestRagas:
    def test_les_routes_ragas_repondent_sans_base(self, module):
        c = TestClient(module.app)
        assert c.get("/api/admin/ragas/scores").json()["count"] == 0
        assert "summary" in c.get("/api/admin/ragas/summary").json()
        assert c.get("/api/admin/ragas/history").json() == {"queries": []}
