"""UI de recueil d'avis (couche 3) : stockage local et branchements V4.

Aucun reseau, aucun GPU. La base SQLite est creee dans un dossier
temporaire propre a chaque test.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def module(tmp_path, monkeypatch):
    """Recharge le module avec une base et une config a lui."""
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        "llm:\n  provider: ollama\n  model: qwen2.5:14b-instruct-q4_K_M\n",
        encoding="utf-8")
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "feedbacks.db"))
    monkeypatch.setenv("INSTANCE_NAME", "brissy")
    import src.ui.feedback_ui as fb
    fb = importlib.reload(fb)
    fb.CONFIG_FILES["settings.yaml"] = str(config / "settings.yaml")
    return fb


class TestAvis:
    def test_depot_relecture_et_statistiques(self, module):
        c = TestClient(module.app)
        depot = c.post("/api/feedback", json={
            "query": "Quel bridage ?", "response": "Tome 4 p. 243",
            "feedback": "up", "comment": "utile", "index_name": "brissy"}).json()
        assert depot["status"] == "success"
        liste = c.get("/api/feedbacks?limit=10").json()
        assert liste["total"] == 1
        assert liste["feedbacks"][0]["query"] == "Quel bridage ?"
        stats = c.get("/api/feedbacks/stats").json()
        assert stats["total"] == 1 and stats["up"] == 1 and stats["down"] == 0

    def test_config_annonce_le_service_actif(self, module):
        """L'UI de chat lit cette route pour afficher ou non le bouton."""
        assert TestClient(module.app).get("/api/feedback/config").json()["enabled"] is True


class TestModeleLLM:
    def test_le_modele_actif_est_lu_dans_settings(self, module):
        d = TestClient(module.app).get("/api/llm/model").json()
        assert d["model"] == "qwen2.5:14b-instruct-q4_K_M"
        assert d["backend"] == "ollama" and d["supports_hot_swap"] is True

    def test_settings_illisible_ne_fait_pas_tomber_la_page(self, module):
        module.CONFIG_FILES["settings.yaml"] = "/introuvable/settings.yaml"
        d = TestClient(module.app).get("/api/llm/model").json()
        assert d["model"] == "" and "backend" in d

    def test_recherche_au_registre_refusee_hors_ligne(self, module):
        r = TestClient(module.app).get("/api/ollama/search?q=qwen")
        assert r.status_code == 501, "une instance V4 tourne sans acces internet"

    def test_activation_sans_modele_refusee(self, module):
        r = TestClient(module.app).post("/api/ollama/activate", json={})
        assert r.status_code == 400


class TestConfiguration:
    def test_lecture_d_un_fichier_connu(self, module):
        d = TestClient(module.app).get("/api/config/settings.yaml").json()
        assert "qwen2.5" in d["content"] and d["readonly"] is False

    def test_fichier_inconnu_refuse(self, module):
        d = TestClient(module.app).get("/api/config/passwd").json()
        assert "error" in d, "seuls les fichiers declares sont lisibles"
