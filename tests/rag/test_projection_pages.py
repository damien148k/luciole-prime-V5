"""Projection des numeros de page vers les passages exposes par l'API.

Regression de la PR #23 : le chunker posait page_start et page_end sur
chaque fragment PDF, mais la projection des passages ne recopiait que la
clef `page`, que les PDF ne produisent pas. Sur les vingt cas du jeu
MRAe, 292 passages remontes, zero avec un numero de page, alors que la
collection Qdrant portait bien la pagination sur tous les points
echantillonnes.

La fonction est extraite plutot que testee au travers de l'application
pour eviter d'importer FastAPI et la pile d'inference dans la suite.
"""

from src.rag.passages import _reporter_pages, projeter


class TestProjeter:

    def test_etiquette_cible_propagee(self):
        """Un passage remonte par une requete ciblee (recherche B) porte
        la requete qui l'a trouve : l'utilisateur sait pourquoi il est la."""
        brut = [
            {"text": "a", "file_name": "T4.pdf", "rerank_score": 0.9,
             "metadata": {"page_start": 3, "page_end": 3}, "quota_requete": "bridage chiropteres"},
            {"text": "b", "file_name": "RNT.pdf", "rrf_score": 0.02, "metadata": {}},
        ]
        p = projeter(brut)
        assert p[0]["cible"] == "bridage chiropteres"
        assert p[0]["page"] == 3
        assert "cible" not in p[1]
        assert p[1]["score"] == 0.02


class TestReporterPages:

    def test_plage_recopiee(self):
        p = {}
        _reporter_pages(p, {"page_start": 12, "page_end": 14})
        assert p["page_start"] == 12
        assert p["page_end"] == 14

    def test_page_deduite_du_debut(self):
        """Les affichages existants lisent `page` : elle doit rester posee."""
        p = {}
        _reporter_pages(p, {"page_start": 7, "page_end": 7})
        assert p["page"] == 7

    def test_page_explicite_prioritaire(self):
        """Un format qui pose `page` lui-meme n'est pas ecrase."""
        p = {}
        _reporter_pages(p, {"page": 3, "page_start": 9, "page_end": 9})
        assert p["page"] == 3
        assert p["page_start"] == 9

    def test_metadonnees_vides(self):
        p = {}
        _reporter_pages(p, {})
        assert "page" not in p
        assert "page_start" not in p

    def test_page_zero_est_une_valeur(self):
        """page_start peut valoir 0 : 0 est une valeur, pas une absence.
        Le test fige le choix de `is not None` plutot que d'un test de
        verite, qui est precisement l'erreur de l'ancienne projection."""
        p = {}
        _reporter_pages(p, {"page_start": 0, "page_end": 0})
        assert p["page_start"] == 0
        assert p["page"] == 0

    def test_debut_seul(self):
        p = {}
        _reporter_pages(p, {"page_start": 5})
        assert p["page_start"] == 5
        assert "page_end" not in p
        assert p["page"] == 5
