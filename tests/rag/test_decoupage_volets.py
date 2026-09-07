"""Decoupage d'une remarque en volets avant la recherche A (v2.13).

Cause mesuree le 7 septembre 2026 (Brissy R7, Beaumont Sud R10) : une
remarque a plusieurs volets, ou dont le vocabulaire dominant decrit la
methode attendue plutot que son objet, produit une requete unique qui
part sur le mauvais champ lexical. Ces tests verifient le remede sans
GPU ni moteur de recherche.
"""
import json
from unittest.mock import MagicMock

from src.config.settings import DEFAULTS, Settings, _fusionner
from src.rag.query2 import IterativePipeline
from src.rag.retriever import Retriever

REMARQUE = ("L'autorite environnementale recommande d'etudier l'evitement en "
            "completant l'etude de variantes, et d'ajuster le plan d'arret "
            "des machines.")
VOLETS = ["l'etude de variantes d'implantation", "le plan d'arret des machines"]
COUVERT = '{"verdict": "COUVERT", "manques": [], "requetes": []}'


def _pipeline(reponse_volets, query2=None, verdict=COUVERT):
    """Pipeline sur doublures. reponse_volets : sortie brute du LLM pour
    l'appel de decoupage (les autres appels JSON rendent le verdict)."""
    cfg = {"catalogue_couverture": False, "garde_contradiction": False}
    cfg.update(query2 or {})
    settings = Settings(_fusionner(DEFAULTS, {"query2": cfg}))
    hybrid = MagicMock()
    hybrid.search.side_effect = lambda q, **kw: [
        {"chunk_id": f"{q[:24]}-{i}", "text": f"passage {q} {i} " * 5,
         "file_name": "Tome_4.pdf", "metadata": {"page_start": i, "page_end": i}}
        for i in range(20)]
    hybrid.bm25_search = None
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, res, top_n: res[:top_n]
    llm = MagicMock()

    def call_llm(system, prompt):
        if "volet" in prompt.lower():          # seul le prompt de decoupage en parle
            return reponse_volets
        if "JSON" in prompt:
            return verdict
        return "le plan d'arret des machines"

    llm.call_llm.side_effect = call_llm
    llm.generate.side_effect = lambda *a, **k: {
        "response": "Pour rappel, le Tome 4 p. 3 detaille la mesure. " * 30,
        "sources": [], "confidence": 0.8, "model": "m"}
    retr = Retriever(lambda: settings, hybrid, llm, reranker)
    return IterativePipeline(retr, query2_config=settings.section("query2")), hybrid


class TestDecoupage:
    def test_deux_volets_donnent_chacun_leur_recherche(self):
        pipe, hybrid = _pipeline(json.dumps({"volets": VOLETS}))
        out = pipe.run(query=REMARQUE)
        cherchees = [c.args[0] for c in hybrid.search.call_args_list]
        for volet in VOLETS:
            assert volet in cherchees, f"le volet '{volet}' doit avoir sa propre recherche"
        assert REMARQUE in cherchees, "la voie generale interroge la remarque entiere"
        assert out["iterative"]["volets"] == VOLETS

    def test_chaque_volet_a_des_places_garanties(self):
        pipe, _ = _pipeline(json.dumps({"volets": VOLETS}))
        out = pipe.run(query=REMARQUE)
        proteges = out["iterative"]["recherche_a"]["proteges"]
        par_volet = {v: sum(1 for p in proteges if p["requete"] == v) for v in VOLETS}
        assert all(n > 0 for n in par_volet.values()), par_volet
        assert par_volet[VOLETS[0]] == par_volet[VOLETS[1]], "quota identique par volet"
        # Le quota laisse sa part a la voie generale (15 passages, 2 volets).
        assert len(proteges) < len(out["search_results"])

    def test_un_seul_volet_ne_change_rien(self):
        """Propriete de securite : comportement identique aux versions
        precedentes des qu'il n'y a pas plusieurs volets."""
        pipe, hybrid = _pipeline(json.dumps({"volets": ["les mesures de bridage"]}))
        out = pipe.run(query=REMARQUE)
        assert out["iterative"]["recherche_a"] == {"passages": len(out["search_results"])}
        assert [c.args[0] for c in hybrid.search.call_args_list] == [REMARQUE]

    def test_json_illisible_retombe_sur_le_comportement_d_avant(self):
        pipe, hybrid = _pipeline("je ne sais pas decouper")
        out = pipe.run(query=REMARQUE)
        assert out["iterative"]["volets"] == []
        assert [c.args[0] for c in hybrid.search.call_args_list] == [REMARQUE]

    def test_volet_incoherent_ecarte(self):
        """Le garde-fou linguistique de l'extraction de sujet s'applique
        aux volets : un volet corrompu ne devient pas une requete."""
        pipe, _ = _pipeline(json.dumps(
            {"volets": [VOLETS[0], "\u00a5\u00e5\u00bf\u7f3a\u00b1 charabia"]}))
        out = pipe.run(query=REMARQUE)
        assert out["iterative"]["volets"] == [VOLETS[0]]

    def test_desactivable_par_configuration(self):
        pipe, hybrid = _pipeline(json.dumps({"volets": VOLETS}),
                                 query2={"decoupage_volets": False})
        out = pipe.run(query=REMARQUE)
        assert out["iterative"]["volets"] == []
        assert [c.args[0] for c in hybrid.search.call_args_list] == [REMARQUE]

    def test_max_volets_borne_le_decoupage(self):
        pipe, _ = _pipeline(json.dumps({"volets": VOLETS + ["un troisieme sujet"]}),
                            query2={"max_volets": 2})
        assert pipe.run(query=REMARQUE)["iterative"]["volets"] == VOLETS

    def test_recherche_b_reste_tracee_a_part(self):
        """Les volets alimentent la recherche A ; la recherche B garde sa
        propre trace quand la couverture est PARTIEL."""
        pipe, _ = _pipeline(
            json.dumps({"volets": VOLETS}),
            verdict='{"verdict": "PARTIEL", "manques": ["bridage"], "requetes": ["bridage chiropteres"]}')
        trace = pipe.run(query=REMARQUE)["iterative"]
        assert trace["recherche_a"]["mode"] == "quota_reserve"
        assert trace["recherche_b"]["effectuee"] is True
        assert trace["recherche_b"]["requetes"] == ["bridage chiropteres"]
