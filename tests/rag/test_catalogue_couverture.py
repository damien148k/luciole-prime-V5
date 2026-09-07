"""Catalogue de titres dans l'analyse de couverture (query2 v2.9).

Régression du 15 août 2026 sur la Panière du Fort : « quels sont les
enjeux paysagers ? » restitué uniquement depuis le RNT (tome de
synthèse), sans le « 5 - ... Volet paysager et patrimonial.pdf ».
L'analyse de couverture, aveugle à l'inventaire du corpus, a rendu un
verdict d'apparence complète et n'a jamais déclenché la seconde passe.

La v2.9 injecte les titres du corpus dans le prompt de couverture avec
une règle d'inventaire : un tome pertinent absent des passages force un
verdict PARTIEL, même si la synthèse paraît suffisante.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from src.rag.catalogue import CatalogueDocuments  # noqa: E402
from src.rag.query2 import (  # noqa: E402
    CATALOGUE_PROMPT_MAX_TITRES,
    COVERAGE_USER_TEMPLATE,
    COVERAGE_USER_TEMPLATE_CATALOGUE,
    IterativePipeline,
    _TERMES_STRUCTURANTS_BASE,
    _formater_titres,
    _mots_contenu,
    _regle_inventaire,
    _sujet_enjeux,
)

TITRES_CORPS = [
    "1 - PE de la Panière du Fort - Volet projet.pdf",
    "3 - PE de la Panière du Fort - Volet environnement naturel.pdf",
    "5 - PE de la Panière du Fort - Volet paysager et patrimonial.pdf",
    "PE de la Panière du Fort - RNT Etude d'impact.pdf",
]


class _ClientFactice:
    """Réponse d'agrégation OpenSearch minimale."""

    def __init__(self, buckets=None, leve=False):
        self._buckets = buckets if buckets is not None else list(TITRES_CORPS)
        self._leve = leve
        self.appels = 0

    def search(self, index, body):
        self.appels += 1
        if self._leve:
            raise RuntimeError("OpenSearch indisponible")
        assert body["size"] == 0
        return {
            "aggregations": {
                "titres": {
                    "buckets": [{"key": t, "doc_count": 12} for t in self._buckets]
                }
            }
        }


class _BM25Factice:
    def __init__(self, client):
        self.client = client
        self.index_name = "documents_bm25"


class _HybrideFactice:
    def __init__(self, bm25):
        self.bm25_search = bm25


class _LLMFactice:
    """Enregistre les prompts, rend un verdict piloté."""

    def __init__(self, verdict="COUVERT"):
        self.reponses = [
            '{"verdict": "%s", "manques": [], "requetes": []}' % verdict
        ]
        self.prompts = []

    def call_llm(self, system, prompt):
        self.prompts.append(prompt)
        return self.reponses.pop(0)


class _AnalyzerFactice:
    def __init__(self, bm25, llm):
        self.hybrid_search = _HybrideFactice(bm25)
        self.llm_generator = llm


def _pipeline(analyzer, catalogue=None):
    pipe = IterativePipeline(analyzer, catalogue=catalogue)
    return pipe


# ---------------------------------------------------------------------
# CatalogueDocuments
# ---------------------------------------------------------------------

def test_titres_tries_et_deduits_des_buckets():
    client = _ClientFactice(buckets=list(reversed(TITRES_CORPS)))
    catalogue = CatalogueDocuments(_BM25Factice(client))
    titres = catalogue.titres()
    assert titres == sorted(TITRES_CORPS)
    assert client.appels == 1


def test_cache_ttl_evite_les_relectures():
    client = _ClientFactice()
    catalogue = CatalogueDocuments(_BM25Factice(client), ttl_seconds=600)
    assert catalogue.titres() == sorted(TITRES_CORPS)
    assert catalogue.titres() == sorted(TITRES_CORPS)
    assert client.appels == 1


def test_cache_expire_relance_la_lecture():
    client = _ClientFactice()
    catalogue = CatalogueDocuments(_BM25Factice(client), ttl_seconds=0)
    catalogue.titres()
    catalogue.titres()
    assert client.appels == 2


def test_echec_lecture_sert_le_dernier_cache():
    client = _ClientFactice()
    catalogue = CatalogueDocuments(_BM25Factice(client), ttl_seconds=0)
    premier = catalogue.titres()
    client._leve = True
    assert catalogue.titres() == premier  # repli sur le cache


def test_echec_lecture_sans_cache_retourne_vide():
    catalogue = CatalogueDocuments(_BM25Factice(_ClientFactice(leve=True)))
    assert catalogue.titres() == []


def test_sans_bm25_retourne_vide():
    assert CatalogueDocuments(None).titres() == []


# ---------------------------------------------------------------------
# Formatage du prompt
# ---------------------------------------------------------------------

def test_formater_titres_a_puces():
    rendu = _formater_titres(["b.pdf", "a.pdf"])
    assert rendu == "- b.pdf\n- a.pdf"


def test_formater_titres_plafonne_avec_mention_du_reliquat():
    titres = [f"tome {i:03d}.pdf" for i in range(CATALOGUE_PROMPT_MAX_TITRES + 7)]
    rendu = _formater_titres(titres)
    assert rendu.count("\n- ") == CATALOGUE_PROMPT_MAX_TITRES
    assert "7 autres documents" in rendu


def test_variante_catalogue_conserve_la_spec_json():
    """Le corps de la consigne et les clés JSON restent ceux de la v2.8."""
    assert COVERAGE_USER_TEMPLATE_CATALOGUE != COVERAGE_USER_TEMPLATE
    for marqueur in ('"verdict"', '"manques"', '"requetes"',
                     "COUVERT", "PARTIEL", "NON_COUVERT", "{query}",
                     "{passages}", "{max_q}"):
        assert marqueur in COVERAGE_USER_TEMPLATE_CATALOGUE
    assert "Inventaire des documents du corpus" in (
        COVERAGE_USER_TEMPLATE_CATALOGUE
    )
    assert "{titres}" in COVERAGE_USER_TEMPLATE_CATALOGUE


# ---------------------------------------------------------------------
# Analyse de couverture
# ---------------------------------------------------------------------

def test_prompt_couverture_embarque_l_inventaire():
    llm = _LLMFactice()
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Les enjeux sont évalués selon plusieurs niveaux...",
          "file_name": TITRES_CORPS[3], "metadata": {"page_start": 20}}],
    )
    prompt = llm.prompts[0]
    assert "Inventaire des documents du corpus" in prompt
    assert "- 5 - PE de la Panière du Fort - Volet paysager et patrimonial.pdf" in prompt
    assert "Règle d'inventaire" in prompt
    assert couverture["catalogue_titres"] == len(TITRES_CORPS)


def test_verdict_partiel_propage_avec_catalogue():
    """Le tome paysager absent des passages déclenche la seconde passe."""
    llm = _LLMFactice(verdict="PARTIEL")
    llm.reponses = [
        '{"verdict": "PARTIEL", "manques": ["enjeux paysagers"],'
        ' "requetes": ["enjeux paysager patrimonial"]}'
    ]
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Synthèse du RNT...", "file_name": TITRES_CORPS[3]}],
    )
    assert couverture["verdict"] == "PARTIEL"
    assert couverture["requetes"] == ["enjeux paysager patrimonial"]


def test_sans_catalogue_le_prompt_est_identique_a_la_v28():
    """Indisponibilité du catalogue = repli strict sur l'ancien prompt."""
    llm = _LLMFactice()
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice(leve=True)), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Synthèse du RNT...", "file_name": TITRES_CORPS[3]}],
    )
    prompt = llm.prompts[0]
    assert "Inventaire des documents du corpus" not in prompt
    assert couverture["catalogue_titres"] == 0


def test_catalogue_desactive_par_environnement(monkeypatch):
    """QUERY2_CATALOGUE_COUVERTURE=false rend le comportement v2.8."""
    monkeypatch.setenv("QUERY2_CATALOGUE_COUVERTURE", "false")
    # Le flag est lu à l'instanciation du pipeline : on force le défaut
    # que la config YAML surchargera (le module ne publie plus de
    # constante CATALOGUE_COUVERTURE_ACTIF — cassé depuis le commit
    # « réglages à chaud » du 22 août 2026, où l'override par défaut est
    # devenu la variable privée _CATALOGUE_COUVERTURE_ENV).
    monkeypatch.setattr("src.rag.query2._CATALOGUE_COUVERTURE_ENV", False)
    llm = _LLMFactice()
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Synthèse du RNT...", "file_name": TITRES_CORPS[3]}],
    )
    assert "Inventaire" not in llm.prompts[0]
    assert couverture["catalogue_titres"] == 0


# ---------------------------------------------------------------------
# Règle d'inventaire déterministe (v2.10)
# ---------------------------------------------------------------------

def test_sujet_enjeux_extrait_le_sujet():
    assert _sujet_enjeux("quels sont les enjeux paysagers ?") == "paysagers"
    assert _sujet_enjeux("quel est l'enjeu acoustique ?") == "acoustique"


def test_sujet_enjeux_absent_sans_mot_enjeu():
    assert _sujet_enjeux("que dit le dossier sur le paysage ?") is None
    assert _sujet_enjeux("quels sont les impacts du projet ?") is None


def test_mots_contenu_normalise_accents_et_pluriels():
    assert "paysagers" in _mots_contenu("Enjeux paysagers")
    assert "paysager" in _mots_contenu("Volet paysager et patrimonial")
    assert "paniere" in _mots_contenu("La Panière-du-Fort")


def test_regle_inventaire_detecte_le_tome_absent():
    """Cas mesuré : « enjeux paysagers » sans le volet paysager.

    Le corpus est celui de la Panière du Fort : les termes propres au
    projet sont explicitement déclarés (rôle de
    query2.termes_structurants dans settings.yaml, qui restait vide dans
    la constante partagée — d'où l'échec de ce test depuis le 22 août
    2026, passé inaperçu).
    """
    passages = [
        {"file_name": TITRES_CORPS[3]},  # RNT
        {"file_name": TITRES_CORPS[0]},  # volet projet
    ]
    forcee = _regle_inventaire(
        "quels sont les enjeux paysagers ?", passages, TITRES_CORPS,
        termes_structurants=_TERMES_STRUCTURANTS_BASE | {"fort", "paniere"},
    )
    assert forcee is not None
    assert forcee["verdict"] == "PARTIEL"
    assert forcee["manques"] == [
        "5 - PE de la Panière du Fort - Volet paysager et patrimonial.pdf"
    ]
    requete = forcee["requetes"][0]
    assert "patrimonial" in requete and "paysager" in requete
    # dédupliqué par racine : pas « paysagers paysager »
    assert len([m for m in requete.split() if m.startswith("paysager")]) == 1
    # les termes structurants n'entrent pas dans la requête
    for terme in ("volet", "paniere", "fort", "pdf"):
        assert terme not in requete


def test_regle_inventaire_muette_quand_le_tome_est_present():
    passages = [{"file_name": t} for t in TITRES_CORPS]
    assert _regle_inventaire(
        "quels sont les enjeux paysagers ?", passages, TITRES_CORPS
    ) is None


def test_regle_inventaire_muette_sans_mot_enjeu():
    """Une question non-enjeux ne déclenche jamais la règle."""
    passages = [{"file_name": TITRES_CORPS[3]}]
    assert _regle_inventaire(
        "que dit le dossier sur le paysage ?", passages, TITRES_CORPS
    ) is None


def test_regle_inventaire_muette_sans_titre_du_sujet():
    """Aucun tome ne porte le sujet : pas de déclenchement."""
    passages = [{"file_name": TITRES_CORPS[3]}]
    assert _regle_inventaire(
        "quels sont les enjeux ferroviaires ?", passages, TITRES_CORPS
    ) is None


def test_regle_inventaire_muette_sans_catalogue():
    passages = [{"file_name": TITRES_CORPS[3]}]
    assert _regle_inventaire(
        "quels sont les enjeux paysagers ?", passages, []
    ) is None


def test_analyse_couverture_force_partiel_malgre_llm_couvert():
    """Le LLM maintient COUVERT : le code force PARTIEL (cas mesuré)."""
    llm = _LLMFactice(verdict="COUVERT")
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Synthèse du RNT...", "file_name": TITRES_CORPS[3]}],
    )
    assert couverture["verdict"] == "PARTIEL"
    assert couverture["requetes"]
    assert "patrimonial" in couverture["requetes"][0]


def test_analyse_couverture_llm_partiel_conserve_ses_requetes():
    """PARTIEL rendu par le LLM : la règle n'écrase pas ses requêtes."""
    llm = _LLMFactice()
    llm.reponses = [
        '{"verdict": "PARTIEL", "manques": ["photomontages"],'
        ' "requetes": ["photomontages vues belvedere"]}'
    ]
    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), llm)
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture(
        "quels sont les enjeux paysagers ?",
        [{"text": "Synthèse du RNT...", "file_name": TITRES_CORPS[3]}],
    )
    assert couverture["requetes"] == ["photomontages vues belvedere"]


def test_echec_llm_conserve_le_repli_couvert_et_la_trace():
    class _LLMPanne(_LLMFactice):
        def call_llm(self, system, prompt):
            raise RuntimeError("LLM indisponible")

    analyzer = _AnalyzerFactice(_BM25Factice(_ClientFactice()), _LLMPanne())
    pipe = _pipeline(analyzer)
    couverture = pipe._analyse_couverture("demande ?", [{"text": "x"}])
    assert couverture["verdict"] == "COUVERT"
    assert couverture["catalogue_titres"] == len(TITRES_CORPS)
