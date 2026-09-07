"""
Pipeline itératif v2.9 (endpoint expérimental /api/query2).

Conception : DESIGN_agent_v2_pipeline_iteratif.md (8 août 2026).

Structure fixe en 4 étapes, par opposition à la boucle libre plan/act/observe
de la route agent v1 (mesurée : 11 esquives sur 16 au benchmark MRAe) :

  1. RECHERCHE A + génération : appel identique à la route classique
     (analyzer.analyze mode chat). Si l'étape 2 conclut COUVERT, la réponse
     retournée EST celle de la route classique — la v2 ne peut pas faire
     moins bien que la classique (propriété de sécurité).
  2. ANALYSE DE COUVERTURE : 1 appel LLM qui lit les passages COMPLETS
     (pas des extraits de 400 caractères comme la boucle v1) et répond en
     JSON structuré : verdict, manques, requêtes ciblées. Depuis la v2.9,
     le prompt embarque l'inventaire des titres du corpus (catalogue) :
     un tome pertinent absent des passages force un verdict PARTIEL,
     même si la synthèse paraît complète (mesuré sur la Panière du
     Fort : « enjeux paysagers » restitué sans le volet paysager).
  3. RECHERCHE B (si besoin) : quota réservé + question construite.
     a. QUESTION CONSTRUITE : le LLM extrait le SUJET de la demande
        (3-6 mots, tâche bornée), puis le code assemble la question
        (« Que disent les documents sur {sujet} ? »). Sans cela, le LLM
        commente l'absence de la demande elle-même dans les documents —
        qui ne la contiennent jamais — au lieu de répondre sur le fond
        (mesuré : R11 esquivé malgré les bonnes pages dans les sources).
        La reformulation libre (v2.4) laissait parfois l'auteur de la
        demande devenir le sujet de la question ; l'extraction bornée
        supprime ce mode d'échec.
     b. QUOTA RÉSERVÉ : la fusion multi-query dilue les passages trouvés
        par les requêtes ciblées (mesuré : une page au rang 3 de sa
        requête tombe au rang 55 du pool fusionné, puis est éliminée par
        le reranker qui ne note que contre la demande d'origine). Chaque
        requête ciblée fait donc sa PROPRE recherche + rerank et réserve
        QUOTA_PAR_REQUETE places garanties dans le top final ; le reste
        vient de la recherche générale sur la question reformulée.
  4. GÉNÉRATION finale : prompt RAG classique, température 0 (mêmes
     appels que la route classique : _build_context + generate), avec
     la question reformulée comme « Question : ».

Plafond dur : 1 round itératif par défaut (2 recherches max au total).
L'esquive honnête reste possible si le corpus ne contient pas l'information
(comportement légitime mesuré : radar, Natura 2000).

v2.10 — règle d'inventaire déterministe : pour les interrogations sur
les ENJEUX d'un sujet, le code force PARTIEL quand un tome de
l'inventaire porte le sujet sans qu'aucun passage n'en provienne ; la
requête ciblée est construite par concaténation sujet + mots du titre,
jamais par le modèle. La règle ne voit que les tomes indexés : un
document non ingéré est absent de l'inventaire et lui reste invisible
(cas mesuré le 15 août 2026 sur la Panière du Fort : le volet paysager
manquait à l'index OpenSearch — le verdict COUVERT était alors correct
au vu du corpus réellement indexé).

v2.11 — règle de granularité dans le prompt de couverture : la présence
d'un tome pertinent dans les passages ne suffit plus, les extraits
doivent fournir le CONTENU au niveau de détail demandé. Mesuré le
15 août 2026 sur la campagne query2 du jeu MRAe (Panière du Fort) :
« décrire les sites et monuments historiques au sein des aires d'étude
immédiate, rapprochée et élargie » déclaré COUVERT avec des passages
liminaires du volet paysager (sommaire, synthèse) — la recherche B ne
se déclenchait jamais et l'esquive survivait aux deux bras, alors que
l'inventaire détaillé existe dans le tome. La règle vise les demandes
de liste ou d'inventaire : un extrait qui annonce le sujet, y renvoie
ou le résume sans le détailler ne compte pas comme couverture.

v2.12 — les demandes d'énumération survivent à la reformulation.
Mesuré le 17 août 2026 sur la Panière du Fort : « fait une liste des
monuments, sites et bâtiments présents autour du projet » — le sujet
extrait (« la liste des monuments, sites et bâtiments ») était rejeté
deux fois par _sujet_incoherent à cause des VIRGULES, le repli sur la
requête de couverture produisait « Que disent les documents sur
monuments historiques zone projet ? », et la génération répondait à
cette question étroite avec les passages réglementaires au lieu de
compiler l'inventaire présent dans les 25 passages. Trois correctifs :
la virgule entre dans la liste blanche du sujet, l'extraction reçoit
un exemple d'énumération (sans le wrapper « la liste des »), et les
demandes de liste sont assemblées avec « Quels sont {sujet} ? » —
l'intention d'inventaire atteint la génération."""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

from loguru import logger

from src.rag.catalogue import CatalogueDocuments


# Nombre de passages complets soumis à l'analyse de couverture.
# 12 passages x ~1200 caractères ~ 5k tokens : tient dans la fenêtre 16k
# avec le prompt et la réponse JSON.
COVERAGE_MAX_PASSAGES = 12
COVERAGE_PASSAGE_CHARS = 1200

# Nombre maximum de requêtes ciblées demandées à l'analyse de couverture.
COVERAGE_MAX_QUERIES = 3

# Découpage de la demande en volets avant la recherche A (v2.13).
# Mesuré le 7 septembre 2026 sur Brissy R7 et Beaumont Sud R10 : une
# remarque à plusieurs volets, ou dont le vocabulaire dominant décrit la
# METHODE attendue (cartes, photomontages) plutôt que son OBJET, produit
# une requête unique qui part sur le mauvais champ lexical. Recherche
# seule, marqueurs attendus dans le top 15 :
#   Beaumont R10 (variantes + plan d'arrêt) : la remarque entière ne
#     ramène ni ECO-R7 ni asservissement ni bridage ; le seul volet
#     « plan d'arrêt des machines bridage chiroptères » les ramène tous
#     au rang 1.
#   Brissy R7 (mesures de réduction démontrées par cartes et
#     photomontages) : la remarque entière ne ramène ni « plantation »
#     ni « entrée/sortie nord » (5 chunks sur 15 viennent du carnet de
#     photomontages) ; le volet « mesures de réduction paysagères
#     plantations » les ramène au rang 1.
# Le remède réutilise le quota réservé de la recherche B : une recherche
# par volet, rerankée contre le volet lui-même, avec des places garanties
# dans le top final. Une demande à un seul volet garde EXACTEMENT le
# comportement d'avant (propriété de sécurité, voir query()).
MAX_VOLETS = 3

# Réglable à chaud via settings.yaml (section query2) ; la variable
# d'environnement QUERY2_DECOUPAGE_VOLETS=false reste un repli.
_DECOUPAGE_VOLETS_ENV = (
    os.environ.get("QUERY2_DECOUPAGE_VOLETS", "true").lower() == "true"
)

# Places garanties dans le top final pour chaque requête ciblée
# (quota de diversité : empêche le sujet dominant d'écraser les manques).
QUOTA_PAR_REQUETE = 5

# Profil « recherche approfondie » (deep_search=true) : entonnoir de
# retrieval élargi pour remonter plus de passages, au prix de bruit et
# de latence. Ce sont les DÉFAUTS du mode deep ; chaque clé est
# surchargeable dans settings.yaml (section query2.deep, relue à chaque
# requête — un reload-config applique les edits sans redémarrage).
# Le mode standard (deep_search=false) reste gouverné par les constantes
# ci-dessus et la section retrieval de l'instance : comportement
# strictement inchangé quand le mode deep n'est pas demandé.
DEEP_DEFAULTS = {
    "bm25_top_k": 80,            # pool brut BM25 par recherche (standard : 40)
    "dense_top_k": 80,           # pool brut dense par recherche (standard : 40)
    "search_top_k": 200,         # sortie de fusion RRF (standard : 100, LIMITS standard)
    "fusion_top_k": 60,          # pool soumis au reranker (standard : 30)
    "rerank_top_n": 30,          # passages conservés pour le LLM (standard : 15)
    "coverage_max_passages": 20, # passages lus par le juge de couverture (standard : 12)
    "coverage_max_queries": 4,   # requêtes ciblées max (standard : 3)
    "quota_par_requete": 6,      # places garanties par requête ciblée (standard : 5)
    "coverage_passage_chars": COVERAGE_PASSAGE_CHARS,
                                 # caractères lus par passage au juge de
                                 # couverture (standard : COVERAGE_PASSAGE_CHARS)
}
# Budget contexte à surveiller côté backend LLM : 20 passages x 1200
# caractères ~ 8k tokens pour le juge, et 30 passages complets pour la
# génération (~15-20k tokens selon le chunker) — le mode deep suppose
# une fenêtre >= 32k (GX10/TensorRT-LLM ; vérifier num_ctx sous Ollama).
# Avec des chunks >= ~1024 tokens (~3800 car.), le juge ne lit plus que
# le tiers de chaque passage à 1200 car. : porter coverage_passage_chars
# à ~2400 en réduisant coverage_max_passages d'autant (16 x 2400 car.
# ~ 11k tokens), et baisser rerank_top_n (~20) pour que les passages
# complets de la génération restent dans la fenêtre.

# Inventaire des titres du corpus dans le prompt de couverture (v2.9).
# Réglable à chaud via settings.yaml (section query2) ; la variable
# d'environnement QUERY2_CATALOGUE_COUVERTURE=false reste un repli.
_CATALOGUE_COUVERTURE_ENV = (
    os.environ.get("QUERY2_CATALOGUE_COUVERTURE", "true").lower() == "true"
)

# Plafond de titres injectés dans le prompt : les 12 passages x 1200
# caractères occupent déjà ~5k tokens sur la fenêtre 16k.
CATALOGUE_PROMPT_MAX_TITRES = 100

# Garde-fou : contradiction entre le juge de couverture (étape 2, lit les
# passages) et le générateur de la route classique (étape 1, déjà exécuté
# dans result_a avant que le juge ne rende son verdict). Mesuré le 22 août
# 2026 sur Beaumont Sud (cas beaumont-09, corridor écologique/TVB) : le
# juge a rendu COUVERT alors que la réponse générée esquivait elle-même
# la question (le passage précis — mesure ECO-E1 — n'était pas dans le
# sous-ensemble retenu pour la génération, bien qu'un autre extrait du
# même tome le soit). Sans ce garde-fou, la recherche B ne se déclenche
# jamais dans ce cas puisqu'elle ne dépend que du verdict du juge.
# Réglable à chaud via settings.yaml (section query2, éditable depuis
# l'UI feedback) ; la variable d'environnement
# QUERY2_GARDE_CONTRADICTION=false reste un repli.
_GARDE_CONTRADICTION_ENV = (
    os.environ.get("QUERY2_GARDE_CONTRADICTION", "true").lower() == "true"
)

# Motif d'esquive sur le TEXTE DE RÉPONSE généré (pas sur les passages).
# Volontairement aligné sur exporter_echanges.verdict() (évaluation) pour
# que le garde-fou système et la mesure utilisent le même critère — mais
# avec un motif de citation élargi : voir _SOURCE_CITEE_REPONSE ci-dessous.
_ESQUIVE_REPONSE = re.compile(
    r"n(?:e |')(?:contien(?:t|nent)|mentionn(?:e|ent)|fourni(?:t|ssent)|"
    r"permet(?:tent)?|cite(?:nt)?|precise(?:nt)?|indique(?:nt)?|"
    r"detaille(?:nt)?|comporte(?:nt)?|evoque(?:nt)?|abord(?:e|ent)) "
    r"(?:pas|aucun)|"
    r"n'en parle(?:nt)? pas|"
    r"(?:n'est|ne sont) pas (?:explicitement )?"
    r"(?:mentionn|precis|indiqu|detaill|abord)|"
    r"aucune information|aucune mention|aucune precision|aucun element|"
    r"pas d'information|pas de mention|pas de precision|"
    r"reste(?:nt)? muet|est absente? d|sont absentes? d", re.I)

# Motif de citation élargi. L'ancien motif (\.pdf|tome[_ ]?\d|source\s*:)
# produisait un faux positif mesuré sur Beaumont Sud (cas beaumont-11) :
# la génération citait "Volet environnement naturel, p. 460" ou "RNT,
# p. 62" — jamais littéralement "Tome 4" ni un nom de fichier .pdf — donc
# une réponse sourcée à 4 reprises était comptée comme non sourcée.
_SOURCE_CITEE_REPONSE = re.compile(
    r"\.pdf|tome[_ ]?\d|\[?source\s*:|"
    r"volet\s+(?:environnement|milieu|paysage)|\bRNT\b|p\.\s*\d+", re.I)


def sans_accents(texte: str) -> str:
    """Ramene le texte a l'ASCII avant comparaison avec un motif ASCII.

    Correctif V4 (defaut n°2) : le motif d'esquive est ecrit sans accents
    mais le texte genere en porte. « ne precise pas », « aucun element »,
    « n'evoque pas » n'etaient jamais reconnus. Le motif reste en ASCII
    (aucun caractere accentue dans le code) ; c'est le texte qui est
    normalise, avec la bibliotheque standard.
    """
    if not texte:
        return ""
    texte = texte.replace("’", "'").replace("‘", "'")
    return "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )


def _reponse_esquive(texte: str) -> bool:
    """Détecte si le texte généré esquive la question (mêmes seuils que
    l'outil d'évaluation : position du motif < 15% du texte, ou
    texte < 700 caractères, ou aucune source citée). Comparaison sur le
    texte désaccentué."""
    if not texte:
        return True
    plat = sans_accents(texte)
    m = _ESQUIVE_REPONSE.search(plat)
    if not m:
        return False
    return (
        m.start() / len(plat) < 0.15
        or len(plat) < 700
        or not _SOURCE_CITEE_REPONSE.search(plat)
    )


SUJET_SYSTEM_PROMPT = (
    "Tu extrais le sujet d'une demande en quelques mots. Tu réponds "
    "uniquement avec le sujet, sans commentaire."
)

SUJET_USER_TEMPLATE = """Quel est le sujet de la demande suivante ?

Règles :
- 3 à 6 mots, avec le déterminant (le, la, les, du, de la, des, l') ;
  davantage si la demande énumère plusieurs objets — garde alors
  l'énumération complète, avec ses virgules
- le sujet désigne les informations recherchées, JAMAIS l'auteur de
  la demande (autorité, client, comité...) ni la demande elle-même
  (« la liste de », « la description de » ne font pas partie du sujet)
- garde la langue de la demande

Exemples :
Demande : Il est recommandé de préciser le calendrier des travaux et les horaires de chantier.
Sujet : le calendrier des travaux et les horaires de chantier

Demande : Le comité demande de compléter le rapport avec une analyse des coûts de maintenance.
Sujet : l'analyse des coûts de maintenance

Demande : Fais la liste des monuments, sites et bâtiments présents autour du projet.
Sujet : les monuments, sites et bâtiments autour du projet

Demande : {query}
Sujet :"""

VOLETS_SYSTEM_PROMPT = (
    "Tu découpes une demande en volets thématiques. Tu réponds "
    "uniquement en JSON valide, sans commentaire ni balise markdown."
)

# Sortie JSON (et non texte libre) : le format qui ne déraille pas sur ce
# modèle multilingual, comme les « requetes » du juge de couverture —
# voir la cascade de replis de _reformuler_en_question.
VOLETS_USER_TEMPLATE = """Demande :
{query}

Découpe cette demande en volets thématiques, c'est-à-dire en sujets
DISTINCTS sur lesquels il faudra chercher des informations séparément.

Règles STRICTES :
- un volet = un seul sujet, exprimé en 3 à 8 mots de contenu
- n'emploie que des noms communs et des termes techniques : jamais de
  verbe d'action de la demande (compléter, préciser, justifier,
  démontrer, actualiser), jamais l'auteur de la demande
- le volet désigne CE QU'IL FAUT TROUVER dans les documents de fond,
  jamais la façon dont la demande veut qu'on le présente : une demande
  de démonstration « par des cartes et des photomontages » porte sur
  l'objet démontré, la carte n'est qu'un support
- {max_volets} volets au maximum, du plus important au moins important
- si la demande ne porte que sur un seul sujet, renvoie UN seul volet
- garde la langue de la demande

Exemples :
Demande : L'autorité recommande de compléter les mesures de réduction proposées en démontrant leur efficacité par la présentation de cartes localisant leur position et des photomontages.
{{"volets": ["les mesures de réduction paysagères", "la localisation des mesures par rapport aux éoliennes"]}}

Demande : L'autorité recommande d'étudier l'évitement en complétant l'étude de variantes, et d'ajuster le cas échéant les conditions du plan d'arrêt des machines.
{{"volets": ["l'étude de variantes d'implantation", "le plan d'arrêt des machines"]}}

Demande : L'autorité recommande de préciser les mesures de bridage prévues pour les chiroptères.
{{"volets": ["les mesures de bridage pour les chiroptères"]}}

JSON :"""

# Consigne renforcée pour la seconde tentative : le modèle multilingual
# décroche parfois vers une autre langue sur les demandes longues
# (mesuré sur Beaumont Sud le 9 août 2026) ; le rappel explicite de la
# langue cible corrige ce glitch stochastique dans la plupart des cas.
SUJET_SYSTEM_PROMPT_RENFORCE = (
    SUJET_SYSTEM_PROMPT
    + " Tu réponds exclusivement en français : aucun caractère chinois,"
      " japonais, coréen, ni aucun symbole spécial."
)

# Gabarit de construction de la question à partir du sujet extrait.
# Mécanique volontairement : la forme de la question ne dépend plus du
# modèle, seul le sujet vient du LLM.
QUESTION_TEMPLATE = "Que disent les documents sur {sujet} ?"

# v2.12 : gabarit des demandes de LISTE. « Que disent les documents sur
# X ? » laisse la génération résumer les passages les plus saillants ;
# « Quels sont {sujet} ? » lui impose de compiler les éléments. Le
# gabarit n'est grammatical qu'avec un déterminant pluriel — d'où la
# garde dans _reformuler_en_question.
QUESTION_TEMPLATE_LISTE = "Quels sont {sujet} ?"

# Demande en forme de liste : « fais la liste de... », « quels sont... »,
# « énumère... ». Le sujet extrait d'une telle demande est souvent une
# énumération à virgules — légitime, voir _sujet_incoherent.
_DEMANDE_LISTE = re.compile(
    r"\blistes?\b|\bquels\b|\bquelles\b|\b[ée]num[ée]r", re.IGNORECASE)

# Lettres attendues dans un sujet en français : alphabet de base et
# accents effectivement utilisés (àâäçéèêëîïôöùûüÿæœ et majuscules).
_LETTRES_FR = re.compile(r"[a-zàâäçéèêëîïôöùûüÿæœ]", re.IGNORECASE)

# Déclencheur de la règle d'inventaire déterministe (v2.10) : la demande
# doit interroger les ENJEUX d'un sujet. Le déclencheur est étroit à
# dessein — un mécanisme générique « titre correspond à la demande »
# échouerait sur les noms de projet présents dans tous les tomes.
_MOTIFS_ENJEUX = re.compile(r"\benjeux?\b", re.IGNORECASE)

# Mots vides exclus du matching de sujet et des termes de titre.
_MOTS_VIDES = frozenset({
    "le", "la", "les", "du", "de", "des", "un", "une", "au", "aux",
    "et", "ou", "en", "dans", "sur", "sont", "est", "quels", "quelles",
    "quel", "quelle", "pe", "d", "l", "etude", "etudes",
})

# Termes structurels des noms de tomes : présents dans tous les fichiers
# du corpus (projet, étude, impact...), ils ne discriminent rien et ne
# doivent jamais déclencher ni composer une requête d'inventaire.
# Liste volontairement générique au domaine étude d'impact : les termes
# propres au projet du corpus courant (nom du parc, de la commune, du
# porteur...) se déclarent par instance via settings.yaml (section
# query2.termes_structurants, éditable à chaud depuis l'UI feedback) ou,
# en repli, la variable QUERY2_TERMES_STRUCTURANTS="fort,paniere" —
# mesuré sur la Panière du Fort : sans eux, les requêtes d'inventaire
# forcé incluent le nom du projet, présent dans tous les titres, et se
# diluent.
_TERMES_STRUCTURANTS_BASE = frozenset({
    "projet", "volet", "tome", "rnt", "impact", "impacts", "environnement",
})

_TERMES_STRUCTURANTS_ENV = frozenset(
    t.strip().lower()
    for t in os.environ.get("QUERY2_TERMES_STRUCTURANTS", "").split(",")
    if t.strip()
)

_TERMES_STRUCTURANTS = _TERMES_STRUCTURANTS_BASE | _TERMES_STRUCTURANTS_ENV


def _normaliser_mot(mot: str) -> str:
    """Minuscules sans accents, apostrophes et tirets normalisés."""
    mot = mot.lower().replace("'", " ").replace("-", " ")
    return "".join(
        c for c in unicodedata.normalize("NFD", mot)
        if unicodedata.category(c) != "Mn"
    )


def _mots_contenu(texte: str) -> set:
    """Mots de contenu normalisés : lettres uniquement, >= 4 caractères,
    racine tronquée (le « s » final est retiré par l'appelant)."""
    mots = set()
    for brut in re.findall(r"[a-zàâäçéèêëîïôöùûüÿæœ]+", _normaliser_mot(texte)):
        if len(brut) >= 4 and brut not in _MOTS_VIDES:
            mots.add(brut)
    return mots


def _racine(mot: str) -> str:
    """Racine de matching : singulier approximatif, jamais < 4 lettres."""
    return mot[:-1] if mot.endswith("s") and len(mot) > 4 else mot


def _sujet_enjeux(query: str) -> Optional[str]:
    """Sujet d'une interrogation sur les enjeux, ou None.

    Le sujet est le texte après le mot « enjeux(x) » : « quels sont les
    enjeux paysagers ? » -> « paysagers ». Sans « enjeux », la demande
    n'est pas du ressort de la règle d'inventaire.
    """
    m = _MOTIFS_ENJEUX.search(query)
    if not m:
        return None
    sujet = query[m.end():].strip().rstrip("?").strip(" .")
    return sujet or None


def _regle_inventaire(
    query: str,
    search_results: List[Dict],
    titres: List[str],
    termes_structurants: Optional[frozenset] = None,
    max_queries: int = COVERAGE_MAX_QUERIES,
) -> Optional[Dict]:
    """Force PARTIEL quand un tome du sujet d'enjeux est absent des passages.

    Déclencheurs cumulatifs, tous requis :
      - la demande interroge les enjeux d'un sujet (_sujet_enjeux) ;
      - un mot de contenu du sujet apparaît dans au moins un titre de
        l'inventaire (racines comparées : « paysagers » matche « paysager ») ;
      - aucun passage de la recherche A ne provient d'un tel document.

    La requête ciblée est assemblée par code : sujet normalisé suivi des
    mots distinctifs du titre (hors termes structurants communs à tous
    les tomes) — jamais par le modèle.
    """
    if termes_structurants is None:
        termes_structurants = _TERMES_STRUCTURANTS
    if not titres:
        return None
    sujet = _sujet_enjeux(query)
    if not sujet:
        return None
    racines_sujet = {_racine(m) for m in _mots_contenu(sujet)}
    if not racines_sujet:
        return None

    tomes_du_sujet = []
    for titre in titres:
        racines_titre = {_racine(m) for m in _mots_contenu(titre)}
        if racines_sujet & racines_titre:
            tomes_du_sujet.append(titre)
    if not tomes_du_sujet:
        return None

    presents = {chunk.get("file_name") for chunk in search_results}
    absents = [t for t in tomes_du_sujet if t not in presents]
    if not absents:
        return None

    # Requête assemblée par code : mots du sujet puis mots distinctifs
    # du titre, dédupliqués par racine (« paysagers » absorbe « paysager »).
    requetes = []
    for titre in absents[:max_queries]:
        termes: List[str] = []
        racines_vues = set()
        for mot in sorted(_mots_contenu(sujet)) + sorted(
            m for m in _mots_contenu(titre) if m not in termes_structurants
        ):
            racine = _racine(mot)
            if racine not in racines_vues:
                racines_vues.add(racine)
                termes.append(mot)
        requete = " ".join(termes).strip()
        if requete:
            requetes.append(requete)
    if not requetes:
        return None

    logger.info(
        f"query2: regle inventaire -> PARTIEL force, "
        f"tomes absents={absents}, requetes={requetes}"
    )
    return {"verdict": "PARTIEL", "manques": absents, "requetes": requetes}


def _sujet_incoherent(sujet: str) -> bool:
    """Détecte une extraction de sujet corrompue ou hors langue.

    Le sujet attendu est un court groupe nominal en français. Le modèle
    multilingual produit parfois du charabia (symboles Latin-1, lettres
    d'autres alphabets) sur les demandes longues — mesuré le 9 août 2026
    sur Beaumont Sud, où le sujet extrait finissait en « ¥å¿ç¼º¤±çæå ».
    Un sujet corrompu garantit une recherche B et une réponse hors sujet.

    Signaux de corruption :
      - lettres hors du répertoire français en proportion notable
        (seuil 15 % : un vrai sujet est à ~100 % de lettres françaises,
        le charabia chute fortement) ;
      - symboles non alphabétiques autres que apostrophe, tiret, espace
        (¥, ¿, ¼, ¤, ±...) ;
      - plus de 12 mots (la consigne en fixe 6).
    """
    if len(sujet.split()) > 12:
        return True
    lettres = sum(1 for c in sujet if c.isalpha())
    if lettres == 0:
        return True
    francaises = len(_LETTRES_FR.findall(sujet))
    if francaises / lettres < 0.85:
        return True
    for c in sujet:
        if c.isalpha():
            # Liste blanche (v2.8) : toute lettre hors du répertoire
            # français signale une sortie corrompue, même en petite
            # proportion — une courte queue CJK (2-4 caractères) passe
            # sous le seuil du ratio 85 % (mesuré sur Beaumont Sud B1,
            # retry du 10 août : « ...éviter réduire补偿Head »).
            if not _LETTRES_FR.match(c):
                return True
        elif c not in " '-,":
            # Virgule admise (v2.12) : les sujets d'énumération en sont
            # pleins — « les monuments, sites et bâtiments ».
            return True
    return False


COVERAGE_SYSTEM_PROMPT = (
    "Tu évalues si des extraits documentaires contiennent les informations "
    "nécessaires pour répondre à une demande. Tu réponds uniquement en JSON "
    "valide, sans commentaire ni balise markdown."
)

COVERAGE_USER_TEMPLATE = """Demande :
{query}

Extraits documentaires trouvés (par ordre de pertinence) :
{passages}

Important : la demande peut prendre la forme d'une recommandation, d'une
instruction ou d'une question portant sur un sujet. Les extraits sont
issus de documents de fond : ils ne contiennent JAMAIS la demande
elle-même. N'évalue donc pas si la demande y est mentionnée, mais si
les extraits contiennent les informations de fond permettant d'y
répondre.

Niveau de détail exigé : la demande n'est couverte que si les extraits
en donnent le CONTENU au niveau demandé, pas s'ils se contentent d'en
parler. Pour une demande de liste ou d'inventaire (sites, monuments,
espèces, enjeux, mesures...), les extraits doivent fournir les éléments
eux-mêmes — noms, localisations, valeurs — et non un renvoi à une
section, une mention générale ou une synthèse qui annonce le sujet
sans le détailler. Un extrait de sommaire, d'introduction ou de
synthèse qui présente le sujet sans le détailler ne compte pas comme
couverture : dans ce cas le verdict est PARTIEL et les requêtes
ciblent le détail manquant.

Ces extraits contiennent-ils ces informations ?

Réponds en JSON avec exactement ces trois clés :
- "verdict" : "COUVERT" si les extraits contiennent l'essentiel des
  informations au niveau de détail demandé, "PARTIEL" s'ils en
  couvrent une partie ou sans le détail exigé, "NON_COUVERT"
  s'ils sont hors sujet ou absents
- "manques" : liste courte des INFORMATIONS manquantes sur le sujet
  (jamais la demande elle-même ; liste vide si COUVERT)
- "requetes" : {max_q} requêtes de recherche maximum, ciblées sur les
  informations manquantes. Règles STRICTES de formulation :
  * 3 à 6 mots de contenu par requête, jamais de phrase
  * une requête = un seul sujet (ne mélange pas plusieurs thèmes)
  * n'emploie que des noms communs et termes techniques, pas de
    verbes d'action ni de mots vides ("impact", "projet", "analyse"
    seuls n'aident pas)
  * les documents cibles emploient souvent un vocabulaire différent
    de la demande : privilégie les synonymes et termes techniques
    français alternatifs (par exemple "variantes" -> "scénarios
    implantation variantes" ; "terres excavées" -> "terres décapées
    stockage"). Ne reprends pas la formulation de la demande
  (liste vide si COUVERT)"""

# Bloc inventaire inséré entre les extraits et la consigne quand le
# catalogue est disponible. La règle d'inventaire neutralise le biais du
# tome de synthèse : une réponse d'apparence complète à partir du RNT ne
# doit plus masquer l'absence du tome spécialisé.
_BLOC_INVENTAIRE = """Inventaire des documents du corpus (titres) :
{titres}

Règle d'inventaire : si le titre d'un document de cet inventaire
correspond clairement au sujet de la demande et qu'AUCUN des extraits
ci-dessus ne provient de ce document, le verdict est PARTIEL — même si
les extraits semblent suffisants — et au moins une requête doit cibler
ce document en reprenant les termes exacts de son titre. Ignore les
documents de l'inventaire sans rapport clair avec la demande."""

# Variante avec catalogue, construite par insertion pour rester
# synchronisée avec COVERAGE_USER_TEMPLATE : le corps de la consigne et
# la spec JSON restent uniques.
COVERAGE_USER_TEMPLATE_CATALOGUE = COVERAGE_USER_TEMPLATE.replace(
    "\nImportant :",
    "\n" + _BLOC_INVENTAIRE + "\n\nImportant :",
)


def _formater_titres(titres: List[str]) -> str:
    """Inventaire à puces pour le prompt, plafonné avec mention du reliquat."""
    if len(titres) > CATALOGUE_PROMPT_MAX_TITRES:
        gardes = titres[:CATALOGUE_PROMPT_MAX_TITRES]
        return "\n".join(f"- {t}" for t in gardes) + (
            f"\n- ... et {len(titres) - len(gardes)} autres documents"
        )
    return "\n".join(f"- {t}" for t in titres)


def _extraire_json(texte: str) -> Optional[Dict]:
    """Extrait le premier objet JSON d'une réponse LLM, tolérant aux
    balises markdown et au texte parasite. Retourne None si échec."""
    if not texte:
        return None
    # Retirer d'éventuelles balises ```json ... ```
    texte = re.sub(r"```(?:json)?", "", texte)
    debut = texte.find("{")
    fin = texte.rfind("}")
    if debut == -1 or fin == -1 or fin <= debut:
        return None
    try:
        return json.loads(texte[debut:fin + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _texte_passage(chunk: Dict) -> str:
    """Texte d'un passage, quel que soit le champ utilisé par le moteur."""
    return (chunk.get("text") or chunk.get("content") or "").strip()


def _etiquette_passage(chunk: Dict) -> str:
    """Etiquette source d'un passage pour le prompt de couverture."""
    nom = chunk.get("file_name") or "document"
    meta = chunk.get("metadata") or {}
    page = meta.get("page") or meta.get("page_start") or ""
    return f"{nom} p.{page}" if page else nom


class IterativePipeline:
    """Pipeline itératif query2.

    Il encapsule l'analyzer pour la recherche et la génération, puis ajoute
    l'analyse de couverture et, si nécessaire, une recherche ciblée.
    """

    def __init__(
        self,
        analyzer,
        catalogue: Optional[CatalogueDocuments] = None,
        query2_config: Optional[Dict] = None,
    ):
        self.analyzer = analyzer
        self._catalogue = catalogue

        # Réglages query2, lus à chaque instanciation (donc à chaque
        # requête /api/query2) : un reload-config à chaud les applique
        # sans redémarrage. Priorité : settings.yaml > variable
        # d'environnement > défaut.
        cfg = query2_config or {}
        self._garde_contradiction_actif = bool(
            cfg.get("garde_contradiction", _GARDE_CONTRADICTION_ENV)
        )
        self._catalogue_couverture_actif = bool(
            cfg.get("catalogue_couverture", _CATALOGUE_COUVERTURE_ENV)
        )
        self._decoupage_volets_actif = bool(
            cfg.get("decoupage_volets", _DECOUPAGE_VOLETS_ENV)
        )
        try:
            self._max_volets = max(1, int(cfg.get("max_volets", MAX_VOLETS)))
        except (TypeError, ValueError):
            logger.warning(
                f"query2.max_volets={cfg.get('max_volets')!r} illisible, "
                f"défaut conservé ({MAX_VOLETS})"
            )
            self._max_volets = MAX_VOLETS
        termes_yaml = cfg.get("termes_structurants") or []
        if isinstance(termes_yaml, str):
            termes_yaml = [t.strip() for t in termes_yaml.split(",")]
        self._termes_structurants = _TERMES_STRUCTURANTS | frozenset(
            str(t).strip().lower() for t in termes_yaml if str(t).strip()
        )

        # Profil deep : défauts du code + surcharges settings.yaml
        # (clés inconnues ignorées, valeurs forcées en int — une chaîne
        # YAML mal quotée ne doit pas casser la requête).
        self._deep = dict(DEEP_DEFAULTS)
        deep_yaml = cfg.get("deep") or {}
        if isinstance(deep_yaml, dict):
            for cle, valeur in deep_yaml.items():
                if cle not in DEEP_DEFAULTS:
                    logger.warning(f"query2.deep: clé inconnue ignorée: {cle}")
                    continue
                try:
                    self._deep[cle] = int(valeur)
                except (TypeError, ValueError):
                    logger.warning(
                        f"query2.deep.{cle}={valeur!r} illisible, "
                        f"défaut conservé ({DEEP_DEFAULTS[cle]})"
                    )

    def _parametres(self, deep: bool) -> Dict:
        """Paramètres effectifs de l'entonnoir pour cette exécution.

        deep=False : réglages courants de l'instance (constantes du
        module + section retrieval de settings.yaml, lus via l'analyzer).
        deep=True : profil élargi (query2.deep).
        """
        if deep:
            return dict(self._deep)
        # Lecture tolérante (getattr) : un analyzer factice de test ne
        # porte pas forcément LIMITS — les défauts reproduisent alors le
        # comportement historique (100/30/15).
        limits = getattr(self.analyzer, "LIMITS", None) or {}
        standard = limits.get("standard", {}) if isinstance(limits, dict) else {}
        return {
            "bm25_top_k": None,   # None = réglage retrieval de l'instance
            "dense_top_k": None,
            "search_top_k": standard.get("max_total_chunks", 100),
            "fusion_top_k": getattr(self.analyzer, "fusion_top_k", 30),
            "rerank_top_n": getattr(self.analyzer, "rerank_top_n", 15),
            "coverage_max_passages": COVERAGE_MAX_PASSAGES,
            "coverage_max_queries": COVERAGE_MAX_QUERIES,
            "quota_par_requete": QUOTA_PAR_REQUETE,
            "coverage_passage_chars": COVERAGE_PASSAGE_CHARS,
        }

    def _catalogue_titres(self) -> List[str]:
        """Titres du corpus pour le prompt de couverture (v2.9).

        Construction paresseuse depuis le BM25 de l'analyzer ; toute
        indisponibilité se traduit par une liste vide, ce qui laisse le
        prompt de couverture identique à la version sans catalogue.
        """
        if not self._catalogue_couverture_actif:
            return []
        if self._catalogue is None:
            bm25 = getattr(self.analyzer.hybrid_search, "bm25_search", None)
            self._catalogue = CatalogueDocuments(bm25)
        return self._catalogue.titres()

    # ------------------------------------------------------------------
    # Étape 2 — analyse de couverture
    # ------------------------------------------------------------------
    def _analyse_couverture(
        self, query: str, search_results: List[Dict],
        params: Optional[Dict] = None,
    ) -> Dict:
        """1 appel LLM sur les passages COMPLETS. Verdict structuré.

        En cas d'échec de parsing ou d'appel : repli sur COUVERT, c'est-à-dire
        comportement identique à la route classique (jamais pire).

        params : paramètres effectifs de l'exécution (voir _parametres) —
        le mode deep y élargit le nombre de passages lus et de requêtes
        ciblées produites."""
        p = params or self._parametres(deep=False)
        max_passages = p["coverage_max_passages"]
        max_queries = p["coverage_max_queries"]
        passage_chars = p["coverage_passage_chars"]
        titres = self._catalogue_titres()
        defaut = {"verdict": "COUVERT", "manques": [], "requetes": [],
                  "catalogue_titres": len(titres)}

        passages = []
        for chunk in search_results[:max_passages]:
            texte = _texte_passage(chunk)[:passage_chars]
            if texte:
                passages.append(f"[{_etiquette_passage(chunk)}]\n{texte}")

        bloc_passages = "\n\n---\n\n".join(passages) if passages else "Aucun extrait trouvé."

        if titres:
            prompt = COVERAGE_USER_TEMPLATE_CATALOGUE.format(
                query=query,
                passages=bloc_passages,
                max_q=max_queries,
                titres=_formater_titres(titres),
            )
        else:
            # Repli strict : prompt identique à la version sans catalogue.
            prompt = COVERAGE_USER_TEMPLATE.format(
                query=query,
                passages=bloc_passages,
                max_q=max_queries,
            )

        try:
            brut = self.analyzer.llm_generator.call_llm(
                COVERAGE_SYSTEM_PROMPT, prompt
            )
        except Exception as e:
            logger.warning(f"query2: echec appel couverture ({e}), repli COUVERT")
            return defaut

        data = _extraire_json(brut)
        if not data:
            logger.warning("query2: JSON couverture illisible, repli COUVERT")
            return defaut

        verdict = str(data.get("verdict", "COUVERT")).upper().strip()
        if verdict not in ("COUVERT", "PARTIEL", "NON_COUVERT"):
            verdict = "COUVERT"

        manques = data.get("manques") or []
        if not isinstance(manques, list):
            manques = [str(manques)]

        requetes = data.get("requetes") or []
        if not isinstance(requetes, list):
            requetes = [str(requetes)]
        # Garder des requêtes non vides, plafonnées
        requetes = [r.strip() for r in requetes if isinstance(r, str) and r.strip()]
        requetes = requetes[:max_queries]

        # Règle d'inventaire déterministe (v2.10) : prioritaire sur le
        # verdict LLM, dont l'adhérence à la consigne d'inventaire n'est
        # pas garantie. Ne voit que les tomes présents dans l'index.
        forcee = _regle_inventaire(
            query, search_results, titres,
            termes_structurants=self._termes_structurants,
            max_queries=max_queries,
        )
        if forcee is not None and verdict == "COUVERT":
            return {**forcee, "catalogue_titres": len(titres)}

        if verdict == "COUVERT":
            manques, requetes = [], []

        return {"verdict": verdict, "manques": manques, "requetes": requetes,
                "catalogue_titres": len(titres)}

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        top_k: int = 20,
        custom_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        max_rounds: int = 1,
        deep: bool = False,
    ) -> Dict:
        """Exécute le pipeline. Retourne le résultat analyze() de l'étape
        finale, augmenté d'une clé 'iterative' tracant la couverture.

        deep=True active le profil « recherche approfondie » : entonnoir
        de retrieval élargi (query2.deep dans settings.yaml) appliqué sur
        les deux recherches et sur l'analyse de couverture. La structure
        fixe en 4 étapes est INCHANGÉE — seule l'amplitude des recherches
        varie, jamais la marche du pipeline.
        """

        params = self._parametres(deep)

        options = {"max_items": top_k}
        if custom_prompt:
            options["custom_prompt"] = custom_prompt
        if deep:
            options["retrieval_overrides"] = {
                cle: params[cle]
                for cle in ("search_top_k", "fusion_top_k",
                            "rerank_top_n", "bm25_top_k", "dense_top_k")
            }
            logger.info(
                f"query2: mode deep actif "
                f"(fusion={params['fusion_top_k']}, "
                f"top_n={params['rerank_top_n']}, "
                f"pools={params['bm25_top_k']}/{params['dense_top_k']})"
            )

        trace = {
            "version": "pipeline_iteratif_v2",
            "mode": "deep" if deep else "standard",
            "recherche_a": None,
            "couverture": None,
            "recherche_b": {"effectuee": False, "requetes": []},
        }

        # ---------------- Étape 0 : découpage en volets -------------------
        # Une demande à plusieurs volets, ou dont le vocabulaire dominant
        # décrit la méthode attendue plutôt que son objet, fait dérailler
        # la requête unique (voir MAX_VOLETS). Chaque volet obtient alors
        # sa propre recherche et des places garanties.
        volets = self._decomposer_en_volets(query)
        trace["volets"] = volets
        if len(volets) > 1:
            logger.info(f"query2: demande decoupee en {len(volets)} volets: {volets}")

        # ---------------- Étape 1 : recherche A + génération classique ----
        if len(volets) > 1:
            # Quota calculé pour laisser sa part à la voie générale :
            # avec 15 passages et 3 volets, 3 places par volet et 6
            # places pour la recherche sur la demande entière.
            quota_volet = max(2, params["rerank_top_n"] // (len(volets) + 1))
            result_a = self._recherche_b_quota(
                query, query, volets, options, history, trace,
                params=params, cle_trace="recherche_a", quota=quota_volet,
            )
            search_a = result_a.get("search_results", [])
        else:
            # Demande à un seul volet : comportement strictement
            # identique aux versions precedentes.
            result_a = self.analyzer.analyze(
                query=query,
                mode="chat",
                options=options,
                history=history,
            )
            search_a = result_a.get("search_results", [])
            trace["recherche_a"] = {"passages": len(search_a)}
        logger.info(f"query2: recherche A -> {len(search_a)} passages")

        # ---------------- Étape 2 : analyse de couverture -----------------
        couverture = self._analyse_couverture(query, search_a, params)
        trace["couverture"] = couverture
        logger.info(
            f"query2: couverture={couverture['verdict']}, "
            f"requetes={couverture['requetes']}"
        )

        # ---------------- Garde-fou : contradiction juge/generateur ------
        # result_a["response"] est deja la reponse de la route classique
        # (generee en etape 1, avant meme l'analyse de couverture). Si le
        # juge dit COUVERT mais que cette reponse esquive elle-meme la
        # question, on force un verdict PARTIEL pour declencher la
        # recherche B malgre tout. A defaut de "manques" fournis par le
        # juge (verdict COUVERT => requetes vide par construction), on
        # utilise la demande d'origine comme seule requete ciblee.
        if (
            self._garde_contradiction_actif
            and couverture["verdict"] == "COUVERT"
            and _reponse_esquive(result_a.get("response", ""))
        ):
            logger.warning(
                "query2: contradiction juge/generateur detectee (verdict "
                "COUVERT mais reponse esquive) -> recherche B forcee"
            )
            couverture = dict(couverture)
            couverture["verdict"] = "PARTIEL"
            couverture["contradiction_forcee"] = True
            if not couverture.get("requetes"):
                couverture["requetes"] = [query]
            trace["couverture"] = couverture

        if couverture["verdict"] == "COUVERT" or not couverture["requetes"] or max_rounds < 1:
            # Propriété de sécurité : la réponse retournée EST la réponse
            # de la route classique (mêmes passages, même génération).
            result_a["iterative"] = trace
            return result_a

        # ---------------- Étape 3 : recherche B à quota réservé ---------
        # Reformulation générique de la demande en question directe,
        # utilisée pour la voie générale et la génération. L'étape 1
        # reste sur la demande d'origine : le repli COUVERT retourne
        # toujours la réponse classique à l'identique.
        question = self._reformuler_en_question(query, couverture["requetes"])
        trace["reformulation"] = {"originale": query, "question": question}
        if question != query:
            logger.info(f"query2: demande transformee en question directe")

        result_b = self._recherche_b_quota(
            query, question, couverture["requetes"], options, history, trace,
            params=params,
        )
        result_b["iterative"] = trace
        return result_b

    # ------------------------------------------------------------------
    # Étape 3a — extraction du sujet, puis construction de la question
    # ------------------------------------------------------------------
    def _reformuler_en_question(
        self, query: str, requetes: Optional[List[str]] = None
    ) -> str:
        """Construit une question directe à partir du sujet de la demande.

        Deux temps (option C) :
          1. le LLM extrait le SUJET de la demande en 3-6 mots — tâche
             simple et bornée, qui ne laisse pas l'auteur de la demande
             (autorité, client...) devenir le sujet de la question ;
          2. le CODE assemble la question via QUESTION_TEMPLATE — la
             forme interrogative ne dépend plus du modèle.

        Une demande déjà interrogative (contient « ? ») est conservée
        telle quelle, sans appel LLM.

        Cascade de replis (v2.7, v2.8) :
          1. extraction simple ;
          2. retry avec consigne de langue renforcée (v2.7) ;
          3. repli sur la première requête de couverture propre (v2.8) :
             les requêtes ciblées sont le même objet qu'un sujet (3-6
             mots de contenu) mais produites en JSON — un format qui ne
             déraille pas, contrairement au texte libre (mesuré sur
             Beaumont Sud B1 : déraillement DÉTERMINISTE à température 0
             sur « éviter réduire compenser », le retry n'y change rien) ;
          4. repli final sur la demande d'origine : jamais pire.
        """
        if "?" in query:
            return query

        sujet = self._extraire_sujet(query)
        if sujet is not None and _sujet_incoherent(sujet):
            logger.warning(
                f"query2: sujet incoherent ({sujet[:60]!r}), retry renforce"
            )
            sujet = None
        if sujet is None:
            sujet = self._extraire_sujet(query, renforce=True)
            if sujet is not None and _sujet_incoherent(sujet):
                logger.warning(
                    f"query2: sujet toujours incoherent au retry ({sujet[:60]!r})"
                )
                sujet = None
        if sujet is None and requetes:
            for req in requetes:
                req = req.strip().rstrip(".")
                if req and len(req) <= 120 and not _sujet_incoherent(req):
                    sujet = req
                    logger.info(
                        f"query2: repli sur requete couverture: '{req[:80]}'"
                    )
                    break
        if sujet is None:
            logger.warning("query2: extraction impossible, demande conservee")
            return query

        # v2.12 : une demande de liste garde son intention d'inventaire.
        # Garde du déterminant pluriel : « Quels sont le calendrier... »
        # serait incorrect, le gabarit standard reste le repli.
        template = QUESTION_TEMPLATE
        if _DEMANDE_LISTE.search(query) and sujet[:4].lower() in ("les ", "des "):
            template = QUESTION_TEMPLATE_LISTE
        question = template.format(sujet=sujet)
        logger.info(f"query2: sujet='{sujet[:80]}' -> question='{question[:120]}'")
        return question

    # ------------------------------------------------------------------
    # Étape 0 — découpage de la demande en volets thématiques
    # ------------------------------------------------------------------
    def _decomposer_en_volets(self, query: str) -> List[str]:
        """Découpe la demande en 1..max_volets sujets de recherche.

        Un seul appel LLM, sortie JSON (format qui ne déraille pas sur ce
        modèle, contrairement au texte libre — voir VOLETS_USER_TEMPLATE).
        Chaque volet passe le garde-fou linguistique _sujet_incoherent,
        déjà utilisé pour l'extraction de sujet.

        Renvoie [] dès que le découpage est inexploitable (appel en
        échec, JSON illisible, volets incohérents) : l'appelant retombe
        alors sur la recherche A d'origine, jamais pire qu'avant.
        """
        if not self._decoupage_volets_actif:
            return []
        try:
            brut = self.analyzer.llm_generator.call_llm(
                VOLETS_SYSTEM_PROMPT,
                VOLETS_USER_TEMPLATE.format(
                    query=query, max_volets=self._max_volets
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"query2: echec decoupage en volets ({e})")
            return []

        brut = re.sub(r"```(?:json)?", "", brut or "")
        debut, fin = brut.find("{"), brut.rfind("}")
        if debut == -1 or fin <= debut:
            logger.warning("query2: JSON volets illisible, decoupage ignore")
            return []
        try:
            data = json.loads(brut[debut:fin + 1])
        except ValueError:
            logger.warning("query2: JSON volets invalide, decoupage ignore")
            return []

        volets: List[str] = []
        for brut_volet in (data.get("volets") or [])[: self._max_volets]:
            volet = str(brut_volet).strip().strip('"').rstrip(".").strip()
            if not volet or len(volet) < 3 or len(volet) > 120:
                continue
            if _sujet_incoherent(volet):
                logger.warning(f"query2: volet incoherent ignore ({volet[:60]!r})")
                continue
            if volet.lower() not in (v.lower() for v in volets):
                volets.append(volet)
        return volets

    def _extraire_sujet(self, query: str, renforce: bool = False) -> Optional[str]:
        """Un appel LLM d'extraction de sujet + nettoyage.

        Retourne le sujet nettoyé, ou None si l'appel a échoué ou si la
        sortie est structurellement invalide (vide, trop courte, trop
        longue). La cohérence linguistique (_sujet_incoherent) est
        jugée par l'appelant, qui décide du retry.
        """
        system = SUJET_SYSTEM_PROMPT_RENFORCE if renforce else SUJET_SYSTEM_PROMPT
        try:
            brut = self.analyzer.llm_generator.call_llm(
                system,
                SUJET_USER_TEMPLATE.format(query=query),
            )
        except Exception as e:
            logger.warning(f"query2: echec extraction sujet ({e})")
            return None

        sujet = (brut or "").strip().strip('"').strip()
        # Une seule ligne, sans préfixe parasite éventuel ni point final
        sujet = sujet.split("\n")[0].strip().rstrip(".").strip()
        sujet = re.sub(r"^sujet\s*[:\-]\s*", "", sujet, flags=re.IGNORECASE)
        if not sujet or len(sujet) < 3 or len(sujet) > 120:
            logger.warning(
                f"query2: sujet extrait invalide ({len(sujet)} car)"
            )
            return None
        return sujet

    # ------------------------------------------------------------------
    # Étape 3 — recherche B à quota réservé
    # ------------------------------------------------------------------
    def _recherche_b_quota(
        self,
        query: str,
        question: str,
        requetes_ciblees: List[str],
        options: Dict,
        history: Optional[List[Dict]],
        trace: Dict,
        params: Optional[Dict] = None,
        cle_trace: str = "recherche_b",
        quota: Optional[int] = None,
    ) -> Dict:
        """Recherche à places garanties pour les requêtes ciblées.

        Chaque requête ciblée effectue sa propre recherche hybride suivie
        d'un rerank CONTRE ELLE-MÊME (et non contre la demande d'origine),
        puis réserve quota_par_requete passages dans le top final. Le
        reste du top est rempli par la recherche générale sur la QUESTION
        REFORMULÉE, rerankée normalement — identique à la route classique.

        La génération réutilise les méthodes exactes de l'analyzer
        (_build_context + llm_generator.generate) : seule la sélection
        des passages change, jamais le prompt ni les paramètres LLM.
        La génération reçoit la question reformulée comme « Question : ».

        Args:
            query: demande d'origine (conservée pour la trace)
            question: forme question de la demande (recherche générale
                et génération)
            params: paramètres effectifs de l'exécution (voir _parametres)
                — profil standard ou deep élargi.
        """
        analyzer = self.analyzer
        hs = analyzer.hybrid_search
        rr = analyzer.reranker
        p = params or self._parametres(deep=False)
        fusion_k = p["fusion_top_k"]              # pool avant rerank
        top_n = p["rerank_top_n"]                 # passages au LLM
        # places par requête ciblée (surchargeable : la recherche A par
        # volets réserve moins, pour laisser sa part à la voie générale)
        quota = quota or p["quota_par_requete"]
        custom_prompt = options.get("custom_prompt")

        def _cid(chunk: Dict):
            return (
                chunk.get("chunk_id")
                or chunk.get("id")
                or (chunk.get("file_name"), _texte_passage(chunk)[:100])
            )

        def _cherche_et_rerank(q: str) -> List[Dict]:
            """Recherche hybride + rerank contre q, comme la route
            classique mais avec q comme unique requête."""
            resultats = hs.search(
                q,
                top_k=p["search_top_k"],
                bm25_top_k=p["bm25_top_k"],
                dense_top_k=p["dense_top_k"],
            )
            if rr and resultats:
                # Pas de troncature à top_n ici : le quota protégé prélève
                # ensuite dans le classement COMPLET (quota têtes de liste
                # par requête), puis le top final est assemblé à part.
                resultats = rr.rerank(q, resultats[:fusion_k], top_n=fusion_k)
            return resultats

        # ---- Voie générale : la question reformulée --------------------
        # (mesuré : la demande administrative brute est une mauvaise
        # requête de recherche — la forme question cherche mieux)
        generaux = _cherche_et_rerank(question)

        # ---- Voies protégées : une recherche par requête ciblée --------
        proteges: List[Dict] = []
        vus = set()
        detail_quota = []
        for req in requetes_ciblees:
            try:
                classement = _cherche_et_rerank(req)
            except Exception as e:
                logger.warning(f"query2: echec recherche ciblee '{req[:50]}' ({e})")
                continue
            pris = 0
            for chunk in classement:
                cid = _cid(chunk)
                if cid in vus:
                    continue
                vus.add(cid)
                chunk = dict(chunk)
                chunk["quota_requete"] = req
                proteges.append(chunk)
                detail_quota.append({
                    "requete": req,
                    "source": _etiquette_passage(chunk),
                    "rang": pris + 1,
                })
                pris += 1
                if pris >= quota:
                    break

        # ---- Assemblage : protégés d'abord, puis généraux --------------
        final: List[Dict] = list(proteges)
        for chunk in generaux:
            if len(final) >= top_n:
                break
            cid = _cid(chunk)
            if cid in vus:
                continue
            vus.add(cid)
            final.append(chunk)
        final = final[:top_n]

        trace[cle_trace] = {
            "effectuee": True,
            "mode": "quota_reserve",
            "requetes": requetes_ciblees,
            "proteges": detail_quota,
            "passages_finaux": len(final),
            "parametres": {
                "fusion_top_k": fusion_k,
                "rerank_top_n": top_n,
                "quota_par_requete": quota,
            },
        }
        logger.info(
            f"query2: {cle_trace} quota -> {len(proteges)} proteges, "
            f"{len(final)} passages au LLM"
        )

        # ---- Génération : appels identiques à la route classique -------
        # Le LLM reçoit la question reformulée après « Question : » —
        # jamais la recommandation administrative brute.
        context = analyzer._build_context(final, top_n=top_n)
        llm_result = analyzer.llm_generator.generate(
            question,
            context,
            final,
            custom_prompt=custom_prompt,
            history=history,
        )

        return {
            "result_type": "chat",
            "response": llm_result.get("response", ""),
            "sources": llm_result.get("sources", []),
            "search_results": final,
            "metadata": {
                "confidence": llm_result.get("confidence", 0),
                "model": llm_result.get("model", "unknown"),
                "custom_prompt_used": custom_prompt is not None,
                "history_used": history is not None and len(history) > 0,
            },
        }
