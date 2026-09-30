"""Pipeline de réponse en trois étapes (endpoint /api/rag/query) — v3.6 (généraliste).

  1. ANALYSE + QUESTIONS
       a. un appel LLM court extrait de la demande une structure :
          {"objet", "type_objet", "acte", "aspects", "motifs", "demontrer"}
          (prompts.yaml : `analyse_system_prompt`, `analyse_prompt`) ;
          type_objet ∈ {comparaison, etude, mesure, element, forme} décrit
          la nature de la partie du dossier visée, acte ce que la demande
          en attend (compléter, justifier, garantir...).
       b. le code compose les questions à partir de gabarits fixes
          (prompts.yaml : `gabarits_questions.objet_<type>`, `.aspect`,
          `.motif`) — une ou deux questions sur l'objet, choisies selon
          son type, puis une par aspect, puis une par motif.
          Le LLM n'écrit plus de question : ses sorties sont des champs
          courts, la variabilité du décodage ne se propage plus.
       Repli : si l'analyse échoue ou si les gabarits manquent, l'ancien
       chemin (`questions_prompt`, questions rédigées par le LLM) est
       utilisé ; s'il échoue aussi, la demande passe telle quelle.
  2. RÉPONSES : pour chaque question, Retriever.analyze() (hybride +
       rerank + génération). La demande d'origine est transmise à la
       génération comme cadrage (prompts.yaml : `cadrage_prompt`, concaténé
       sous le system prompt avec l'éventuel prompt personnalisé).
       Gardes déterministes par bloc, dans l'ordre :
         - esquive      (regex `_reponse_esquive`)  -> passe élargie `query2.deep`
         - tronqué      (génération arrêtée par max_tokens)
         - dégénéré     (lignes répétées en boucle)   -> relance sous
                                                        `query2.relance` (repeat_penalty)
         - hors lot     (pages citées absentes des étiquettes du lot) -> signalé
  3. SYNTHÈSE + ASSEMBLAGE : quand la demande a été analysée (remarque,
       instruction) et qu'il y a plusieurs blocs, un appel LLM rédige la
       réponse à partir de la demande, de l'analyse et des blocs cités
       (prompts.yaml : `synthese_system_prompt`, `synthese_prompt`) ; mêmes
       gardes (tronqué / dégénéré -> relance ; pages hors lot -> signalé) ;
       garde de rétention (v3.6) : les nombres et noms propres des blocs
       doivent se retrouver dans la synthèse (`retention_min`), sinon
       relance avec la liste des lignes omises, puis lignes jointes
       telles quelles ; si la synthèse échoue, la concaténation des blocs
       sert de repli. Les
       blocs restent joints en annexe (`annexe_blocs`). Sources et passages
       fusionnés, dédupliqués ; trace `iterative` complète (questions,
       origine, gardes, passages par question, synthèse, verdict
       COUVERT / PARTIEL / NON_COUVERT).

Aucun prompt, aucun lexique et aucun seuil métier dans ce module : tout
vient de prompts.yaml et de settings.yaml. Le module ne connaît du
Retriever que `analyze()` et `llm_generator` (`call_llm()`, et s'ils
existent `last_usage` et `surcharge()`).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from contextlib import nullcontext
from typing import Dict, List, Optional, Set

from loguru import logger

from src.config.prompts import load_prompts


# --------------------------------------------------------------------------
# Détection d'esquive (partagée avec l'évaluation via service.py)
# --------------------------------------------------------------------------

# Une réponse « esquive » quand elle affirme l'absence d'information tôt
# dans le texte, ou (si un motif d'absence est présent) qu'elle est très
# courte ou ne cite aucune source. Motifs en ASCII : le texte comparé est
# désaccentué.
_ESQUIVE = re.compile(
    r"n(?:e |')(?:contien|mentionn|fourni|permet|cite|precise|indique|"
    r"detaille|comporte|evoque|abord)\w* (?:pas|aucun)|"
    r"aucune? (?:information|mention|precision|element)|"
    r"pas d(?:e |')(?:information|mention|precision)|"
    r"reste(?:nt)? muet|(?:est|sont) absente?s? d", re.I)

# Une citation = un numéro de page ou une étiquette de source.
_CITATION = re.compile(r"\bp(?:age)?s?\.?\s*\d+|source\s*:|\.pdf", re.I)

# Page(s) citée(s) dans une réponse : « page 73 », « pages 104-106 », « p. 60 ».
_PAGE_CITEE = re.compile(
    r"\b(?:pages?|p\.)\s*(\d{1,4})(?:\s*(?:-|–|à)\s*(\d{1,4}))?", re.I)
_NOM_PDF = re.compile(r"([\w\-()]+(?:[ .][\w\-()]+)*\.pdf)", re.I)
_NUMEROTATION = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def sans_accents(texte: str) -> str:
    texte = (texte or "").replace("’", "'").replace("‘", "'")
    return "".join(c for c in unicodedata.normalize("NFD", texte)
                   if unicodedata.category(c) != "Mn")


def _reponse_esquive(texte: str, position_max: float = 0.15,
                     longueur_min: int = 700) -> bool:
    """Vrai si la réponse esquive la question (mêmes seuils que l'outil
    d'évaluation). NB : la longueur et la citation ne sont testées que si
    un motif d'absence a été trouvé."""
    if not texte:
        return True
    plat = sans_accents(texte)
    m = _ESQUIVE.search(plat)
    if not m:
        return False
    return (m.start() / len(plat) < position_max
            or len(plat) < longueur_min
            or not _CITATION.search(plat))


def _reponse_degeneree(texte: str, min_lignes: int = 8,
                       ratio_min: float = 0.6, repetitions_max: int = 4) -> bool:
    """Vrai si la réponse boucle : trop peu de lignes distinctes, ou une
    même ligne répétée `repetitions_max` fois (numérotation ignorée)."""
    lignes = []
    for brute in (texte or "").splitlines():
        l = _NUMEROTATION.sub("", brute)
        l = re.sub(r"\W+", " ", sans_accents(l)).strip().lower()
        if len(l) > 20:
            lignes.append(l)
    if len(lignes) < min_lignes:
        return False
    compte = Counter(lignes)
    return (len(compte) / len(lignes) < ratio_min
            or compte.most_common(1)[0][1] >= repetitions_max)


def _pages_du_lot(r: Dict) -> Dict[str, Set[int]]:
    """Pages couvertes par les étiquettes du lot, par nom de fichier."""
    pages: Dict[str, Set[int]] = {}
    candidats = list(r.get("sources") or [])
    for p in r.get("search_results") or []:
        meta = p.get("metadata") or {}
        candidats.append({"file_name": p.get("file_name") or meta.get("file_name"),
                          "page_start": p.get("page_start", meta.get("page_start")),
                          "page_end": p.get("page_end", meta.get("page_end"))})
    for s in candidats:
        nom = (s.get("file_name") or "").strip().lower()
        debut, fin = s.get("page_start"), s.get("page_end")
        if not nom or debut is None:
            continue
        try:
            debut, fin = int(debut), int(fin if fin is not None else debut)
        except (TypeError, ValueError):
            continue
        pages.setdefault(nom, set()).update(range(min(debut, fin), max(debut, fin) + 1))
    return pages


def _pages_hors_lot(texte: str, pages_lot: Dict[str, Set[int]],
                    fenetre: int = 300) -> List[str]:
    """Pages citées dans la réponse qui n'appartiennent à aucune étiquette.

    Le document d'une citation est le dernier « *.pdf » nommé dans les
    `fenetre` caractères précédents ; à défaut, la page est comparée à
    l'union des pages du lot. Retourne des libellés « fichier p. N »."""
    if not texte or not pages_lot:
        return []
    union = set().union(*pages_lot.values())
    hors, vus = [], set()
    for m in _PAGE_CITEE.finditer(texte):
        debut = int(m.group(1))
        fin = int(m.group(2)) if m.group(2) else debut
        if fin < debut or fin - debut > 20:
            fin = debut
        avant = texte[max(0, m.start() - fenetre):m.start()].lower()
        # Document de la citation : le nom de fichier du lot cité en dernier
        # dans la fenêtre ; à défaut un « *.pdf » inconnu ; à défaut l'union.
        # Document de la citation : la dernière référence avant la page, qu'elle
        # soit un nom de fichier (complet ou abrégé par le modèle) ou un
        # « Tome N » ; la plus proche de la page l'emporte.
        nom, position = None, -1
        for cite in _NOM_PDF.finditer(avant):
            mots = cite.group(1).strip().split(" ")
            candidat = mots[-1]
            for i in range(len(mots)):
                token = " ".join(mots[i:])
                correspondants = [c for c in pages_lot if c == token or c.endswith(token)]
                if correspondants:
                    candidat = min(correspondants, key=len)
                    break
            nom, position = candidat, cite.end()
        for tome in re.finditer(r"\btome\s*(\d+)", avant):
            if tome.end() > position:
                cles = [c for c in pages_lot if f"tome_{tome.group(1)}" in c or f"tome {tome.group(1)}" in c]
                if len(cles) == 1:
                    nom, position = cles[0], tome.end()
        reference = pages_lot.get(nom) if nom in pages_lot else union
        if fin != debut and debut in reference and fin in reference:
            continue  # plage dont les deux bornes sont des étiquettes du lot : citations fusionnées
        for page in range(debut, fin + 1):
            if page not in reference:
                libelle = f"{nom} p. {page}" if nom else f"p. {page}"
                if libelle not in vus:
                    vus.add(libelle)
                    hors.append(libelle)
    return hors


# --------------------------------------------------------------------------
# Garde sur les nombres : chaque valeur chiffrée de la réponse doit exister
# dans le texte des extraits du lot. C'est la garde qui rend sûr d'utiliser
# le savoir du modèle en amont (choix des questions) : en aval, un nombre
# qu'aucun extrait ne porte est signalé.
# --------------------------------------------------------------------------

_NOMBRE = re.compile(r"(?<![\w/])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?(?![\w/])")
_CITATION_ZONE = re.compile(r"\[Source:[^\]]*\]|\((?:Source|Tome)[^)]*\)|\b(?:pages?|p\.)\s*\d+(?:\s*(?:-|–|à)\s*\d+)?", re.I)
_NUMEROTATION_LIGNE = re.compile(r"^\s*\d+[.)]\s", re.M)


def _nombres_de(texte: str) -> Set[str]:
    """Nombres normalisés d'un texte : espaces de milliers retirés, virgule
    décimale -> point, zéros de fin des décimales retirés (« 2 830 » -> 2830,
    « 0,80 » -> 0.8)."""
    resultat: Set[str] = set()
    for m in _NOMBRE.finditer(texte or ""):
        entier = re.sub(r"[   ]", "", m.group(1))
        dec = (m.group(2) or "").rstrip("0")
        resultat.add(f"{entier}.{dec}" if dec else entier)
    return resultat


def _texte_du_lot(r: Dict) -> str:
    morceaux = []
    for p in r.get("search_results") or []:
        meta = p.get("metadata") or {}
        morceaux.append(str(p.get("text") or p.get("text_with_context") or meta.get("text") or ""))
    return "\n".join(morceaux)


def _nombres_hors_extraits(reponse: str, texte_lot: str, min_chiffres: int = 2) -> List[str]:
    """Nombres de la réponse absents des extraits, hors citations de pages,
    numérotation de liste et petits entiers (moins de `min_chiffres` chiffres
    et sans décimale, trop ambigus). Retourne les nombres tels qu'écrits."""
    if not reponse or not texte_lot:
        return []
    nettoye = _CITATION_ZONE.sub(" ", reponse)
    nettoye = _NUMEROTATION_LIGNE.sub(" ", nettoye)
    presents = _nombres_de(texte_lot)
    hors, vus = [], set()
    for m in _NOMBRE.finditer(nettoye):
        entier = re.sub(r"[   ]", "", m.group(1))
        dec = (m.group(2) or "").rstrip("0")
        if not dec and len(entier) < min_chiffres:
            continue
        cle = f"{entier}.{dec}" if dec else entier
        if cle in presents:
            continue
        # « 2 830 m » écrit « 2830 » ou « 2.830 » dans l'extrait, et l'inverse
        if entier in presents or cle.replace(".", "") in presents:
            continue
        brut = m.group(0).strip()
        if brut not in vus:
            vus.add(brut)
            hors.append(brut)
    return hors


# --------------------------------------------------------------------------
# Garde de rétention (synthèse, v3.6) : les données concrètes que les blocs
# portent (nombres, noms propres) doivent survivre à la rédaction. Le LLM
# résume par défaut ; la règle « reprends les éléments concrets » du prompt
# n'est pas appliquée de façon stable, le code mesure donc ce qui a survécu
# et relance une fois avec la liste de ce qui manque.
# --------------------------------------------------------------------------

# Un nom propre : mot à initiale majuscule suivi de minuscules, avec ses
# composants liés (« Brissy-Hamégicourt », « Séry-lès-Mézières »,
# « Moÿ-de-l'Aisne »). Les sigles (RTE, EUROBATS) ne sont pas suivis.
_MOT_PROPRE = re.compile(
    r"(?<![\w'’-])([A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+(?:[-'’][A-Za-zÀ-ÖØ-öø-ÿ]+)*)")
_LIGNE_TITRE = re.compile(r"^\s*#{1,6}\s")
_DEBUT_LIGNE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)]|#{1,6})?\s*$")


def _noms_propres(texte: str, texte_lot: str) -> Set[str]:
    """Noms propres d'un texte, retenus seulement s'ils figurent tels quels
    dans les extraits (ils viennent du dossier, pas d'un titre du LLM) et si
    leur forme en minuscules n'y figure pas (un nom commun en début de
    phrase, « Selon », « Enjeux », existe aussi en minuscules)."""
    if not texte or not texte_lot:
        return set()
    resultat: Set[str] = set()
    plat = _CITATION_ZONE.sub(" ", texte)
    for m in _MOT_PROPRE.finditer(plat):
        mot = m.group(1)
        if len(mot) < 4 or mot not in texte_lot:
            continue
        # En tête de ligne (après puce, gras, numérotation) ou de phrase, la
        # majuscule ne prouve rien : ignoré (un nom perdu vaut mieux qu'un
        # nom commun exigé).
        avant = plat[:m.start(1)].rstrip("*_ \t")
        if not avant or avant.endswith(("\n", ".", "!", "?", ";")) or _DEBUT_LIGNE.search(avant):
            continue
        # Forme en minuscules présente telle quelle dans les extraits : nom commun
        if re.search(r"(?<![\w'’-])" + re.escape(mot.lower()) + r"(?![\w-])", texte_lot):
            continue
        resultat.add(mot)
    return resultat


def _valeurs_de(texte: str, texte_lot: str, min_chiffres: int = 2) -> Set[str]:
    """Données concrètes d'un texte : nombres (clé normalisée, hors citations
    et numérotation, présents dans les extraits) et noms propres."""
    nettoye = _NUMEROTATION_LIGNE.sub(" ", _CITATION_ZONE.sub(" ", texte or ""))
    presents = _nombres_de(texte_lot)
    valeurs: Set[str] = set()
    for m in _NOMBRE.finditer(nettoye):
        entier = re.sub(r"[   ]", "", m.group(1))
        dec = (m.group(2) or "").rstrip("0")
        if not dec and len(entier) < min_chiffres:
            continue
        cle = f"{entier}.{dec}" if dec else entier
        if cle in presents:
            valeurs.add(cle)
    return valeurs | _noms_propres(texte, texte_lot)


def _retention(synthese: str, blocs: str, texte_lot: str):
    """Mesure ce que la synthèse a gardé des données des blocs. Retourne
    (taux, valeurs manquantes, lignes des blocs qui les portent)."""
    attendues = _valeurs_de(blocs, texte_lot)
    if not attendues:
        return 1.0, [], []
    obtenues = _valeurs_de(synthese, texte_lot)
    plat_synthese = _CITATION_ZONE.sub(" ", synthese or "")
    manquantes = []
    for v in sorted(attendues):
        if v in obtenues:
            continue
        # Un nombre écrit autrement (« 2830 » pour « 2 830 », « 0,80 » pour « 0,8 »)
        if v[0].isdigit() and (v.replace(".", ",") in plat_synthese or v in plat_synthese):
            continue
        manquantes.append(v)
    taux = 1.0 - len(manquantes) / len(attendues)
    lignes: List[str] = []
    for ligne in (blocs or "").splitlines():
        if not ligne.strip() or _LIGNE_TITRE.match(ligne):
            continue
        valeurs_ligne = _valeurs_de(ligne, texte_lot)
        if valeurs_ligne & set(manquantes):
            propre = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ligne).replace("**", "").strip()
            if propre and propre not in lignes:
                lignes.append(propre)
    return taux, manquantes, lignes


# --------------------------------------------------------------------------
# Plafond des documents secondaires (v3.6). Les résumés (RNT, synthèse des
# impacts) sont des chunks denses et autoportants que le reranker note
# mieux que les pages de détail ; sans plafond ils occupent le lot et la
# réponse reste au niveau du résumé. À appeler sur la liste classée par le
# reranker, avant de couper à top_n (retriever.py ou reranker.py).
# --------------------------------------------------------------------------

def _nom_fichier(chunk: Dict) -> str:
    meta = chunk.get("metadata") or {}
    return str(chunk.get("file_name") or meta.get("file_name")
               or chunk.get("file_path") or meta.get("file_path") or "")


# Une ligne de sommaire ou de liste de figures : « Carte 12 : … 45 »,
# « 3.2.1 Titre ........ 18 », « Figure 4 – … (page 20) ».
_LIGNE_SOMMAIRE = re.compile(
    r"^\s*(?:(?:carte|figure|tableau|photo|planche|annexe|illustration)\s*n?°?\s*\d+\b"
    r"|\d+(?:\.\d+){1,3}\s+\S)"
    r".*?(?:\.{3,}\s*\d{1,3}|\bp(?:age)?\.?\s*\d{1,3}\)?|\s\d{1,3})\s*$", re.I)


def _est_sommaire(texte: str, min_lignes: int = 5, ratio_min: float = 0.4) -> bool:
    """Vrai si le passage est un sommaire ou une liste de cartes/figures :
    au moins `min_lignes` lignes de sommaire, formant au moins `ratio_min`
    des lignes non vides. Ces passages font citer des pages qui ne sont
    pas dans le lot et ne portent aucune information."""
    lignes = [l for l in (texte or "").splitlines() if l.strip()]
    if len(lignes) < min_lignes:
        return False
    n = sum(1 for l in lignes if _LIGNE_SOMMAIRE.match(l))
    return n >= min_lignes and n / len(lignes) >= ratio_min


def _texte_chunk(chunk: Dict) -> str:
    meta = chunk.get("metadata") or {}
    return str(chunk.get("text") or chunk.get("text_with_context") or meta.get("text") or "")


def plafonner_secondaires(classes: List[Dict], motifs: List[str], plafond: int,
                          top_n: int, exclure_sommaires: bool = True) -> List[Dict]:
    """Retourne les `top_n` premiers de `classes` (déjà classés) en gardant
    au plus `plafond` passages dont le nom de fichier contient un des
    `motifs` (comparaison désaccentuée, insensible à la casse) et, si
    `exclure_sommaires`, sans les passages de sommaire ; les places
    libérées vont aux candidats suivants. `plafond` <= 0 ou `motifs` vide :
    pas de plafond."""
    cles = [sans_accents(m).lower() for m in motifs if m] if plafond > 0 else []
    retenus: List[Dict] = []
    secondaires = 0
    ecartes_plafond = ecartes_sommaire = 0
    for c in classes:
        if len(retenus) >= top_n:
            break
        if exclure_sommaires and _est_sommaire(_texte_chunk(c)):
            ecartes_sommaire += 1
            continue
        nom = sans_accents(_nom_fichier(c)).lower()
        if cles and any(k in nom for k in cles):
            if secondaires >= plafond:
                ecartes_plafond += 1
                continue
            secondaires += 1
        retenus.append(c)
    if ecartes_plafond or ecartes_sommaire:
        logger.info(f"query2: lot filtré : {ecartes_sommaire} sommaire(s) écarté(s), "
                    f"{ecartes_plafond} au-delà du plafond {plafond} des documents "
                    f"secondaires, {len(retenus)} retenus")
    return retenus


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------

def _objet_equilibre(texte: str, debut: int) -> int:
    """Indice de l'accolade fermant l'objet ouvert en `debut` (chaînes et
    échappements respectés), ou -1."""
    prof, en_chaine, echappe = 0, False, False
    for i in range(debut, len(texte)):
        c = texte[i]
        if en_chaine:
            if echappe:
                echappe = False
            elif c == "\\":
                echappe = True
            elif c == '"':
                en_chaine = False
            continue
        if c == '"':
            en_chaine = True
        elif c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
            if prof == 0:
                return i
    return -1


def _extraire_json(texte: str) -> Optional[Dict]:
    """Premier objet JSON d'une réponse LLM, tolérant aux balises markdown,
    au texte qui suit l'objet, aux virgules finales et aux guillemets
    simples (défauts fréquents d'un modèle 14B)."""
    if not texte:
        return None
    texte = re.sub(r"```(?:json)?", "", texte)
    debut = texte.find("{")
    if debut == -1:
        return None
    fin = _objet_equilibre(texte, debut)
    candidats = [texte[debut:fin + 1]] if fin > debut else []
    candidats.append(texte[debut:texte.rfind("}") + 1])
    for brut in candidats:
        if not brut:
            continue
        essais = [brut, re.sub(r",\s*([}\]])", r"\1", brut)]
        if '"' not in brut and "'" in brut:
            essais.append(re.sub(r",\s*([}\]])", r"\1", brut).replace("'", '"'))
        for e in essais:
            try:
                data = json.loads(e)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _formater(nom: str, gabarit: str, **champs) -> str:
    """`str.format` qui échoue bruyamment : un gabarit de prompts.yaml
    contenant une accolade non doublée ou un champ inconnu est une erreur
    de configuration, pas un cas à contourner en silence."""
    try:
        return gabarit.format(**champs)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"prompts.yaml : gabarit `{nom}` invalide ({e!r}). Champs disponibles : "
            f"{sorted(champs)} ; doubler les accolades littérales ({{ }})."
        ) from e


def _question_directe(texte: str) -> bool:
    """Vrai si la demande est une question directe : une seule phrase
    terminée par « ? » (une remarque d'autorité est une phrase déclarative)."""
    t = (texte or "").strip()
    if not t.endswith("?"):
        return False
    corps = t[:-1].strip()
    return not re.search(r"[.!?]\s+[A-ZÀ-Ý]", corps)


TYPES_OBJET = ("comparaison", "etude", "mesure", "element", "forme")


def _normaliser_type(valeur) -> str:
    """Ramène la valeur `type_objet` de l'analyse à l'un de TYPES_OBJET
    (sans accent, minuscules ; préfixe accepté : « étude » -> etude).
    Chaîne vide si la valeur est absente ou inconnue."""
    t = str(valeur or "").strip().strip('"').lower()
    t = (t.replace("é", "e").replace("è", "e").replace("ê", "e"))
    for connu in TYPES_OBJET:
        if t == connu or t.startswith(connu):
            return connu
    return ""


def _cle_passage(chunk: Dict):
    return (chunk.get("chunk_id") or chunk.get("id")
            or (chunk.get("file_name"), (chunk.get("text") or "")[:100]))


def _etiquette(chunk: Dict) -> str:
    meta = chunk.get("metadata") or {}
    nom = chunk.get("file_name") or meta.get("file_name") or "?"
    debut = chunk.get("page_start", meta.get("page_start"))
    fin = chunk.get("page_end", meta.get("page_end"))
    if debut is None:
        return nom
    return f"{nom} p. {debut}" if fin in (None, debut) else f"{nom} p. {debut}-{fin}"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class Pipeline:
    """Analyse -> questions -> réponses -> assemblage.

    `analyzer` : le Retriever (analyze, llm_generator).
    `query2_config` : section `query2` de settings.yaml (le service
    instancie un Pipeline par requête, la section est donc relue à chaque
    requête). Clés reconnues, toutes optionnelles :
      transformation     (bool, défaut true)  étape 1 active
      max_questions      (int,  défaut 4)     questions au maximum
      cadrage            (bool, défaut true)  transmettre la demande à la génération
      repli_esquive      (bool, défaut true)  passe élargie si esquive
      deep               (dict)               surcharges retrieval du profil élargi
                                              (bm25_top_k, dense_top_k, search_top_k,
                                              fusion_top_k, rerank_top_n)
      relance_boucle     (bool, défaut true)  relancer un bloc tronqué/dégénéré
      relance            (dict)               options LLM de la relance
                                              (défaut repeat_penalty 1.1, repeat_last_n 256)
      signaler_hors_lot  (bool, défaut true)  ajouter une note au bloc si pages hors lot
      degenere           (dict)               seuils : min_lignes, ratio_min, repetitions_max
      objet              (dict)               surcharges retrieval des questions sur l'objet
                                              (gabarits objet_<type> ; mêmes clés que deep),
                                              p. ex. rerank_top_n 15 : l'objet couvre un
                                              chapitre entier
      synthese           (bool, défaut true)  étape 3 : rédaction LLM à partir des blocs
                                              (prompts.yaml : synthese_system_prompt, synthese_prompt) ;
                                              la concaténation reste le repli
      annexe_blocs       (bool, défaut true)  joindre les blocs par question sous la synthèse
      signaler_nombres   (bool, défaut true)  ajouter une note si la réponse contient des
                                              nombres absents du texte des extraits
      motifs_generiques  (liste)              motifs ignorés car sans thème (défaut :
                                              environnement, milieux, milieu, impacts,
                                              effets, incidences, enjeux)
      retention_min      (float, défaut 0.6)  part des données des blocs (nombres, noms
                                              propres) que la synthèse doit reprendre
      retention_relance  (bool, défaut true)  sous le seuil, relancer la synthèse une fois
                                              avec la liste des lignes omises
      retention_annexe   (bool, défaut true)  toujours sous le seuil, joindre ces lignes
                                              telles quelles sous la synthèse
      documents_secondaires (liste)           motifs de nom de fichier des documents de
                                              résumé (p. ex. RNT) — voir plafonner_secondaires
      plafond_secondaires (int, défaut 2)     passages au plus issus de ces documents par
                                              lot (0 : désactivé)
    """

    CLES_DEEP = ("bm25_top_k", "dense_top_k", "search_top_k",
                 "fusion_top_k", "rerank_top_n")
    RELANCE_DEFAUT = {"repeat_penalty": 1.1, "repeat_last_n": 256}
    RETENTION_LIGNES_MAX = 25

    def __init__(self, analyzer, query2_config: Optional[Dict] = None,
                 prompts=None):
        self.analyzer = analyzer
        cfg = query2_config or {}
        self._transformation = bool(cfg.get("transformation", True))
        self._max_questions = max(1, int(cfg.get("max_questions", 4) or 4))
        self._cadrage = bool(cfg.get("cadrage", True))
        self._repli_esquive = bool(cfg.get("repli_esquive", True))
        deep = cfg.get("deep") or {}
        self._deep = {k: int(deep[k]) for k in self.CLES_DEEP if k in deep}
        self._relance_boucle = bool(cfg.get("relance_boucle", True))
        self._relance = dict(self.RELANCE_DEFAUT, **(cfg.get("relance") or {}))
        self._signaler_hors_lot = bool(cfg.get("signaler_hors_lot", True))
        self._signaler_nombres = bool(cfg.get("signaler_nombres", True))
        self._degenere = dict(cfg.get("degenere") or {})
        objet = cfg.get("objet") or {}
        self._objet = {k: int(objet[k]) for k in self.CLES_DEEP if k in objet}
        self._questions_objet: Set[str] = set()
        self._synthese = bool(cfg.get("synthese", True))
        self._annexe_blocs = bool(cfg.get("annexe_blocs", True))
        self._motifs_generiques = [str(x) for x in (cfg.get("motifs_generiques") or [])]
        self._retention_min = float(cfg.get("retention_min", 0.6) or 0)
        self._retention_relance = bool(cfg.get("retention_relance", True))
        self._retention_annexe = bool(cfg.get("retention_annexe", True))
        self._documents_secondaires = [str(x) for x in (cfg.get("documents_secondaires") or [])]
        self._plafond_secondaires = int(cfg.get("plafond_secondaires", 2) or 0)
        self._prompts = prompts if prompts is not None else self._charger_prompts()
        if self._repli_esquive and not self._deep:
            logger.warning("query2: repli_esquive actif mais `query2.deep` vide : "
                           "aucune passe élargie ne sera faite")

    @staticmethod
    def _charger_prompts() -> Dict:
        try:
            return dict(load_prompts().config or {})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prompts.yaml illisible ({e}) : transformation désactivée")
            return {}

    # ---------------------------------------------------------------- run
    def run(self, query: str, custom_prompt: Optional[str] = None,
            history: Optional[List[Dict]] = None, deep: bool = False,
            **_ignore) -> Dict:
        """Retourne le contrat de la route classique (response, sources,
        search_results, metadata) augmenté d'une clé `iterative` (trace)."""
        analyse: Optional[Dict] = None
        origine = "demande"
        questions = [query]
        if self._transformation and _question_directe(query):
            # Une question directe est déjà la meilleure formulation possible :
            # ni analyse, ni gabarit, ni cadrage (elle est son propre cadrage).
            origine = "question_directe"
            logger.info("query2: question directe, posée telle quelle")
        elif self._transformation:
            analyse = self._analyser(query)
            if analyse:
                questions = self._composer(analyse)
                origine = "gabarits"
            if not analyse or not questions:
                questions = self._questions_llm(query)
                origine = "llm" if questions != [query] else "demande"
                questions = questions or [query]

        cadrage = self._cadrage_texte(query, analyse) if (self._cadrage and analyse) else None
        prompt_generation = "\n\n".join(x for x in (cadrage, custom_prompt) if x) or None

        trace = {"version": "pipeline_v3.6", "mode": "deep" if deep else "standard",
                 "analyse": analyse, "origine_questions": origine,
                 "questions": questions, "cadrage": bool(cadrage),
                 "custom_prompt": bool(custom_prompt), "reponses": []}
        logger.info(f"query2: {len(questions)} question(s) [{origine}] : {questions}")

        resultats = []
        for q in questions:
            overrides = self._objet if (self._objet and q in self._questions_objet) else None
            r, bloc = self._repondre_avec_gardes(q, prompt_generation, history, deep, overrides)
            trace["reponses"].append(bloc)
            resultats.append((q, r))

        trace["couverture"] = {"verdict": self._verdict(trace["reponses"])}
        trace["recherche_b"] = {"effectuee": any(x["repli_elargi"] for x in trace["reponses"])}
        trace["gardes"] = {
            "esquive": any(x["esquive"] for x in trace["reponses"]),
            "tronque": any(x["tronque"] for x in trace["reponses"]),
            "degenere": any(x["degenere"] for x in trace["reponses"]),
            "hors_lot": sorted({p for x in trace["reponses"] for p in x["pages_hors_lot"]}),
            "nombres_hors_extraits": sorted({n for x in trace["reponses"] for n in x["nombres_hors_extraits"]}),
        }

        result = self._assembler(query, resultats)
        if self._synthese and analyse and len(resultats) > 1:
            texte, info = self._synthetiser(query, analyse, resultats)
            trace["synthese"] = info
            if info.get("pages_hors_lot"):
                trace["gardes"]["hors_lot"] = sorted(set(trace["gardes"]["hors_lot"])
                                                     | set(info["pages_hors_lot"]))
                if trace["couverture"]["verdict"] == "COUVERT":
                    trace["couverture"]["verdict"] = "PARTIEL"
            if texte:
                if self._annexe_blocs:
                    texte = f"{texte.rstrip()}\n\n---\n\n## Détail par question\n\n{result['response']}"
                result["response"] = texte
        else:
            trace["synthese"] = {"effectuee": False, "motif": "non applicable"}
        result["iterative"] = trace
        return result

    # ------------------------------------------------------- étape 3 (LLM)
    @staticmethod
    def _blocs_texte(resultats: List) -> str:
        """Blocs question/réponse pour la synthèse, sans les notes de garde."""
        sections = []
        for question, r in resultats:
            lignes = [l for l in (r.get("response") or "").splitlines()
                      if not l.startswith("_Pages citées sans extrait")
                      and not l.startswith("_Nombres sans extrait")]
            sections.append(f"### {question}\n\n" + "\n".join(lignes).strip())
        return "\n\n".join(sections)

    def _synthetiser(self, query: str, analyse: Dict, resultats: List):
        """Rédige la réponse finale à partir des blocs. Retourne (texte, info) ;
        texte vaut None si la synthèse est impossible ou rejetée par les gardes
        (le pipeline garde alors la concaténation)."""
        system = self._prompts.get("synthese_system_prompt")
        template = self._prompts.get("synthese_prompt")
        info: Dict = {"effectuee": False, "tronque": False, "degenere": False,
                      "relance": False, "pages_hors_lot": [], "tokens": {}}
        if not system or not template:
            info["motif"] = "synthese_prompt absent de prompts.yaml"
            logger.info("query2: synthèse non configurée, concaténation conservée")
            return None, info
        prompt = _formater("synthese_prompt", template, query=query,
                           blocs=self._blocs_texte(resultats),
                           **self._champs_analyse(analyse))
        llm = self.analyzer.llm_generator

        def appel() -> Optional[str]:
            try:
                return llm.call_llm(system, prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"query2: échec de la synthèse ({e})")
                return None

        texte = appel()
        usage = self._usage()
        if not texte:
            info["motif"] = "appel LLM en échec"
            return None, info
        tronque = bool(usage.get("tronque"))
        degenere = _reponse_degeneree(texte, **self._degenere)
        if (tronque or degenere) and self._relance_boucle:
            logger.warning(f"query2: synthèse {'tronquée' if tronque else 'dégénérée'} -> relance {self._relance}")
            with self._surcharge_llm(**self._relance):
                texte2 = appel()
            usage2 = self._usage()
            info["relance"] = True
            if texte2 and not usage2.get("tronque") and not _reponse_degeneree(texte2, **self._degenere):
                texte, usage, tronque, degenere = texte2, usage2, False, False
        info.update(tronque=tronque, degenere=degenere,
                    tokens={"prompt": usage.get("prompt"), "generes": usage.get("generes")})
        if tronque or degenere:
            info["motif"] = "synthèse tronquée ou dégénérée après relance"
            logger.warning("query2: synthèse rejetée, concaténation conservée")
            return None, info

        texte_lot = "\n".join(_texte_du_lot(r) for _, r in resultats)

        # Rétention des données : ce que les blocs portent (nombres, noms
        # propres présents dans les extraits) doit se retrouver dans la
        # synthèse. Sous le seuil, une relance nomme ce qui manque ; si
        # cela ne suffit pas, les lignes manquantes sont jointes telles
        # quelles (jamais réécrites) sous la synthèse.
        blocs = self._blocs_texte(resultats)
        taux, manquantes, lignes = _retention(texte, blocs, texte_lot)
        info["retention"] = {"taux_initial": round(taux, 2), "taux": round(taux, 2),
                             "manquantes": manquantes, "relance": False, "annexe": False}
        if manquantes:
            logger.info(f"query2: synthèse, rétention des données {taux:.2f} ; "
                        f"manquantes : {manquantes}")
        if taux < self._retention_min and lignes and self._retention_relance:
            complement = (
                "\n\nLes éléments documentés contiennent les données suivantes, "
                "que la rédaction précédente a omises. Reprends chacune, avec sa "
                "valeur exacte et sa citation, dans la partie où elle répond à la "
                "demande :\n" + "\n".join(f"- {l}" for l in lignes[: self.RETENTION_LIGNES_MAX]))
            logger.warning(f"query2: rétention {taux:.2f} < {self._retention_min} -> "
                           f"relance de la synthèse avec {len(lignes)} ligne(s) manquante(s)")
            texte2 = None
            try:
                texte2 = llm.call_llm(system, prompt + complement)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"query2: échec de la relance de synthèse ({e})")
            usage2 = self._usage()
            info["retention"]["relance"] = True
            if (texte2 and not usage2.get("tronque")
                    and not _reponse_degeneree(texte2, **self._degenere)):
                taux2, manquantes2, lignes2 = _retention(texte2, blocs, texte_lot)
                logger.info(f"query2: synthèse relancée, rétention {taux2:.2f} (avant {taux:.2f})")
                if taux2 > taux:
                    texte, usage = texte2, usage2
                    taux, manquantes, lignes = taux2, manquantes2, lignes2
            info["retention"].update(taux=round(taux, 2), manquantes=manquantes)
        if taux < self._retention_min and lignes and self._retention_annexe:
            texte = (texte.rstrip() + "\n\n### Données du dossier non reprises ci-dessus\n\n"
                     + "\n".join(f"- {l}" for l in lignes[: self.RETENTION_LIGNES_MAX]))
            info["retention"]["annexe"] = True
            logger.warning(f"query2: rétention {taux:.2f} après relance, "
                           f"{len(lignes)} ligne(s) des blocs jointes à la synthèse")

        # Pages citées : comparées à l'union des lots de toutes les questions
        pages_lot: Dict[str, Set[int]] = {}
        for _, r in resultats:
            for nom, pages in _pages_du_lot(r).items():
                pages_lot.setdefault(nom, set()).update(pages)
        hors = _pages_hors_lot(texte, pages_lot)
        info["pages_hors_lot"] = hors
        nombres = _nombres_hors_extraits(texte, texte_lot)
        info["nombres_hors_extraits"] = nombres
        if nombres:
            logger.warning(f"query2: synthèse, nombres absents des extraits : {nombres}")
            if self._signaler_nombres:
                texte = (texte.rstrip() + "\n\n_Nombres sans extrait correspondant dans le lot : "
                         + ", ".join(nombres) + "._")
        if hors:
            logger.warning(f"query2: synthèse, pages citées hors lot : {hors}")
            if self._signaler_hors_lot:
                texte = (texte.rstrip() + "\n\n_Pages citées sans extrait correspondant dans le lot : "
                         + ", ".join(hors) + "._")
        info["effectuee"] = True
        logger.info(f"query2: synthèse rédigée ({usage.get('generes')} tokens, hors lot : {len(hors)})")
        return texte, info

    @staticmethod
    def _verdict(blocs: List[Dict]) -> str:
        if blocs and all(b["esquive"] for b in blocs):
            return "NON_COUVERT"
        if any(b["esquive"] or b["tronque"] or b["degenere"] or b["pages_hors_lot"]
               for b in blocs):
            return "PARTIEL"
        return "COUVERT"

    # ------------------------------------------------------- étape 1a
    def _analyser(self, query: str) -> Optional[Dict]:
        """Structure courte de la demande : objet, motifs, à démontrer."""
        system = self._prompts.get("analyse_system_prompt")
        template = self._prompts.get("analyse_prompt")
        if not system or not template:
            logger.info("query2: analyse_prompt absent de prompts.yaml, chemin `questions_prompt`")
            return None
        prompt = _formater("analyse_prompt", template, query=query)
        try:
            brut = self.analyzer.llm_generator.call_llm(system, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"query2: échec de l'analyse ({e})")
            return None
        data = _extraire_json(brut) or {}
        objet = str(data.get("objet") or "").strip().strip('"').rstrip(".")
        objet, type_impose = self._normaliser_objet(objet)
        objet, type_impose = self._objet_integration(query, objet, type_impose)
        motifs: List[str] = []
        for m in data.get("motifs") or []:
            m = str(m).strip().strip('"').rstrip(".")
            if len(m) >= 3 and m.lower() not in (x.lower() for x in motifs):
                motifs.append(m)
        motifs = self._filtrer_motifs(motifs)
        motifs = self._regrouper_motifs(query, motifs)
        demontrer = str(data.get("demontrer") or "").strip().strip('"')
        aspects: List[str] = []
        for a in data.get("aspects") or []:
            a = str(a).strip().strip('"').rstrip(".")
            if len(a) >= 3 and a.lower() not in (x.lower() for x in aspects):
                aspects.append(a)
        # Un aspect qui redit l'objet (fréquent quand l'objet vient d'être
        # corrigé par le code) ne mérite pas de question à part.
        cle_objet = self._cle_item(objet)
        redits = [a for a in aspects if cle_objet and
                  (cle_objet in self._cle_item(a) or self._cle_item(a) in cle_objet)]
        if redits:
            logger.info(f"query2: aspects redisant l'objet, ignorés : {redits}")
            aspects = [a for a in aspects if a not in redits]
        comparaison = data.get("comparaison")
        if isinstance(comparaison, str):
            comparaison = comparaison.strip().lower() in ("true", "oui", "vrai", "yes", "1")
        comparaison = bool(comparaison)
        type_objet = _normaliser_type(data.get("type_objet"))
        if comparaison and type_objet == "":
            type_objet = "comparaison"
        if type_impose and type_objet in ("", "etude"):
            # L'objet nommait l'action (« impacts de X ») : la chose visée
            # est X, un élément du projet, pas une étude.
            if type_objet:
                logger.info(f"query2: type_objet « {type_objet} » remplacé par « {type_impose} » (objet = élément du projet)")
            type_objet = type_impose
        if type_objet == "":
            type_objet = self.TYPE_DEFAUT
        comparaison = comparaison or type_objet == "comparaison"
        acte = str(data.get("acte") or "").strip().strip('"').rstrip(".")
        if not objet and not motifs:
            logger.warning(f"query2: analyse vide ou illisible : {brut[:800]!r}")
            return None
        items = self._items_attendus(data.get("items"), objet, motifs, aspects)
        analyse = {"objet": objet, "type_objet": type_objet, "acte": acte,
                   "motifs": motifs, "aspects": aspects, "items": items,
                   "demontrer": demontrer, "comparaison": comparaison}
        logger.info(f"query2: analyse {analyse}")
        return analyse

    # ------------------------------------------------- items attendus
    # Sortes de données concrètes que chaque question doit nommer, pour que
    # le retrieval ramène les pages de détail et que la génération recopie
    # les valeurs. Deux sources : la table `items_par_theme` de prompts.yaml
    # (vocabulaire d'étude d'impact, relu, prioritaire) et le champ `items`
    # de l'analyse LLM, gardé seulement sous la forme d'une donnée (tête de
    # grandeur : « nom de chaque… », « distance… », « niveau de… »).
    ITEMS_MAX = 10
    _TETE_ITEM = re.compile(
        r"^(?:nom|noms|nombre|liste|longueur|largeur|profondeur|hauteur|surface|"
        r"volume|distance|distances|tension|puissance|niveau|niveaux|statut|statuts|"
        r"periode|periodes|date|dates|intitule|intitules|type|types|section|numero|"
        r"numeros|valeur|valeurs|taux|effectif|effectifs|espece|especes|localisation|"
        r"trace|capacite|duree|frequence|angle|angles|emprise|superficie|pourcentage|"
        r"denomination|reference)\b")

    @staticmethod
    def _cle_item(texte: str) -> str:
        return re.sub(r"\s+", " ", sans_accents(texte).lower()).strip(" .")

    @staticmethod
    def _theme_present(theme: str, plat: str) -> bool:
        """Un déclencheur est reconnu en début de mot seulement (« eau » ne
        doit pas reconnaître « oiseaux », « sol » reconnaît « sols »)."""
        return bool(theme) and re.search(r"(?<![a-z0-9])" + re.escape(theme), plat) is not None

    def _items_table(self, libelle: str, exclure: Optional[Set[int]] = None):
        """Items de la table `items_par_theme` dont un déclencheur apparaît
        dans `libelle` (comparaison désaccentuée, sous-chaîne). Retourne
        (items, indices des entrées retenues) ; `exclure` = entrées déjà
        utilisées pour l'objet, à ne pas répéter sur un aspect."""
        plat = " " + self._cle_item(libelle) + " "
        trouves: List[str] = []
        retenues: Set[int] = set()
        for i, entree in enumerate(self._prompts.get("items_par_theme") or []):
            if exclure and i in exclure:
                continue
            themes = entree.get("themes") if isinstance(entree, dict) else None
            valeurs = entree.get("items") if isinstance(entree, dict) else None
            if not themes or not valeurs:
                continue
            if any(self._theme_present(self._cle_item(str(t)), plat) for t in themes):
                retenues.add(i)
                for v in valeurs:
                    v = str(v).strip()
                    if v and v.lower() not in (x.lower() for x in trouves):
                        trouves.append(v)
        return trouves, retenues

    def _items_llm(self, brut) -> Dict[str, List[str]]:
        """Champ `items` de l'analyse, nettoyé : clé normalisée -> items qui
        ont la forme d'une donnée. Les rubriques (« photomontages »,
        « emprise des ouvrages ») sont écartées."""
        if not isinstance(brut, dict):
            return {}
        source: Dict[str, List[str]] = {}
        ecartes: List[str] = []
        for cle, valeurs in brut.items():
            if isinstance(valeurs, str):
                valeurs = re.split(r"[;,]", valeurs)
            if not isinstance(valeurs, list):
                continue
            propres: List[str] = []
            for v in valeurs:
                v = str(v).strip().strip('"').strip(" .;")
                if len(v) < 2 or v.lower() in (x.lower() for x in propres):
                    continue
                if self._TETE_ITEM.match(self._cle_item(v)):
                    propres.append(v)
                else:
                    ecartes.append(v)
            if propres:
                source[self._cle_item(str(cle))] = propres
        if ecartes:
            logger.info(f"query2: items du LLM sans forme de donnée, ignorés : {ecartes}")
        return source

    def _items_attendus(self, brut, objet: str, motifs: List[str],
                        aspects: List[str]) -> Dict[str, List[str]]:
        """{"objet": [...], <motif>: [...], <aspect>: [...]} : table d'abord,
        puis items LLM valides, sans doublon, ITEMS_MAX par entrée. Un motif
        regroupé par le code (« X et Y ») reçoit les items donnés pour X et
        pour Y."""
        llm = self._items_llm(brut)
        resultat: Dict[str, List[str]] = {}
        deja: Set[int] = set()  # entrées de table déjà posées sur l'objet

        def composer(cle_resultat: str, libelle: str, cles_llm: List[str], exclure=None):
            fusion, entrees = self._items_table(libelle, exclure)
            deja.update(entrees)
            origine = "table" if fusion else ""
            for c in cles_llm:
                for v in llm.get(c, []):
                    if v.lower() not in (x.lower() for x in fusion):
                        fusion.append(v)
                        origine = origine + "+llm" if origine and not origine.endswith("llm") else (origine or "llm")
            if fusion:
                resultat[cle_resultat] = fusion[: self.ITEMS_MAX]
                logger.info(f"query2: items « {cle_resultat} » ({origine}) : {resultat[cle_resultat]}")

        if objet:
            composer("objet", objet, ["objet", self._cle_item(objet)])
        for motif in motifs:
            composer(motif, motif, [self._cle_item(motif)]
                     + [self._cle_item(p) for p in re.split(r"\s+et\s+", motif)])
        for aspect in aspects:
            # Un aspect ne reprend pas les items déjà posés sur l'objet (un
            # aspect du raccordement n'a pas besoin de « tension des câbles »),
            # mais reçoit ceux d'un autre thème qu'il nomme (« parcs voisins »).
            composer(aspect, aspect, [self._cle_item(aspect)], exclure=set(deja))
        return resultat

    # « impacts du X », « évaluation des effets prévisibles de X » : l'objet
    # est X. Comparé sur le texte désaccentué (même longueur que l'original,
    # les positions restent valables).
    _OBJET_ACTION = re.compile(
        r"^(?:l')?(?:evaluation|analyse|etude|estimation|prise en compte)?\s*"
        r"(?:des|de la|de l'|du|d')?\s*(?:impacts?|effets?|incidences?)\s+"
        r"(?:previsibles?|potentiels?|residuels?|attendus?|environnementaux)?\s*"
        r"(?:du|de la|de l'|des|d')\s*(.+)$", re.I)

    def _normaliser_objet(self, objet: str):
        """Si l'objet nomme l'action d'évaluer les impacts de X plutôt que
        X lui-même, retourne (X, "element") ; sinon (objet, None). Règle de
        langue, sans contenu de dossier ; « impacts du projet … » est laissé
        tel quel (le projet n'est pas un élément)."""
        if not objet:
            return objet, None
        m = self._OBJET_ACTION.match(sans_accents(objet).lower())
        if not m:
            return objet, None
        chose = objet[m.start(1):m.end(1)].strip()
        if not chose or sans_accents(chose).lower().startswith("projet"):
            return objet, None
        if re.search(r"\s(?:sur|vis-a-vis|pour)\s", sans_accents(chose).lower()):
            # « impacts du fonctionnement des éoliennes sur les oiseaux » :
            # une relation d'effet, donc une étude, pas un élément du projet.
            return objet, None
        logger.info(f"query2: objet « {objet} » ramené à l'élément visé « {chose} »")
        return chose, "element"

    # « intégrer X dans Y », « prendre en compte X dans Y » : la demande
    # porte sur X (ce qu'il faut ajouter), pas sur Y (l'étude qui doit
    # l'accueillir). Comparé sur la demande désaccentuée (même longueur).
    _INTEGRATION = [re.compile(
        r"\b(?:d')?(?:integrer|inclure|ajouter|prendre en compte|considerer|introduire)\s+"
        r"(?:les?\s+|la\s+|l'|des\s+|du\s+|de la\s+|de l'|une?\s+)?(.+?)\s+"
        r"(?:" + lien + r")\s+(?:les?\s+|la\s+|l'|des\s+|du\s+|de la\s+|de l'|"
        r"son\s+|sa\s+|ses\s+|leur\s+|leurs\s+)?(.+?)(?:[.,;:]|$)", re.I)
        for lien in ("dans|au sein de", "a|au|aux")]  # « dans » d'abord : « à » est fréquent dans X

    def _objet_integration(self, query: str, objet: str, type_impose):
        """Si la demande est de la forme « intégrer X dans Y » et que l'objet
        de l'analyse n'est pas X, l'objet devient X (type element sauf type
        déjà imposé). Règle de langue sur le texte de la demande."""
        plat = sans_accents(query).lower()
        m = next((m for m in (r.search(plat) for r in self._INTEGRATION) if m), None)
        if not m:
            return objet, type_impose
        x = query[m.start(1):m.end(1)].strip(" .,;:")
        y = query[m.start(2):m.end(2)].strip(" .,;:")
        if not x or len(x) < 3 or len(x) > 120:
            return objet, type_impose
        cle_x, cle_o = self._cle_item(x), self._cle_item(objet)
        if cle_o and (cle_o in cle_x or cle_x in cle_o):
            return objet, type_impose
        logger.info(f"query2: demande « intégrer X dans Y » : objet « {objet} » "
                    f"remplacé par « {x} » (Y = « {y} »)")
        return x, type_impose or "element"

    MOTIFS_GENERIQUES = ("environnement", "milieux", "milieu", "impacts", "effets",
                         "incidences", "enjeux")

    def _filtrer_motifs(self, motifs: List[str]) -> List[str]:
        """Écarte les motifs qui ne nomment aucun thème (« environnement »,
        « milieux »…) : ils ne produiraient qu'une question sans objet."""
        generiques = {sans_accents(g).lower() for g in
                      (self._motifs_generiques or self.MOTIFS_GENERIQUES)}
        gardes = [m for m in motifs if sans_accents(m).lower().strip() not in generiques]
        if gardes != motifs:
            logger.info(f"query2: motifs génériques ignorés : "
                        f"{[m for m in motifs if m not in gardes]}")
        return gardes

    @staticmethod
    def _regrouper_motifs(query: str, motifs: List[str]) -> List[str]:
        """Règle de regroupement de la demande, appliquée par le code : deux
        motifs consécutifs que la demande coordonne (« X et Y », article
        facultatif devant Y) ne forment qu'un motif « X et Y ». Le LLM
        n'applique pas cette règle de façon stable ; ici elle ne dépend
        que du texte de la demande."""
        if len(motifs) < 2:
            return motifs
        plat = re.sub(r"\s+", " ", sans_accents(query).lower())
        resultat: List[str] = []
        i = 0
        while i < len(motifs):
            a = motifs[i]
            if i + 1 < len(motifs):
                b = motifs[i + 1]
                motif_re = (re.escape(sans_accents(a).lower())
                            + r"\s+et\s+(?:les?\s+|la\s+|l')?"
                            + re.escape(sans_accents(b).lower()))
                if re.search(motif_re, plat):
                    resultat.append(f"{a} et {b}")
                    i += 2
                    continue
            resultat.append(a)
            i += 1
        if resultat != motifs:
            logger.info(f"query2: motifs regroupés selon la demande : {motifs} -> {resultat}")
        return resultat

    # ------------------------------------------------------- étape 1b
    def _composer(self, analyse: Dict) -> List[str]:
        """Questions composées par gabarits : l'objet d'abord, puis un
        motif par question. Aucun contenu métier ici : les gabarits sont
        dans prompts.yaml (`gabarits_questions`)."""
        gabarits = self._prompts.get("gabarits_questions") or {}
        g_motif, g_aspect = gabarits.get("motif"), gabarits.get("aspect")
        if not gabarits:
            logger.info("query2: gabarits_questions absent de prompts.yaml")
            return []
        questions: List[str] = []
        objet = analyse.get("objet")
        type_objet = analyse.get("type_objet") or self.TYPE_DEFAUT
        items = analyse.get("items") or {}
        # `{items}` dans un gabarit reçoit « — en précisant : a, b, c » (ou
        # rien) : les sortes de données concrètes que l'analyse attend à ce
        # sujet, qui orientent le retrieval vers les pages de détail et la
        # génération vers les valeurs. Un gabarit sans `{items}` l'ignore.
        if objet:
            # Une question par gabarit du type (« objet_<type> », puis
            # « objet_<type>_2 » s'il existe : la seconde vise les pages de
            # détail que la première ne fait pas remonter). Sans gabarit
            # pour ce type, repli sur le type par défaut, puis sur « objet ».
            for cle in self._gabarits_objet(gabarits, type_objet):
                questions.append(_formater(f"gabarits_questions.{cle}", gabarits[cle],
                                           objet=objet,
                                           items=self._items_texte(items.get("objet"))))
        self._questions_objet.update(questions)
        # Aspects : points précis que la demande pose sur l'objet ; la question
        # reste rattachée à l'objet pour ne pas dériver vers un autre sujet.
        q_aspects: List[str] = []
        for aspect in analyse.get("aspects") or []:
            if objet and g_aspect:
                q_aspects.append(_formater("gabarits_questions.aspect", g_aspect,
                                           objet=objet, aspect=aspect,
                                           items=self._items_texte(items.get(aspect))))
        q_motifs: List[str] = []
        for motif in analyse.get("motifs") or []:
            if g_motif:
                q_motifs.append(_formater("gabarits_questions.motif", g_motif, motif=motif,
                                          items=self._items_texte(items.get(motif))))
        total = len(questions) + len(q_aspects) + len(q_motifs)
        if total > self._max_questions:
            # Plafond : l'objet et les motifs passent avant les aspects (le
            # champ le moins stable de l'analyse) ; un motif invoqué par la
            # demande ne doit jamais être sacrifié à un aspect.
            place = max(0, self._max_questions - len(questions) - len(q_motifs))
            ecartees = q_aspects[place:]
            q_aspects = q_aspects[:place]
            logger.warning(f"query2: {total} questions composées, limitées à "
                           f"max_questions={self._max_questions} ; aspects écartés : "
                           f"{[a[:60] for a in ecartees]}")
        questions = questions + q_aspects + q_motifs
        if len(questions) > self._max_questions:
            logger.warning(f"query2: objet et motifs dépassent encore max_questions, "
                           f"{len(questions) - self._max_questions} motif(s) écarté(s)")
        return questions[: self._max_questions]

    @staticmethod
    def _items_texte(liste) -> str:
        """Suffixe de question à partir d'une liste d'items ; vide si aucun."""
        liste = [str(x).strip() for x in (liste or []) if str(x).strip()]
        return f" — en précisant : {', '.join(liste)}" if liste else ""

    TYPE_DEFAUT = "element"
    # Compatibilité v3.1/v3.2 : anciens noms de gabarits pour chaque type.
    ALIAS_GABARITS = {
        "comparaison": ("objet", "objet_comparaison"),
        "element": ("objet_description",),
    }

    def _gabarits_objet(self, gabarits: Dict, type_objet: str) -> List[str]:
        """Clés de gabarits à poser pour l'objet, dans l'ordre."""
        for t in (type_objet, self.TYPE_DEFAUT):
            # Anciens noms (prompts.yaml v5.1/v5.2) : utilisés s'ils sont tous
            # présents, ce qui n'est jamais le cas d'un prompts.yaml v5.3.
            anciens = self.ALIAS_GABARITS.get(t, ())
            cles = list(anciens) if anciens and all(gabarits.get(c) for c in anciens) else []
            if not cles:
                cles = [c for c in (f"objet_{t}", f"objet_{t}_2") if gabarits.get(c)]
            if cles:
                if t != type_objet:
                    logger.info(f"query2: pas de gabarit pour le type « {type_objet} », "
                                f"repli sur « {t} »")
                return cles
        return ["objet"] if gabarits.get("objet") else []

    def _questions_llm(self, query: str) -> List[str]:
        """Ancien chemin : questions rédigées par le LLM (`questions_prompt`).
        Repli sur la demande telle quelle en cas d'échec."""
        system = self._prompts.get("questions_system_prompt")
        template = self._prompts.get("questions_prompt")
        if not system or not template:
            logger.warning("query2: questions_prompt absent de prompts.yaml, demande conservée")
            return [query]
        prompt = _formater("questions_prompt", template, query=query,
                           max_questions=self._max_questions)
        try:
            brut = self.analyzer.llm_generator.call_llm(system, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"query2: échec transformation ({e}), demande conservée")
            return [query]
        data = _extraire_json(brut) or {}
        questions: List[str] = []
        for brute in (data.get("questions") or [])[: self._max_questions]:
            q = str(brute).strip().strip('"').strip()
            if len(q) >= 10 and q.lower() not in (x.lower() for x in questions):
                questions.append(q)
        if not questions:
            logger.warning("query2: transformation vide, demande conservée")
            return [query]
        return questions

    def _cadrage_texte(self, query: str, analyse: Optional[Dict]) -> Optional[str]:
        template = self._prompts.get("cadrage_prompt")
        if not template:
            return None
        a = analyse or {}
        return _formater("cadrage_prompt", template, query=query,
                         **self._champs_analyse(a)).strip() or None

    def _champs_analyse(self, a: Dict) -> Dict:
        """Champs de l'analyse disponibles dans cadrage_prompt et synthese_prompt."""
        type_objet = a.get("type_objet") or self.TYPE_DEFAUT
        libelles = self._prompts.get("types_objet") or {}
        return {"objet": a.get("objet", ""), "demontrer": a.get("demontrer", ""),
                "type_objet": type_objet,
                "type_objet_libelle": str(libelles.get(type_objet) or type_objet),
                "acte": a.get("acte") or "répondre à la demande",
                "motifs": ", ".join(a.get("motifs") or []) or "aucun",
                "aspects": ", ".join(a.get("aspects") or []) or "aucun",
                "items": "; ".join(f"{k} : {', '.join(v)}" for k, v in (a.get("items") or {}).items()) or "aucun"}

    # ------------------------------------------------------- étape 2
    def _repondre(self, question: str, prompt_generation: Optional[str],
                  history: Optional[List[Dict]], deep: bool,
                  overrides: Optional[Dict] = None) -> Dict:
        options: Dict = {}
        if prompt_generation:
            options["custom_prompt"] = prompt_generation
        if deep and self._deep:
            options["retrieval_overrides"] = dict(self._deep)
            logger.info(f"query2: profil élargi {self._deep}")
        elif overrides:
            options["retrieval_overrides"] = dict(overrides)
            logger.info(f"query2: profil « objet » {overrides}")
        return self.analyzer.analyze(query=question, mode="chat",
                                     options=options, history=history)

    def _usage(self) -> Dict:
        """Mesure de la dernière génération (llm.py v3.1 : `last_usage`)."""
        return dict(getattr(self.analyzer.llm_generator, "last_usage", None) or {})

    def _surcharge_llm(self, **options):
        """Contexte de surcharge des options LLM (llm.py v3.1 : `surcharge`)."""
        surcharge = getattr(self.analyzer.llm_generator, "surcharge", None)
        return surcharge(**options) if callable(surcharge) else nullcontext()

    def _repondre_avec_gardes(self, q: str, prompt_generation: Optional[str],
                              history: Optional[List[Dict]], deep: bool,
                              overrides: Optional[Dict] = None):
        r = self._repondre(q, prompt_generation, history, deep, overrides)
        usage = self._usage()
        bloc = {"question": q, "esquive": False, "repli_elargi": False,
                "tronque": False, "degenere": False, "relance": False,
                "pages_hors_lot": [], "nombres_hors_extraits": [], "passages": 0, "etiquettes": [],
                "tokens": {"prompt": usage.get("prompt"), "generes": usage.get("generes")}}

        # 1. esquive -> passe élargie
        esquive = _reponse_esquive(r.get("response", ""))
        if esquive and self._repli_esquive and not deep and self._deep:
            logger.info(f"query2: esquive sur '{q[:60]}' -> passe élargie")
            r2 = self._repondre(q, prompt_generation, history, deep=True)
            usage2 = self._usage()
            if not _reponse_esquive(r2.get("response", "")):
                r, esquive, usage = r2, False, usage2
                bloc["repli_elargi"] = True
                bloc["tokens"] = {"prompt": usage2.get("prompt"), "generes": usage2.get("generes")}
            else:
                logger.info("query2: la passe élargie esquive aussi, première réponse conservée")
        bloc["esquive"] = esquive

        # 2. tronqué / dégénéré -> relance sous pénalité de répétition
        tronque = bool(usage.get("tronque"))
        degenere = _reponse_degeneree(r.get("response", ""), **self._degenere)
        if (tronque or degenere) and self._relance_boucle:
            logger.warning(f"query2: bloc {'tronqué' if tronque else 'dégénéré'} sur "
                           f"'{q[:60]}' -> relance {self._relance}")
            with self._surcharge_llm(**self._relance):
                r2 = self._repondre(q, prompt_generation, history,
                                    deep=bool(bloc["repli_elargi"]), overrides=overrides)
            usage2 = self._usage()
            tronque2 = bool(usage2.get("tronque"))
            degenere2 = _reponse_degeneree(r2.get("response", ""), **self._degenere)
            bloc["relance"] = True
            if not (tronque2 or degenere2) and not _reponse_esquive(r2.get("response", "")):
                r, tronque, degenere = r2, False, False
                bloc["tokens"] = {"prompt": usage2.get("prompt"), "generes": usage2.get("generes")}
            else:
                logger.warning("query2: la relance est encore tronquée/dégénérée, "
                               "première réponse conservée")
        bloc["tronque"], bloc["degenere"] = tronque, degenere

        # 3. pages citées hors lot -> signalé
        hors = _pages_hors_lot(r.get("response", ""), _pages_du_lot(r))
        if hors:
            logger.warning(f"query2: pages citées hors lot sur '{q[:60]}' : {hors}")
            if self._signaler_hors_lot:
                r = dict(r)
                r["response"] = ((r.get("response") or "").rstrip()
                                 + "\n\n_Pages citées sans extrait correspondant dans le lot : "
                                 + ", ".join(hors) + "._")
        bloc["pages_hors_lot"] = hors

        # 4. nombres absents des extraits -> signalé (jamais un repli : un
        #    nombre que le lot ne porte pas ne doit pas être « corrigé »)
        nombres = _nombres_hors_extraits(r.get("response", ""), _texte_du_lot(r))
        if nombres:
            logger.warning(f"query2: nombres absents des extraits sur '{q[:60]}' : {nombres}")
            if self._signaler_nombres:
                r = dict(r)
                r["response"] = ((r.get("response") or "").rstrip()
                                 + "\n\n_Nombres sans extrait correspondant dans le lot : "
                                 + ", ".join(nombres) + "._")
        bloc["nombres_hors_extraits"] = nombres

        passages = r.get("search_results", []) or []
        bloc["passages"] = len(passages)
        bloc["etiquettes"] = [_etiquette(p) for p in passages]
        bloc["pages_citees"] = sorted({f"{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "")
                                       for m in _PAGE_CITEE.finditer(r.get("response", ""))})
        logger.info(f"query2: lot '{q[:50]}' ({len(passages)} passages) : {bloc['etiquettes']}")
        logger.info(f"query2: pages citées : {bloc['pages_citees']} ; hors lot : {hors}")
        return r, bloc

    # ------------------------------------------------------- étape 3
    @staticmethod
    def _assembler(query: str, resultats: List) -> Dict:
        if len(resultats) == 1:
            r = dict(resultats[0][1])
            r["metadata"] = dict(r.get("metadata") or {}, query=query)
            return r

        sections, sources, passages, vus_src, vus_pas = [], [], [], set(), set()
        modele = "unknown"
        for question, r in resultats:
            sections.append(f"### {question}\n\n{(r.get('response') or '').strip()}")
            for s in r.get("sources", []):
                cle = (s.get("file_name"), s.get("page_start"), s.get("page_end"))
                if cle not in vus_src:
                    vus_src.add(cle)
                    sources.append(s)
            for p in r.get("search_results", []):
                cle = _cle_passage(p)
                if cle not in vus_pas:
                    vus_pas.add(cle)
                    passages.append(p)
            modele = (r.get("metadata") or {}).get("model", modele)

        return {
            "result_type": "chat",
            "response": "\n\n".join(sections),
            "sources": sources,
            "search_results": passages,
            "metadata": {"model": modele, "query": query,
                         "questions": [q for q, _ in resultats]},
        }


# Nom historique, conservé pour service.py et les tests.
IterativePipeline = Pipeline
