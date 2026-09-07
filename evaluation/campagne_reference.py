#!/usr/bin/env python3
"""Campagne de comparaison Luciole / memoires wpd (avis MRAe).

Rejoue un jeu de remarques MRAe sur /api/query2 et compare chaque
reponse a la reponse REELLE redigee par wpd dans son memoire en reponse.

Le jeu (JSONL, une ligne par remarque) :

  {"id": "beaumont-03", "projet": "Beaumont Sud",
   "question": "<texte de la remarque MRAe>",
   "reponse_reference": "<texte de la reponse wpd>",
   "sources_reference": ["Tome 4 p. 239", "RNT p. 62"]}

Pour chaque cas, quatre mesures :

  1. esquive      : detecteur a trois etats (ESQUIVE / concede / repond),
                    calcule sur le texte SANS ACCENTS (correctif du
                    detecteur : les formes "ne precise pas", "aucun
                    element" n'etaient jamais reconnues) ;
  2. bon tome     : au moins un tome de sources_reference figure dans
                    les passages soumis au modele ;
  3. couverture   : verdict du juge query2 (COUVERT / PARTIEL /
                    NON_COUVERT) et recherche B effectuee ou non ;
  4. fidelite     : un juge LLM compare la reponse Luciole a la reponse
                    wpd et note trois criteres de 0 a 2 en JSON :
                      - faits    : les faits cles de wpd sont-ils presents ?
                      - sources  : les memes documents sont-ils cites ?
                      - invention: la reponse ajoute-t-elle des faits
                                   absents de la reference ? (2 = aucun)
                    Le juge ne voit pas le corpus, seulement les deux
                    textes : il mesure la proximite avec wpd, pas la
                    verite documentaire. Une note "faits" basse peut
                    aussi signifier que wpd a utilise une information
                    hors dossier — a relire par un humain.

Deux passages sont conseilles (LABEL=ref-1 puis ref-2) pour mesurer le
plancher de bruit avant de conclure sur un ecart d'un ou deux cas.

Variables d'environnement :
  JEU           chemin du jeu (defaut /app/evaluation/jeu_reference.jsonl)
  LUCIOLE_API   URL de l'agent (defaut http://localhost:8000)
  LLM_URL       URL du backend LLM pour le juge (defaut celle de l'agent)
  LLM_MODEL     nom du modele pour le juge
  CONSIGNE      fichier de consigne passe en custom_prompt (optionnel)
  LABEL         suffixe des fichiers de sortie
  DEEP          1 pour deep_search=true
"""
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

BASE = os.environ.get("EVAL_DIR", "/app/evaluation")
JEU = os.environ.get("JEU", os.path.join(BASE, "jeu_reference.jsonl"))
API = os.environ.get("LUCIOLE_API", "http://localhost:8000").rstrip("/")
LLM_URL = os.environ.get("LLM_URL", "http://ollama:11434").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
CONSIGNE = os.environ.get("CONSIGNE", "")
LABEL = os.environ.get("LABEL", "reference")
DEEP = os.environ.get("DEEP", "0") == "1"
API_VERSION = os.environ.get("API_VERSION", "v4")  # v4 (/api/rag/query) ou v3 (/api/query2)

SORTIE = os.path.join(BASE, f"campagne_{LABEL}.jsonl")
RAPPORT = os.path.join(BASE, f"rapport_{LABEL}.txt")

# ---------------------------------------------------------------------------
# Detecteur d'esquive : motif ASCII, texte desaccentue avant comparaison.
# Le code source ne contient aucun caractere accentue : c'est le texte
# analyse qui est ramene a l'ASCII, pas le motif qui est enrichi.
# ---------------------------------------------------------------------------
ESQUIVE = re.compile(
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

SOURCE_CITEE = re.compile(
    r"\.pdf|tome[_ ]?\d|\[?source\s*:|"
    r"volet\s+(?:environnement|milieu|paysage)|\bRNT\b|p\.\s*\d+", re.I)


def sans_accents(texte: str) -> str:
    """Ramene le texte a l'ASCII (e accent aigu -> e), apostrophes
    typographiques -> apostrophe simple. Ne modifie jamais le motif."""
    texte = texte.replace("’", "'").replace("‘", "'")
    return "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )


def verdict(texte: str) -> str:
    if not texte:
        return "ESQUIVE"
    plat = sans_accents(texte)
    m = ESQUIVE.search(plat)
    if not m:
        return "repond"
    if m.start() / len(plat) < 0.15 or len(plat) < 700 \
            or not SOURCE_CITEE.search(plat):
        return "ESQUIVE"
    return "concede"


# ---------------------------------------------------------------------------
# Bon tome : un numero de tome ou un mot-cle de volet de la reference
# apparait dans le nom d'un passage soumis au modele.
# ---------------------------------------------------------------------------
TOME = re.compile(r"tome[_ ]?(\d+)", re.I)
VOLET = re.compile(r"\b(rnt|paysag|naturel|acoust|sant|milieu humain|"
                   r"physique)", re.I)


def cles_reference(sources_reference):
    cles = set()
    for s in sources_reference or []:
        s = sans_accents(str(s)).lower()
        for t in TOME.findall(s):
            cles.add(f"tome{t}")
        for v in VOLET.findall(s):
            cles.add(v)
    return cles


def cles_passage(nom):
    nom = sans_accents(str(nom)).lower()
    cles = set()
    for t in TOME.findall(nom):
        cles.add(f"tome{t}")
    for v in VOLET.findall(nom):
        cles.add(v)
    return cles


def bon_tome(sources_reference, passages) -> bool:
    attendu = cles_reference(sources_reference)
    if not attendu:
        return True  # reference vide : pas de tome attendu
    for p in passages or []:
        if attendu & cles_passage(p.get("file_name", "")):
            return True
    return False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def poster(url, charge, timeout=1200):
    req = urllib.request.Request(
        url, data=json.dumps(charge).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def interroger_luciole(question, consigne):
    """V4 : POST /api/rag/query (contrat {answer, verdict, passages, trace}).
    V3 : POST /api/query2 en repli, reponse projetee sur le meme contrat,
    pour rejouer le meme jeu sur les deux versions."""
    if API_VERSION == "v3":
        charge = {"query": question, "deep_search": DEEP, "history": []}
        if consigne:
            charge["custom_prompt"] = consigne
        rep = poster(f"{API}/api/query2", charge)
        return {"answer": rep.get("response", ""), "passages": rep.get("passages", []),
                "trace": rep.get("iterative", {}), "verdict": "", "esquive": None}
    charge = {"question": question, "deep": DEEP, "history": []}
    if consigne:
        charge["custom_prompt"] = consigne
    return poster(f"{API}/api/rag/query", charge)


JUGE_SYSTEM = (
    "Tu compares deux reponses a une meme remarque d'autorite "
    "environnementale. Tu reponds uniquement en JSON valide."
)

JUGE_USER = """Remarque MRAe :
{question}

Reponse de REFERENCE (redigee par le bureau d'etudes) :
{reference}

Reponse CANDIDATE (generee) :
{candidate}

Note la reponse candidate par rapport a la reference, avec exactement
ces cles :
- "faits" : 0 (les faits cles de la reference sont absents), 1 (une
  partie), 2 (l'essentiel est present)
- "sources" : 0 (aucun document commun cite), 1 (une partie), 2 (les
  memes documents ou volets sont cites)
- "invention" : 2 (la candidate n'ajoute aucun fait absent de la
  reference), 1 (ajoute des faits mineurs), 0 (ajoute des faits
  importants non presents dans la reference)
- "commentaire" : une phrase, en francais, qui explique la note la plus
  basse

JSON :"""


def juger(question, reference, candidate):
    if not reference:
        return None
    charge = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": JUGE_SYSTEM},
            {"role": "user", "content": JUGE_USER.format(
                question=question, reference=reference[:6000],
                candidate=candidate[:6000])},
        ],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 16384},
    }
    try:
        data = poster(f"{LLM_URL}/api/chat", charge, timeout=600)
        brut = data["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return {"erreur": str(e)}
    brut = re.sub(r"```(?:json)?", "", brut)
    d, f = brut.find("{"), brut.rfind("}")
    if d == -1 or f <= d:
        return {"erreur": "json illisible", "brut": brut[:300]}
    try:
        note = json.loads(brut[d:f + 1])
    except ValueError:
        return {"erreur": "json invalide", "brut": brut[:300]}
    for cle in ("faits", "sources", "invention"):
        try:
            note[cle] = max(0, min(2, int(note.get(cle, 0))))
        except (TypeError, ValueError):
            note[cle] = 0
    return note


# ---------------------------------------------------------------------------
def main():
    cas = [json.loads(x) for x in open(JEU, encoding="utf-8") if x.strip()]
    consigne = ""
    if CONSIGNE:
        consigne = open(CONSIGNE, encoding="utf-8").read()

    lignes = []
    with open(SORTIE, "w", encoding="utf-8") as out:
        for c in cas:
            debut = time.time()
            try:
                rep = interroger_luciole(c["question"], consigne)
                erreur = ""
            except (urllib.error.URLError, OSError, ValueError) as e:
                rep, erreur = {}, str(e)
            duree = round(time.time() - debut, 1)
            texte = rep.get("answer", "")
            passages = rep.get("passages", [])
            it = rep.get("trace", {}) or {}
            couverture = (it.get("couverture") or {}).get("verdict", "")
            rech_b = bool((it.get("recherche_b") or {}).get("effectuee"))
            ligne = {
                "id": c["id"],
                "projet": c.get("projet", ""),
                "question": c["question"],
                "reponse": texte,
                "reponse_reference": c.get("reponse_reference", ""),
                "sources_reference": c.get("sources_reference", []),
                "passages": [p.get("file_name", "") for p in passages],
                "esquive": verdict(texte),
                "bon_tome": bon_tome(c.get("sources_reference"), passages),
                "couverture": couverture,
                "verdict_api": rep.get("verdict", ""),
                "recherche_b": rech_b,
                "cibles": sum(1 for p in passages if p.get("cible")),
                "juge": juger(c["question"], c.get("reponse_reference", ""),
                              texte) if texte else None,
                "duree_s": duree,
                "erreur": erreur,
            }
            lignes.append(ligne)
            out.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            out.flush()
            j = ligne["juge"] or {}
            print(f"{ligne['id']:<14} {ligne['esquive']:<8} "
                  f"tome={'oui' if ligne['bon_tome'] else 'NON':<3} "
                  f"{couverture:<11} B={'oui' if rech_b else 'non':<3} "
                  f"faits={j.get('faits', '-')} sources={j.get('sources', '-')} "
                  f"invention={j.get('invention', '-')} {duree}s", flush=True)

    n = len(lignes)
    esq = sum(1 for l in lignes if l["esquive"] == "ESQUIVE")
    tome = sum(1 for l in lignes if l["bon_tome"])
    juges = [l["juge"] for l in lignes if l["juge"] and "erreur" not in l["juge"]]

    def moy(cle):
        return round(sum(j[cle] for j in juges) / len(juges), 2) if juges else "-"

    par_projet = {}
    for l in lignes:
        p = par_projet.setdefault(l["projet"] or "?", {"n": 0, "esq": 0, "tome": 0})
        p["n"] += 1
        p["esq"] += l["esquive"] == "ESQUIVE"
        p["tome"] += l["bon_tome"]

    rapport = [
        f"Campagne {LABEL} — {n} cas, deep={DEEP}, consigne={'oui' if consigne else 'non'}",
        f"esquives          : {esq} / {n}",
        f"bon tome          : {tome} / {n}",
        f"couverture        : " + ", ".join(
            f"{v}={sum(1 for l in lignes if l['couverture'] == v)}"
            for v in ("COUVERT", "PARTIEL", "NON_COUVERT")),
        f"recherche B       : {sum(1 for l in lignes if l['recherche_b'])} / {n}",
        f"juge (moyenne /2) : faits={moy('faits')} sources={moy('sources')} "
        f"invention={moy('invention')}   ({len(juges)} cas juges)",
        "",
        "Par projet :",
    ]
    for p, v in par_projet.items():
        rapport.append(f"  {p:<16} esquives {v['esq']}/{v['n']}  bon tome {v['tome']}/{v['n']}")
    rapport += [
        "",
        "Cas a relire en priorite (faits <= 1 ou invention <= 1) :",
    ]
    for l in lignes:
        j = l["juge"] or {}
        if "erreur" in j or j.get("faits", 2) <= 1 or j.get("invention", 2) <= 1:
            rapport.append(f"  {l['id']}: {j.get('commentaire', j.get('erreur', ''))}")
    with open(RAPPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(rapport) + "\n")
    print("\n" + "\n".join(rapport))


if __name__ == "__main__":
    main()
