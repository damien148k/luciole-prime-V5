# -*- coding: utf-8 -*-
"""Construit un jeu de reference JSONL a partir d'un memoire en reponse wpd (.docx).

Usage (Python 3, bibliotheque standard uniquement) :
  python construire_jeu.py "<memoire_reponse.docx>" evaluation/jeu_<instance>.jsonl

Structure attendue du docx (memoires wpd, ex. Brissy avril 2024) :
  - chaque remarque MRAe est dans un tableau : ligne 1 = "Remarque N",
    ligne 2 = texte de la remarque (une puce par paragraphe) ;
  - la reponse wpd = paragraphes et tableaux qui suivent, jusqu'au tableau
    "Remarque N+1" ou au titre "Annexes" ;
  - les renvois "Voir etude d'impact sur l'environnement : Tome X ... p. Y"
    (style liste) deviennent sources_reference, completes par les mentions
    "page Y du Tome X" trouvees dans le texte.
Les titres, legendes (Carte/Figure/Avant/Apres) et "-Absence de
recommandation-" sont ignores. Le champ "section" (titres englobants) est
informatif ; campagne_reference.py l'ignore.
"""
import sys, re, json, zipfile, io
import xml.etree.ElementTree as ET
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
src, dst = sys.argv[1], sys.argv[2]
PROJET = sys.argv[3] if len(sys.argv) > 3 else "Brissy"
PREFIX = sys.argv[4] if len(sys.argv) > 4 else "brissy"
root = ET.fromstring(zipfile.ZipFile(src).read("word/document.xml"))
body = root.find(W + "body")

def ptext(p): return "".join(t.text or "" for t in p.iter(W + "t")).replace(" ", " ").replace("’", "'").strip()
def pstyle(p):
    ps = p.find(W + "pPr/" + W + "pStyle")
    return ps.get(W + "val") if ps is not None else ""
def cell_paras(tc): return [ptext(p) for p in tc.findall(W + "p") if ptext(p)]

CAPTION = re.compile(r"^(Carte|Figure|Photo|Illustration|Avant|Après|Apres)\b", re.I)
REMARQUE = re.compile(r"^Remarque\s+(\d+)\s*$")

remarques, current, headings = [], None, {}
def finalize():
    global current
    if current: remarques.append(current)
    current = None

for el in body:
    tag = el.tag.replace(W, "")
    if tag == "tbl":
        rows = el.findall(W + "tr")
        first = cell_paras(rows[0].findall(W + "tc")[0]) if rows else []
        m = REMARQUE.match(first[0]) if first else None
        if m:
            finalize()
            qparas = []
            for row in rows[1:]:
                for tc in row.findall(W + "tc"):
                    qparas += cell_paras(tc)
            # puces : "recommande :" suivi de plusieurs items
            if len(qparas) > 1:
                question = qparas[0] + "\n" + "\n".join("- " + re.sub(r"^[•\-–]\s*", "", q) for q in qparas[1:])
            else:
                question = qparas[0] if qparas else ""
            current = {"num": int(m.group(1)), "question": question,
                       "section": " > ".join(v for k, v in sorted(headings.items()) if v),
                       "blocs": []}
        elif current:
            for row in rows:
                cells = [" / ".join(cell_paras(tc)) for tc in row.findall(W + "tc")]
                cells = [c for c in cells if c]
                if not cells or all(CAPTION.match(c) for c in cells): continue
                current["blocs"].append(("table", " | ".join(cells)))
    elif tag == "p":
        st, t = pstyle(el), ptext(el)
        if not t: continue
        if st.startswith("Titre") and st != "Titretableaux":
            if t == "Annexes": finalize(); break
            lvl = {"Titre1bis": 1, "Titre2bis": 2, "Titre3bis": 3, "Titre3bis2": 3, "Titre4bis": 4}.get(st, 5)
            headings[lvl] = t
            for k in list(headings):
                if k > lvl: headings[k] = ""
            continue
        if st in ("TM1", "Titretableaux", "Lgende") or t == "-Absence de recommandation-" or CAPTION.match(t):
            continue
        if current:
            current["blocs"].append(("liste" if st == "Paragraphedeliste" else "para", t))
finalize()

VOIR = re.compile(r"^Voir (?:l')?[ée]tude d[’']impact[^:]*:\s*(.*)$", re.I)
def sources_de(blocs, texte):
    srcs = []
    i = 0
    while i < len(blocs):
        kind, t = blocs[i]
        m = VOIR.match(t) if kind == "liste" else None
        if m:
            parts = [m.group(1).strip()]
            j = i + 1
            while j < len(blocs) and blocs[j][0] == "liste" and re.match(r"^(Tome\s*\d|pp?\.\s*\d)", blocs[j][1]):
                parts.append(blocs[j][1].strip()); j += 1
            s = " ".join(p for p in parts if p).strip(" :")
            if s: srcs.append(s)
            i = j
        else:
            i += 1
    plat = texte
    tomes_voir = {t for s_ in srcs for t in re.findall(r"Tome\s*(\d)", s_)}
    def couvert(n): return n in tomes_voir
    for m in re.finditer(r"(?:pages?|p\.)\s*(\d+)(?:\s*(?:à|a|-|et)\s*(\d+))?[^.;]{0,50}?du Tome\s*(\d)", plat, re.I):
        pg = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        if not couvert(m.group(3)): srcs.append(f"Tome {m.group(3)} p. {pg}")
    for m in re.finditer(r"Tome\s*(\d)[^.;]{0,90}?(?:page|p\.)\s*(\d+)", plat, re.I):
        if not couvert(m.group(1)): srcs.append(f"Tome {m.group(1)} p. {m.group(2)}")
    for m in re.finditer(r"Tome\s*(\d)", plat):
        if not any(re.search(rf"Tome\s*{m.group(1)}\b", s) for s in srcs):
            srcs.append(f"Tome {m.group(1)}")
    out = []
    for s in srcs:
        if s not in out: out.append(s)
    return out

with io.open(dst, "w", encoding="utf-8") as f:
    for r in remarques:
        lignes = []
        for kind, t in r["blocs"]:
            lignes.append(("- " + t) if kind == "liste" else t)
        texte = "\n".join(lignes)
        rec = {"id": f"{PREFIX}-{r['num']:02d}", "projet": PROJET,
               "section": r["section"], "question": r["question"],
               "reponse_reference": texte,
               "sources_reference": sources_de(r["blocs"], texte)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{rec['id']}  q={len(rec['question']):4d}c  rep={len(texte):5d}c  src={rec['sources_reference']}")
print(f"\n{len(remarques)} cas ecrits dans {dst}")
