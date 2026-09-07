# -*- coding: utf-8 -*-
"""Genere un document HTML de relecture humaine d'une campagne de reference.

Usage :
  python relecture.py <dossier_evaluation> <label1> [label2] [-o sortie.html]

Pour chaque cas : remarque MRAe, reponse Luciole (label1) et reponse wpd cote
a cote, indicateurs (esquive, couverture, recherche B, bon tome, notes du
juge), passages soumis au modele. Si un second label est donne, ajoute la
comparaison des deux passages (reponses identiques ou non, ecarts de notes).
Fichier local uniquement : il contient des donnees de projet.
"""
import html
import io
import json
import os
import sys
from collections import Counter


def charger(dossier, label):
    p = os.path.join(dossier, f"campagne_{label}.jsonl")
    return {d["id"]: d for d in (json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip())}


def rapport(dossier, label):
    p = os.path.join(dossier, f"rapport_{label}.txt")
    return io.open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def esc(t):
    return html.escape(str(t or "")).replace("\n", "<br>")


def badge(txt, cls):
    return f'<span class="b {cls}">{esc(txt)}</span>'


def main():
    args = [a for a in sys.argv[1:]]
    out = None
    if "-o" in args:
        i = args.index("-o"); out = args[i + 1]; del args[i:i + 2]
    dossier, labels = args[0], args[1:]
    if not labels:
        sys.exit("label manquant")
    out = out or os.path.join(dossier, f"relecture_{'_'.join(labels)}.html")
    p1 = charger(dossier, labels[0])
    p2 = charger(dossier, labels[1]) if len(labels) > 1 else None

    parts = [f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Relecture campagne {esc(' / '.join(labels))}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;font-size:14px;margin:24px;color:#222;max-width:1500px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:36px;border-top:2px solid #ccc;padding-top:14px}}
pre{{background:#f6f6f6;padding:10px;white-space:pre-wrap;font-size:13px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.col{{border:1px solid #ddd;padding:10px;border-radius:4px;line-height:1.45}}
.col h4{{margin:0 0 8px 0;font-size:13px;text-transform:uppercase;color:#555}}
.q{{background:#eef3fa;padding:10px;border-left:4px solid #4a6fa5;margin:8px 0 12px 0}}
.b{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px;background:#e5e5e5}}
.ok{{background:#d8f0d8}} .warn{{background:#fff0c2}} .bad{{background:#f8d0d0}}
.juge{{margin:10px 0;padding:8px;background:#fbf7ee;border-left:4px solid #c9a227}}
.small{{color:#666;font-size:12px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
</style></head><body>
<h1>Relecture campagne {esc(' / '.join(labels))}</h1>
<p class="small">Document local de relecture humaine. Le juge LLM ne voit pas le corpus : une note « faits » ou « invention » basse
signifie un écart avec la réponse wpd, pas nécessairement une erreur de Luciole (wpd peut avoir utilisé des informations hors dossier,
Luciole peut avoir cité des éléments exacts du dossier que wpd n'a pas repris).</p>
<pre>{esc(rapport(dossier, labels[0]))}</pre>"""]

    if p2:
        parts.append(f"<pre>{esc(rapport(dossier, labels[1]))}</pre>")
        ident = sum(1 for k in p1 if k in p2 and p1[k]["reponse"] == p2[k]["reponse"])
        parts.append(f"<h2>Reproductibilité {esc(labels[0])} → {esc(labels[1])}</h2>"
                     f"<p>Réponses identiques au caractère près : <b>{ident} / {len(p1)}</b> "
                     f"(seuil README : 18 / 20).</p><table><tr><th>cas</th><th>identique</th><th>esquive</th>"
                     f"<th>couverture</th><th>recherche B</th><th>faits</th><th>sources</th><th>invention</th><th>durée</th></tr>")
        for k in p1:
            a, b = p1[k], p2.get(k)
            if not b:
                continue
            ja, jb = a.get("juge") or {}, b.get("juge") or {}
            def c(x, y):
                x, y = str(x), str(y)
                return esc(x) if x == y else f'<b class="warn">{esc(x)} → {esc(y)}</b>'
            same = a["reponse"] == b["reponse"]
            parts.append(f"<tr><td>{k}</td><td>{'oui' if same else '<b>NON</b>'}</td>"
                         f"<td>{c(a['esquive'], b['esquive'])}</td><td>{c(a['couverture'], b['couverture'])}</td>"
                         f"<td>{c(a['recherche_b'], b['recherche_b'])}</td><td>{c(ja.get('faits'), jb.get('faits'))}</td>"
                         f"<td>{c(ja.get('sources'), jb.get('sources'))}</td><td>{c(ja.get('invention'), jb.get('invention'))}</td>"
                         f"<td>{a['duree_s']}s / {b['duree_s']}s</td></tr>")
        parts.append("</table>")

    parts.append("<h2>Cas</h2>")
    for k, d in p1.items():
        j = d.get("juge") or {}
        esq = d["esquive"]
        flags = [badge(esq, "bad" if esq == "ESQUIVE" else ("warn" if esq == "concede" else "ok")),
                 badge("bon tome" if d["bon_tome"] else "TOME ABSENT", "ok" if d["bon_tome"] else "bad"),
                 badge(d["couverture"] or "?", "ok" if d["couverture"] == "COUVERT" else "warn"),
                 badge("recherche B" if d["recherche_b"] else "pas de recherche B", ""),
                 badge(f"faits {j.get('faits', '-')}/2", "ok" if j.get("faits") == 2 else ("warn" if j.get("faits") == 1 else "bad")),
                 badge(f"sources {j.get('sources', '-')}/2", "ok" if j.get("sources") == 2 else ("warn" if j.get("sources") == 1 else "bad")),
                 badge(f"invention {j.get('invention', '-')}/2", "ok" if j.get("invention") == 2 else ("warn" if j.get("invention") == 1 else "bad")),
                 badge(f"{d['duree_s']} s", "")]
        d2 = p2.get(k) if p2 else None
        juge2, col2, ncol = "", "", 2
        if d2 and d2["reponse"] != d["reponse"]:
            j2 = d2.get("juge") or {}
            juge2 = ('<div class="juge"><b>Juge (' + esc(labels[1]) + ') :</b> '
                     + esc(j2.get("commentaire", j2.get("erreur", "non jugé"))) + '</div>')
            col2 = ('<div class="col"><h4>Réponse Luciole (' + esc(labels[1]) + ')</h4>'
                    + (esc(d2["reponse"]) or "<i>vide</i>") + '</div>')
            ncol = 3
        passages = Counter(d["passages"])
        pl = ", ".join(f"{n} ×{c}" if c > 1 else n for n, c in passages.most_common())
        parts.append(f"""<h2 id="{k}">{esc(k)}</h2>
<div>{''.join(flags)}</div>
<div class="q"><b>Remarque MRAe :</b><br>{esc(d['question'])}</div>
<div class="juge"><b>Juge ({esc(labels[0])}) :</b> {esc(j.get('commentaire', j.get('erreur', 'non jugé')))}</div>
{juge2}<div class="cols" style="grid-template-columns:repeat({ncol},1fr)">
<div class="col"><h4>Réponse Luciole ({esc(labels[0])})</h4>{esc(d['reponse']) or '<i>vide</i>'}</div>
{col2}<div class="col"><h4>Réponse wpd (référence)</h4>{esc(d['reponse_reference'])}
<p class="small">Sources wpd : {esc(', '.join(d.get('sources_reference') or []) or 'aucune')}</p></div>
</div>
<p class="small">Passages soumis au modèle : {esc(pl)}</p>""")
        if d.get("erreur"):
            parts.append(f'<p class="bad">Erreur : {esc(d["erreur"])}</p>')

    parts.append("</body></html>")
    io.open(out, "w", encoding="utf-8").write("\n".join(parts))
    print(f"{out} ({len(p1)} cas)")


if __name__ == "__main__":
    main()
