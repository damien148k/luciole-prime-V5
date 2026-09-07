"""Projection des passages internes vers le contrat d'API.

Reprend `_reporter_pages` de V3 (regression PR #23 : les PDF portent
page_start/page_end, pas `page`) et ajoute l'etiquette « cible » (defaut
n°6) : un passage remonte par une requete ciblee de la recherche B garde
la requete qui l'a trouve, pour que l'utilisateur comprenne pourquoi il
est la et que ses scores ne se comparent pas aux passages generaux.
"""

from typing import Dict, List


def _reporter_pages(passage: dict, meta: dict) -> None:
    debut = meta.get("page_start")
    fin = meta.get("page_end")
    if debut is not None:
        passage["page_start"] = debut
    if fin is not None:
        passage["page_end"] = fin
    if meta.get("page"):
        passage["page"] = meta["page"]
    elif debut is not None:
        passage["page"] = debut


def projeter(search_results: List[Dict], limite: int = 30, texte_max: int = 1000) -> List[Dict]:
    passages = []
    for chunk in search_results[:limite]:
        p = {
            "text": (chunk.get("text") or chunk.get("content") or "")[:texte_max],
            "file_name": chunk.get("file_name", ""),
            "score": round(chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0))) or 0, 4),
        }
        meta = chunk.get("metadata", {}) or {}
        _reporter_pages(p, meta)
        if meta.get("section"):
            p["section"] = meta["section"]
        if chunk.get("quota_requete"):
            p["cible"] = chunk["quota_requete"]
        passages.append(p)
    return passages
