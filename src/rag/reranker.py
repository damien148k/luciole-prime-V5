"""Reranker cross-encoder local, sans etat de configuration.

Correction V4 (defaut n°5 de l'analyse V3) : le reranker ne connait plus
de `top_n`. L'appelant lit `retrieval.rerank_top_n` sur Settings et le
passe a chaque appel. Une seule definition, aucune copie.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.utils.device import resolve_device

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _chemin_local(model_name: str) -> Optional[str]:
    """Modele dans le cache HuggingFace local (mode offline)."""
    dossier = f"models--{model_name.replace('/', '--')}"
    racines = [
        Path(os.environ.get("HF_HOME", "/app/models/huggingface")) / "hub",
        Path(os.environ.get("HF_HOME", "/app/models/huggingface")),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for racine in racines:
        p = racine / dossier
        if p.exists():
            snaps = p / "snapshots"
            if snaps.exists() and any(snaps.iterdir()):
                return str(next(snaps.iterdir()))
            if (p / "config.json").exists():
                return str(p)
        plat = racine / model_name.split("/")[-1]
        if (plat / "config.json").exists():
            return str(plat)
    return None


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "auto", batch_size: int = 32, model=None):
        env_device = os.environ.get("RERANKER_DEVICE", "").strip().lower()
        self.device = resolve_device(env_device or device)
        self.batch_size = batch_size if self.device == "cuda" else min(batch_size, 8)
        if model is not None:  # doublure de test
            self.model = model
            return
        from sentence_transformers import CrossEncoder
        local = _chemin_local(model_name)
        source = local or model_name
        logger.info(f"Chargement reranker {source} sur {self.device}")
        try:
            self.model = CrossEncoder(source, device=self.device,
                                      model_kwargs={"use_safetensors": True})
        except Exception as e:
            raise RuntimeError(
                f"Reranker {model_name} introuvable dans le cache local ({e})."
            ) from e

    def rerank(self, query: str, results: List[Dict], top_n: int) -> List[Dict]:
        """Note chaque passage contre `query`, renvoie les `top_n` meilleurs.

        `top_n` est obligatoire : il vient de Settings a chaque appel.
        """
        if not results:
            return []
        if top_n is None or top_n <= 0:
            raise ValueError("rerank(): top_n doit etre un entier positif issu de la configuration")
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=self.batch_size)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        classement = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"Rerank : {len(results)} candidats -> top {min(top_n, len(classement))}")
        return classement[:top_n]
