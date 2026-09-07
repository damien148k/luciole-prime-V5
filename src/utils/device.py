# -*- coding: utf-8 -*-
"""
Device Resolver — Auto-détection CUDA / CPU

Fichier partagé par embedder.py, reranker.py et ocr.py.
Quand settings.yaml contient device: "auto", ce module détecte
le matériel disponible et renvoie "cuda" ou "cpu".
"""

from loguru import logger


def resolve_device(setting: str) -> str:
    """Résout 'auto' → 'cuda' ou 'cpu' selon le matériel disponible.

    torch est importé ici et non en tête de module : les tests et les
    outils qui n'ont pas besoin de GPU ne doivent pas charger PyTorch.
    """
    if setting == "auto":
        import torch  # import différé volontaire
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device auto-détecté : {device}")
        return device
    return setting
