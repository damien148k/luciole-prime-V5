"""
index_routing — Dérivation automatique de l'index_name à partir du chemin fichier.

Stratégie : l'index_name est le premier sous-dossier sous le watched_path.
Exemples (watched_path = /app/data) :
  /app/data/chavenay/foo.pdf       -> 'chavenay'
  /app/data/client-x/sub/bar.pdf   -> 'client_x'
  /app/data/foo.pdf                -> fallback (racine = pas de projet)

Sanitization identique à `IngestionPipeline._sanitize_index_name` pour garantir
la cohérence entre le watcher (qui crée les jobs) et le pipeline (qui crée les
collections Qdrant / index OpenSearch).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def sanitize_index_name(name: str) -> str:
    """
    Nettoie un nom d'index pour le rendre compatible avec Qdrant et OpenSearch.

    Règles (identiques à IngestionPipeline._sanitize_index_name) :
    - Remplace les espaces par des underscores
    - Garde uniquement [a-zA-Z0-9_-]
    - Limite à 64 caractères
    - Ne commence pas par un chiffre ou tiret (préfixe 'idx_' sinon)
    - Lowercase final (OpenSearch requirement)
    """
    sanitized = name.replace(" ", "_")
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", sanitized)
    sanitized = sanitized[:64]
    if sanitized and (sanitized[0].isdigit() or sanitized[0] == "-"):
        sanitized = "idx_" + sanitized
    return sanitized.lower()


def derive_index_name(
    file_path: str,
    watched_paths: Iterable,
    default_index_name: str,
) -> str:
    """
    Dérive l'index_name d'un fichier en fonction de son chemin.

    Stratégie (dans l'ordre) :
    1. Trouver le `WatchedPath` parent du fichier.
    2. Si le fichier est dans un sous-dossier sous ce watched_path :
       utiliser le premier composant du chemin relatif comme index_name.
    3. Sinon, fallback sur `watched_path.index_name` si défini explicitement.
    4. Sinon, fallback sur `default_index_name` (= 'documents' typiquement).

    Args:
        file_path: Chemin absolu du fichier.
        watched_paths: Itérable de `WatchedPath` (depuis WatcherConfig).
        default_index_name: Fallback ultime.

    Returns:
        Nom d'index sanitisé, prêt à utiliser pour Qdrant + OpenSearch.
    """
    try:
        file_abs = Path(file_path).resolve()
    except (OSError, RuntimeError):
        return default_index_name

    for wp in watched_paths:
        try:
            wp_abs = Path(wp.path).resolve()
            rel = file_abs.relative_to(wp_abs)
        except (ValueError, OSError, RuntimeError):
            continue

        # Si le watched_path a un index_name explicite (mode mono-instance),
        # l'utiliser directement sans dériver depuis les sous-dossiers.
        if wp.index_name:
            return sanitize_index_name(wp.index_name) or default_index_name

        # rel.parts = (sous_dossier_projet, ..., fichier.ext)
        # On veut au moins 2 composants : <projet>/<fichier>
        if len(rel.parts) >= 2:
            sanitized = sanitize_index_name(rel.parts[0])
            if sanitized:
                return sanitized

        # Sinon, fallback global
        return default_index_name

    # Fichier hors de tout watched_path connu
    return default_index_name
