"""
Utilitaires de hachage et de vérification de stabilité des fichiers.

Deux niveaux de hash :
- `quick_hash`   : taille + mtime — filtre rapide sans lecture du contenu
- `content_hash` : SHA-256 du contenu par blocs — décision finale d'indexation

La lecture par blocs permet de traiter des fichiers volumineux (> 100 Mo)
sans saturer la mémoire du conteneur.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from loguru import logger

from .constants import CONTENT_HASH_BLOCK_SIZE


def quick_hash(path: Path) -> str:
    """
    Calcule un hash rapide basé sur la taille et la date de modification.

    Ne lit pas le contenu du fichier. Utilisé comme premier filtre pour
    éviter de calculer le SHA-256 quand rien n'a visiblement changé.

    Args:
        path: Chemin du fichier.

    Returns:
        Hash hexadécimal MD5 (16 caractères).

    Raises:
        OSError: Si le fichier n'est pas accessible.
    """
    stat = path.stat()
    raw = f"{path}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def content_hash(path: Path) -> str:
    """
    Calcule le SHA-256 du contenu d'un fichier, lu par blocs de 8 Mo.

    Lecture en streaming pour éviter de charger les gros fichiers en mémoire.

    Args:
        path: Chemin du fichier.

    Returns:
        Hash SHA-256 hexadécimal (64 caractères).

    Raises:
        OSError: Si le fichier n'est pas lisible.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CONTENT_HASH_BLOCK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def wait_stable(
    path: Path,
    checks: int = 3,
    interval: float = 2.0,
) -> bool:
    """
    Vérifie qu'un fichier a atteint une taille stable (copie terminée).

    Effectue `checks` lectures de la taille à `interval` secondes d'intervalle.
    Retourne True si la taille est identique sur deux lectures consécutives
    et strictement positive.

    Bloquant — à appeler dans un thread (worker), pas dans la boucle asyncio.

    Args:
        path: Chemin du fichier à vérifier.
        checks: Nombre maximum de vérifications.
        interval: Délai entre chaque vérification (secondes).

    Returns:
        True si le fichier est stable, False sinon.
    """
    prev_size: int = -1

    for attempt in range(checks):
        try:
            current_size = path.stat().st_size
        except OSError as exc:
            logger.debug(f"wait_stable : fichier inaccessible ({exc}) — tentative {attempt + 1}/{checks}")
            time.sleep(interval)
            prev_size = -1
            continue

        if current_size > 0 and current_size == prev_size:
            logger.debug(f"Fichier stable : {path} ({current_size} octets)")
            return True

        logger.debug(
            f"wait_stable : taille {current_size} octets "
            f"(précédente : {prev_size}) — tentative {attempt + 1}/{checks}"
        )
        prev_size = current_size

        if attempt < checks - 1:
            time.sleep(interval)

    # Dernière vérification après la dernière pause
    try:
        final_size = path.stat().st_size
        if final_size > 0 and final_size == prev_size:
            logger.debug(f"Fichier stable (vérification finale) : {path}")
            return True
    except OSError:
        pass

    logger.warning(f"Fichier instable après {checks} vérifications : {path}")
    return False


async def wait_stable_async(
    path: Path,
    checks: int = 3,
    interval: float = 2.0,
) -> bool:
    """
    Version asynchrone de `wait_stable`.

    Utilise `asyncio.sleep` au lieu de `time.sleep` pour ne pas bloquer
    la boucle d'événements. À utiliser uniquement depuis du code asyncio.

    Args:
        path: Chemin du fichier à vérifier.
        checks: Nombre maximum de vérifications.
        interval: Délai entre chaque vérification (secondes).

    Returns:
        True si le fichier est stable, False sinon.
    """
    prev_size: int = -1

    for attempt in range(checks):
        try:
            current_size = path.stat().st_size
        except OSError:
            await asyncio.sleep(interval)
            prev_size = -1
            continue

        if current_size > 0 and current_size == prev_size:
            return True

        prev_size = current_size
        if attempt < checks - 1:
            await asyncio.sleep(interval)

    try:
        final_size = path.stat().st_size
        return final_size > 0 and final_size == prev_size
    except OSError:
        return False
