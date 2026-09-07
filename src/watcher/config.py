"""
Chargement de la configuration du watcher depuis settings.yaml.

La configuration du watcher est lue depuis la section `watcher:` du fichier
settings.yaml existant. Si la section est absente, des valeurs par défaut
raisonnables sont utilisées.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .models import WatcherConfig


# Chemin par défaut du fichier de configuration
_DEFAULT_CONFIG_PATH = "config/settings.yaml"


def load_watcher_config(config_path: str | None = None) -> WatcherConfig:
    """
    Charge la configuration du watcher depuis settings.yaml.

    La section `watcher:` est optionnelle ; si elle est absente,
    des valeurs par défaut sont utilisées.

    Args:
        config_path: Chemin vers settings.yaml. Si None, utilise
                     la variable d'environnement CONFIG_PATH ou le
                     chemin par défaut 'config/settings.yaml'.

    Returns:
        Instance de WatcherConfig validée par Pydantic.
    """
    resolved_path = _resolve_config_path(config_path)

    raw_watcher: dict[str, Any] = {}
    if resolved_path.exists():
        try:
            with open(resolved_path, encoding="utf-8") as fh:
                full_config: dict[str, Any] = yaml.safe_load(fh) or {}
            raw_watcher = full_config.get("watcher", {}) or {}
            logger.debug(f"Configuration watcher chargée depuis : {resolved_path}")
        except Exception as exc:
            logger.warning(
                f"Impossible de lire {resolved_path} : {exc}. "
                "Utilisation des valeurs par défaut."
            )
    else:
        logger.warning(
            f"Fichier de configuration introuvable : {resolved_path}. "
            "Utilisation des valeurs par défaut."
        )

    # Priorité aux variables d'environnement sur la config YAML
    raw_watcher = _apply_env_overrides(raw_watcher)

    # Normaliser la liste watched_paths
    if "watched_paths" in raw_watcher:
        raw_watcher["watched_paths"] = _normalize_watched_paths(
            raw_watcher["watched_paths"]
        )

    # Règle '1 instance = 1 index' OU multi-index :
    # - Si INSTANCE_NAME est défini ET MULTI_INDEX_MODE=false → surveillance
    #   uniquement sur /app/data/${INSTANCE_NAME}/ (comportement historique).
    # - Si MULTI_INDEX_MODE=true → surveillance sur /app/data/ entier ;
    #   l'index_name est dérivé du premier sous-dossier (via index_routing).
    instance_name = os.environ.get("INSTANCE_NAME", "").strip().lower() or None
    multi_index_mode = os.environ.get("MULTI_INDEX_MODE", "false").lower() == "true"
    if instance_name and not multi_index_mode:
        instance_path = f"/app/data/{instance_name}"
        raw_watcher["watched_paths"] = [{
            "path": instance_path,
            "recursive": True,
            "index_name": instance_name,
        }]
        raw_watcher["default_index_name"] = instance_name
        logger.info(
            f"🎯 Mode mono-instance : watcher contraint à '{instance_path}' → index '{instance_name}'"
        )
    elif instance_name and multi_index_mode:
        instance_path = "/app/data"
        raw_watcher["watched_paths"] = [{
            "path": instance_path,
            "recursive": True,
            "index_name": None,
        }]
        raw_watcher["default_index_name"] = instance_name
        logger.info(
            f"🎯 Mode multi-index : watcher surveille '{instance_path}' — index déduit du sous-dossier"
        )

    config = WatcherConfig(**raw_watcher)
    _validate_paths(config)
    return config


def _resolve_config_path(config_path: str | None) -> Path:
    """Détermine le chemin effectif du fichier de configuration."""
    if config_path:
        return Path(config_path)
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return Path(_DEFAULT_CONFIG_PATH)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Surcharge les valeurs YAML avec les variables d'environnement
    préfixées par WATCHER_.
    """
    env_map = {
        "WATCHER_ENABLED":             ("enabled", lambda v: v.lower() == "true"),
        "WATCHER_DB_PATH":             ("db_path", str),
        "WATCHER_POLLING_INTERVAL":    ("polling_interval", float),
        "WATCHER_DEBOUNCE_SECONDS":    ("debounce_seconds", float),
        "WATCHER_MAX_FILE_SIZE_MB":    ("max_file_size_mb", int),
        "WATCHER_RECONCILE_INTERVAL":  ("reconcile_interval", int),
        "WATCHER_RETRY_MAX_ATTEMPTS":  ("retry_max_attempts", int),
        "WATCHER_WORKER_THREADS":      ("max_worker_threads", int),
        "WATCHER_PORT":                ("api_port", int),
    }
    result = dict(raw)
    for env_key, (field_name, cast) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            try:
                result[field_name] = cast(value)
                logger.debug(f"Override env : {env_key}={value} → {field_name}")
            except (ValueError, TypeError) as exc:
                logger.warning(f"Variable d'environnement ignorée {env_key}={value!r} : {exc}")
    return result


def _normalize_watched_paths(raw_paths: Any) -> list[dict[str, Any]]:
    """
    Normalise la liste des chemins surveillés depuis YAML.
    Accepte :
      - une liste de chaînes : ["/app/data/docs"]
      - une liste de dicts  : [{path: "/app/data/docs", recursive: true}]
    """
    if not isinstance(raw_paths, list):
        return []

    normalized = []
    for item in raw_paths:
        if isinstance(item, str):
            normalized.append({"path": item})
        elif isinstance(item, dict):
            normalized.append(item)
        else:
            logger.warning(f"Entrée watched_paths ignorée (type inattendu) : {item!r}")
    return normalized


def _validate_paths(config: WatcherConfig) -> None:
    """
    Vérifie que les chemins surveillés existent sur le filesystem.
    Logue un avertissement pour chaque chemin absent (ne bloque pas).
    """
    for wp in config.watched_paths:
        p = Path(wp.path)
        if not p.exists():
            logger.warning(
                f"Chemin surveillé introuvable au démarrage : {wp.path}. "
                "Le watcher attendra qu'il apparaisse."
            )
        elif not p.is_dir():
            logger.warning(
                f"Chemin surveillé n'est pas un répertoire : {wp.path}. "
                "Il sera ignoré."
            )
