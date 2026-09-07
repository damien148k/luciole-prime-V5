"""Configuration d'instance : une seule source de verite.

Regle V4 : aucun composant ne conserve de copie d'un reglage. Chaque
composant recoit la valeur en argument au moment ou il en a besoin, lue
sur l'objet Settings courant. Modifier settings.yaml puis appeler
`reload()` suffit : le prochain appel voit la nouvelle valeur, sans
propagation vers d'autres fichiers.

Priorite : variable d'environnement (Docker) > settings.yaml > defaut.
Les variables d'environnement reconnues sont celles de V3 (QDRANT_URL,
OPENSEARCH_URL, LLM_URL) pour rester compatible avec les compose
existants.
"""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml
from loguru import logger

DEFAULT_CONFIG_PATH = "config/settings.yaml"

# Valeurs par defaut minimales : tout ce qui est absent du YAML retombe
# ici, et nulle part ailleurs dans le code.
DEFAULTS: Dict[str, Any] = {
    "instance": {"name": "documents"},
    "embedding": {"model": "BAAI/bge-m3", "device": "auto", "batch_size": 32},
    "reranker": {"model": "BAAI/bge-reranker-v2-m3", "device": "auto", "batch_size": 32},
    "retrieval": {
        "bm25_top_k": 40, "dense_top_k": 40, "fusion_top_k": 30,
        "bm25_weight": 0.45, "dense_weight": 0.55, "rrf_k": 60,
        "rerank_top_n": 15, "search_top_k": 100,
    },
    "llm": {
        "provider": "ollama", "model": "qwen2.5:14b-instruct-q4_K_M",
        "base_url": "http://ollama:11434", "api_format": "ollama",
        "temperature": 0, "seed": 42, "max_tokens": 4096,
        "num_ctx": 32768, "timeout": 1800,
    },
    "chunking": {
        "chunk_size": 800, "chunk_overlap": 100, "strategy": "sentence",
        "include_file_context": True, "adaptive": True,
    },
    "chunking_strategies": {},
    "pdf": {
        "enable_ocr": False, "markdown_timeout": 0,
        "markdown_timeout_per_page": 2.0, "ocr_dpi": 300,
        "max_drawings_per_page": 500, "min_yield_ratio": 0.5,
    },
    "excel": {"max_rows_per_chunk": 50, "overlap_rows": 5, "enable_sql_storage": False},
    "qdrant": {"host": "localhost", "port": 6333, "collection_name": "documents"},
    "opensearch": {"host": "localhost", "port": 9200, "index_name": "documents"},
    "watcher": {},
    "query2": {},
}


def _fusionner(base: Dict, surcharge: Dict) -> Dict:
    """Fusion recursive : les cles de `surcharge` l'emportent."""
    resultat = copy.deepcopy(base)
    for cle, valeur in (surcharge or {}).items():
        if isinstance(valeur, dict) and isinstance(resultat.get(cle), dict):
            resultat[cle] = _fusionner(resultat[cle], valeur)
        else:
            resultat[cle] = copy.deepcopy(valeur)
    return resultat


class Settings:
    """Vue en lecture seule sur la configuration fusionnee."""

    def __init__(self, data: Dict[str, Any], path: Optional[Path] = None):
        self._data = data
        self.path = path

    def section(self, nom: str) -> Dict[str, Any]:
        """Copie d'une section (jamais la reference interne)."""
        return copy.deepcopy(self._data.get(nom, {}) or {})

    def get(self, chemin: str, defaut: Any = None) -> Any:
        """Lecture par chemin pointe : settings.get("retrieval.rerank_top_n")."""
        courant: Any = self._data
        for morceau in chemin.split("."):
            if not isinstance(courant, dict) or morceau not in courant:
                return defaut
            courant = courant[morceau]
        return copy.deepcopy(courant)

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # Raccourcis utilises par l'ingestion et le RAG
    @property
    def collection_name(self) -> str:
        return str(self._data["qdrant"]["collection_name"])

    @property
    def opensearch_index(self) -> str:
        return str(self._data["opensearch"]["index_name"])


def _appliquer_environnement(data: Dict[str, Any]) -> Dict[str, Any]:
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        p = urlparse(qdrant_url)
        data["qdrant"]["host"] = p.hostname or data["qdrant"]["host"]
        data["qdrant"]["port"] = p.port or data["qdrant"]["port"]
    opensearch_url = os.environ.get("OPENSEARCH_URL")
    if opensearch_url:
        p = urlparse(opensearch_url)
        data["opensearch"]["host"] = p.hostname or data["opensearch"]["host"]
        data["opensearch"]["port"] = p.port or data["opensearch"]["port"]
    llm_url = os.environ.get("LLM_URL") or os.environ.get("OLLAMA_HOST")
    if llm_url:
        data["llm"]["base_url"] = llm_url
    instance = os.environ.get("INSTANCE_NAME")
    if instance:
        data["instance"]["name"] = instance
        data["qdrant"]["collection_name"] = instance
        data["opensearch"]["index_name"] = instance.lower()
    return data


def charger(config_path: Optional[str] = None) -> Settings:
    """Charge settings.yaml, applique les defauts puis l'environnement."""
    chemin = Path(config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    brut: Dict[str, Any] = {}
    if chemin.exists():
        with open(chemin, "r", encoding="utf-8") as f:
            brut = yaml.safe_load(f) or {}
        if not isinstance(brut, dict):
            raise RuntimeError(f"{chemin} : contenu YAML invalide (attendu un dictionnaire)")
    else:
        logger.warning(f"{chemin} introuvable : configuration par defaut utilisee")
    data = _appliquer_environnement(_fusionner(DEFAULTS, brut))
    return Settings(data, chemin)


_verrou = threading.Lock()
_courant: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Instance partagee, chargee une fois. `reload()` la remplace."""
    global _courant
    with _verrou:
        if _courant is None or (config_path and Path(config_path) != _courant.path):
            _courant = charger(config_path)
        return _courant


def reload(config_path: Optional[str] = None) -> Settings:
    """Relit le fichier. Les composants qui lisent via get_settings()
    voient la nouvelle valeur au prochain appel."""
    global _courant
    with _verrou:
        _courant = charger(config_path or (str(_courant.path) if _courant else None))
        logger.info(f"Configuration rechargee depuis {_courant.path}")
        return _courant
