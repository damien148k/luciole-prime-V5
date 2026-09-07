"""Pipeline d'ingestion V4 : parser -> chunker -> embedder -> Qdrant + OpenSearch.

Repris du fonctionnement valide de V3, avec trois corrections :

1. Identifiants Qdrant deterministes. V3 calculait l'id d'un point avec
   `hash(chunk_id)`, dont la valeur change a chaque redemarrage du
   processus Python (PYTHONHASHSEED aleatoire). Une reingestion depuis un
   autre processus creait donc des doublons dans Qdrant, alors
   qu'OpenSearch (id = chunk_id texte) remplacait correctement : les deux
   index divergeaient. V4 derive l'id d'un UUID v5 du chunk_id : meme
   entree, meme id, quel que soit le processus. Reingerer = remplacer.

2. Plus de tracker SQLite dans le pipeline : le watcher est la seule
   autorite sur l'etat d'indexation (son StateStore). Le pipeline est
   sans etat, il ingere ce qu'on lui donne.

3. Configuration lue sur l'objet Settings (src.config.settings), jamais
   copiee : un reload de configuration est vu a la prochaine ingestion.

La signature `IngestionPipeline(config_path=..., index_name=...,
enable_tracking=...)` et `ingest_file(file_path, skip_if_indexed,
index_name)` sont conservees pour que le watcher V3 fonctionne sans
modification (enable_tracking et skip_if_indexed sont acceptes et
ignores).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config.settings import Settings, charger
from .chunker import Chunk, Chunker
from .embedder import Embedder
from .parsers import DocumentParser

# Champs metier extraits du YAML front matter (MarkdownParser), promus a
# la racine des payloads Qdrant et des documents OpenSearch pour le
# filtrage direct (editor=fortinet, severity=critical, projet=...).
METADATA_FILTERABLE_FIELDS = (
    "client", "editor", "technology", "product", "version",
    "support_type", "severity", "ticket_id", "date",
    "projet", "phase", "thematique", "departement",
)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".svg", ".eps", ".ai", ".psd", ".ico", ".webp",
}

# Espace de noms fixe pour les UUID v5 des points Qdrant. Ne jamais le
# changer : il rend les identifiants reproductibles d'une machine a
# l'autre.
_NAMESPACE_POINTS = uuid.UUID("6f7f2c3a-1b6e-4b8a-9c2d-5e1f0a9b8c7d")


def point_id(chunk_id: str) -> str:
    """Identifiant Qdrant deterministe d'un fragment."""
    return str(uuid.uuid5(_NAMESPACE_POINTS, chunk_id))


def document_hash(file_path: str) -> str:
    """SHA-256 du contenu du fichier (tracabilite par document)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def sanitize_index_name(name: str) -> str:
    """Nom compatible Qdrant et OpenSearch (memes regles que V3)."""
    s = name.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-]", "", s)[:64]
    if s and (s[0].isdigit() or s[0] == "-"):
        s = "idx_" + s
    return s or "documents"


class IngestionPipeline:
    """Parse, decoupe, encode et indexe un fichier."""

    def __init__(
        self,
        config_path: str = None,
        custom_params: dict = None,
        index_name: str = None,
        enable_tracking: bool = False,  # accepte pour compatibilite, ignore
        settings: Optional[Settings] = None,
        qdrant: Optional[QdrantClient] = None,
        opensearch: Optional[OpenSearch] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.settings = settings or charger(config_path)
        cfg = self.settings.as_dict()

        if index_name:
            safe = sanitize_index_name(index_name)
            cfg["qdrant"]["collection_name"] = safe
            cfg["opensearch"]["index_name"] = safe.lower()

        # Surcharges ponctuelles (UI) : appliquees a CETTE instance de
        # pipeline seulement, jamais ecrites dans settings.yaml.
        for cle, cible in (("chunk_size", "chunk_size"), ("overlap", "chunk_overlap"),
                           ("chunk_overlap", "chunk_overlap")):
            if custom_params and cle in custom_params:
                cfg["chunking"][cible] = custom_params[cle]
        if custom_params and "batch_size" in custom_params:
            cfg["embedding"]["batch_size"] = custom_params["batch_size"]

        self.config = cfg
        self.parser = DocumentParser(pdf_config=cfg.get("pdf", {}),
                                     excel_config=cfg.get("excel", {}))
        ch = cfg["chunking"]
        self.chunker = Chunker(
            chunk_size=ch["chunk_size"],
            chunk_overlap=ch["chunk_overlap"],
            strategy=ch["strategy"],
            include_file_context=ch.get("include_file_context", True),
            adaptive=ch.get("adaptive", False),
            chunking_strategies=cfg.get("chunking_strategies", {}),
        )
        self.embedder = embedder or Embedder(
            model_name=cfg["embedding"]["model"],
            device=cfg["embedding"]["device"],
            batch_size=cfg["embedding"]["batch_size"],
        )
        self.qdrant = qdrant or self._build_qdrant()
        self.opensearch = opensearch or self._build_opensearch()
        self._index_swap_lock = threading.RLock()
        self._init_qdrant()
        self._init_opensearch()
        logger.info(
            f"Ingestion pipeline pret (qdrant={cfg['qdrant']['collection_name']}, "
            f"opensearch={cfg['opensearch']['index_name']})"
        )

    # ------------------------------------------------------------------
    # Clients et structures
    # ------------------------------------------------------------------
    def _build_qdrant(self) -> QdrantClient:
        url = os.environ.get("QDRANT_URL")
        if url:
            return QdrantClient(url=url)
        return QdrantClient(host=self.config["qdrant"]["host"], port=self.config["qdrant"]["port"])

    def _build_opensearch(self) -> OpenSearch:
        return OpenSearch(
            hosts=[{"host": self.config["opensearch"]["host"],
                    "port": self.config["opensearch"]["port"]}],
            http_compress=True, use_ssl=False, verify_certs=False,
        )

    def _init_qdrant(self):
        name = self.config["qdrant"]["collection_name"]
        existantes = {c.name for c in self.qdrant.get_collections().collections}
        if name in existantes:
            return
        self.qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.embedder.embedding_dim, distance=Distance.COSINE),
        )
        for field in METADATA_FILTERABLE_FIELDS:
            if field == "date":
                continue
            try:
                self.qdrant.create_payload_index(collection_name=name, field_name=field, field_schema="keyword")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"index payload {field} : {e}")
        # document_id indexe pour les suppressions par filtre du watcher
        try:
            self.qdrant.create_payload_index(collection_name=name, field_name="document_id", field_schema="keyword")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"index payload document_id : {e}")
        logger.info(f"Collection Qdrant creee : {name}")

    def _init_opensearch(self):
        name = self.config["opensearch"]["index_name"]
        if self.opensearch.indices.exists(index=name):
            return
        body = {
            "settings": {
                "index": {"number_of_shards": 1, "number_of_replicas": 0},
                "analysis": {"analyzer": {
                    "french_analyzer": {"type": "french"},
                    "path_analyzer": {"type": "custom", "tokenizer": "path_hierarchy"},
                    "filename_analyzer": {"type": "custom", "tokenizer": "standard",
                                          "filter": ["lowercase", "asciifolding"]},
                }},
            },
            "mappings": {"properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_hash": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "french_analyzer"},
                "text_with_context": {"type": "text", "analyzer": "french_analyzer"},
                "file_path": {"type": "text", "analyzer": "path_analyzer",
                              "fields": {"keyword": {"type": "keyword"}}},
                "file_name": {"type": "text", "analyzer": "filename_analyzer",
                              "fields": {"keyword": {"type": "keyword"}}},
                "metadata": {"type": "object"},
                **{f: {"type": "keyword"} for f in METADATA_FILTERABLE_FIELDS if f != "date"},
                "date": {"type": "date",
                         "format": "yyyy-MM-dd||yyyy-MM-dd'T'HH:mm:ss||strict_date_optional_time"},
            }},
        }
        self.opensearch.indices.create(index=name, body=body)
        logger.info(f"Index OpenSearch cree : {name}")

    @contextmanager
    def _override_index(self, index_name: Optional[str]):
        """Bascule temporaire sur un autre index (structures creees si absentes)."""
        if not index_name:
            yield
            return
        safe = sanitize_index_name(index_name)
        with self._index_swap_lock:
            old_q = self.config["qdrant"]["collection_name"]
            old_o = self.config["opensearch"]["index_name"]
            deja = old_q == safe and old_o == safe.lower()
            try:
                if not deja:
                    self.config["qdrant"]["collection_name"] = safe
                    self.config["opensearch"]["index_name"] = safe.lower()
                self._init_qdrant()
                self._init_opensearch()
                yield
            finally:
                if not deja:
                    self.config["qdrant"]["collection_name"] = old_q
                    self.config["opensearch"]["index_name"] = old_o

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    @staticmethod
    def _is_image_file(file_path: str) -> bool:
        return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS

    def _image_metadata_chunk(self, file_path: str) -> Chunk:
        path = Path(file_path)
        mots = re.sub(r"^\d{8}_?", "", path.stem).replace("_", " ").replace("-", " ")
        dossier = "/".join(path.parent.parts[-4:])
        text = (f"Fichier image: {path.name}\nType: {path.suffix.upper().lstrip('.')}\n"
                f"Nom: {mots}\nDossier: {dossier}\nMots-cles: {mots}")
        return Chunk(
            text=text, text_with_context=f"[Fichier: {path.name}] {text}",
            chunk_id=f"{path.name}_metadata", document_id=path.name,
            file_path=str(file_path), file_name=path.name, start_char=0, end_char=len(text),
            metadata={"type": "image", "file_name": path.name, "file_path": str(file_path),
                      "chunk_index": 0, "is_metadata_only": True},
        )

    def refresh_settings(self) -> bool:
        """Relit settings.yaml s'il a change depuis la construction et
        reconstruit parser et chunker (categorie « au prochain fichier »).
        L'embedder (GPU) et les clients ne sont pas touches. Retourne True
        si une relecture a eu lieu."""
        chemin = self.settings.path
        if not chemin or not Path(chemin).exists():
            return False
        mtime = Path(chemin).stat().st_mtime
        if mtime == getattr(self, "_settings_mtime", None):
            return False
        if getattr(self, "_settings_mtime", None) is not None:
            nouveau = charger(str(chemin)).as_dict()
            for section in ("chunking", "chunking_strategies", "pdf", "excel"):
                self.config[section] = nouveau.get(section, {})
            self.parser = DocumentParser(pdf_config=self.config.get("pdf", {}),
                                        excel_config=self.config.get("excel", {}))
            ch = self.config["chunking"]
            self.chunker = Chunker(
                chunk_size=ch["chunk_size"], chunk_overlap=ch["chunk_overlap"],
                strategy=ch["strategy"], include_file_context=ch.get("include_file_context", True),
                adaptive=ch.get("adaptive", False),
                chunking_strategies=self.config.get("chunking_strategies", {}),
            )
            logger.info("Configuration d'ingestion relue (chunking/pdf) avant ce fichier")
            self._settings_mtime = mtime
            return True
        self._settings_mtime = mtime
        return False

    def ingest_file(self, file_path: str, skip_if_indexed: bool = False,
                    index_name: Optional[str] = None) -> Dict:
        """Ingere un fichier. Reingerer le meme fichier remplace ses points."""
        self.refresh_settings()
        logger.info(f"Ingestion : {file_path}")
        with self._override_index(index_name):
            if self._is_image_file(file_path):
                chunks = [self._image_metadata_chunk(file_path)]
                doc_id = chunks[0].document_id
            else:
                document = self.parser.parse(file_path)
                chunks = self.chunker.chunk(document)
                doc_id = document["metadata"].get("file_name")
                if not chunks:
                    logger.warning(f"Aucun fragment produit : {file_path}")
                    return {"status": "empty", "file": file_path, "chunks": 0}

            doc_hash = document_hash(file_path)
            for c in chunks:
                c.metadata["document_hash"] = doc_hash

            embedded = self.embedder.embed_chunks(chunks)
            self._index_qdrant(embedded)
            self._index_opensearch(embedded)

            result = {
                "status": "success", "file": file_path, "chunks": len(chunks),
                "document_id": doc_id, "document_hash": doc_hash,
                "index_name": self.config["qdrant"]["collection_name"],
            }
            logger.info(f"Ingestion terminee : {result}")
            return result

    def ingest_directory(self, dir_path: str, recursive: bool = True,
                         include_images: bool = True) -> List[Dict]:
        exts = set(self.parser.get_supported_extensions())
        if include_images:
            exts |= IMAGE_EXTENSIONS
        motif = "**/*" if recursive else "*"
        resultats = []
        for p in sorted(Path(dir_path).glob(motif)):
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    resultats.append(self.ingest_file(str(p)))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Echec ingestion {p} : {e}")
                    resultats.append({"status": "error", "file": str(p), "error": str(e)})
        return resultats

    # ------------------------------------------------------------------
    # Indexation
    # ------------------------------------------------------------------
    def _payload(self, chunk: Dict) -> Dict:
        payload = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_hash": chunk["metadata"].get("document_hash"),
            "text": chunk["text"],
            "text_with_context": chunk["text_with_context"],
            "file_path": chunk["file_path"],
            "file_name": chunk["file_name"],
            "metadata": chunk["metadata"],
        }
        for key in METADATA_FILTERABLE_FIELDS:
            v = chunk["metadata"].get(key)
            if v is not None:
                payload[key] = v
        return payload

    def _index_qdrant(self, embedded: List[Dict], batch_size: int = 100):
        name = self.config["qdrant"]["collection_name"]
        points = [PointStruct(id=point_id(c["chunk_id"]), vector=c["embedding"],
                              payload=self._payload(c)) for c in embedded]
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(collection_name=name, points=points[i:i + batch_size])
        logger.info(f"Qdrant : {len(points)} points upsertes dans {name}")

    def _index_opensearch(self, embedded: List[Dict], batch_size: int = 100):
        name = self.config["opensearch"]["index_name"]
        for i in range(0, len(embedded), batch_size):
            actions = []
            for c in embedded[i:i + batch_size]:
                actions.append({"index": {"_index": name, "_id": c["chunk_id"]}})
                actions.append(self._payload(c))
            if actions:
                rep = self.opensearch.bulk(body=actions)
                if rep.get("errors"):
                    nb = sum(1 for it in rep["items"] if "error" in it.get("index", {}))
                    logger.warning(f"OpenSearch : {nb} erreurs dans le lot {i}")
        self.opensearch.indices.refresh(index=name)
        logger.info(f"OpenSearch : {len(embedded)} documents indexes dans {name}")

    def get_stats(self) -> Dict:
        q = self.config["qdrant"]["collection_name"]
        o = self.config["opensearch"]["index_name"]
        return {
            "qdrant_vectors": self.qdrant.count(collection_name=q).count,
            "opensearch_documents": self.opensearch.count(index=o)["count"],
            "embedding_model": self.config["embedding"]["model"],
            "chunk_size": self.config["chunking"]["chunk_size"],
        }
