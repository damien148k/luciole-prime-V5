"""
Embedder — Generate dense embeddings
Mode OFFLINE : charge les modèles depuis le cache local
V3 : auto-device via resolve_device(), support BGE-M3 + E5
"""

import os
from typing import List, Optional
from pathlib import Path
from loguru import logger

try:
    # Import différé possible : les tests avec embedder factice n'ont pas
    # besoin de sentence-transformers ni de PyTorch.
    from sentence_transformers import SentenceTransformer
except ImportError as _e:  # pragma: no cover
    # Cause conservée : un ImportError ici signifie presque toujours un
    # paquet PRESENT mais dont l'import échoue (ex: torch incompatible
    # avec les CUDA runtime libs du conteneur), pas un paquet absent.
    # Sans ce message d'origine, l'erreur relevée plus bas ("pip install
    # sentence-transformers") est trompeuse quand le paquet est en fait
    # installé — vécu le 7 septembre 2026 : pip avait remplacé le torch
    # CUDA 12.4 du Dockerfile par une version incompatible en résolvant
    # les dépendances de sentence-transformers, cassant l'import sans
    # rapport avec une absence du paquet lui-même.
    SentenceTransformer = None
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

from .chunker import Chunk
from ..utils.device import resolve_device

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Embedder:
    """
    Generate embeddings using HuggingFace sentence-transformers.
    Default model v3 : BAAI/bge-m3
    Mode OFFLINE supporté.
    """

    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """Cherche le modèle dans les caches HuggingFace locaux."""
        model_folder = f"models--{model_name.replace('/', '--')}"

        cache_paths = [
            Path(os.environ.get("HF_HOME", "/app/models/huggingface")) / "hub",
            Path(os.environ.get("HF_HOME", "/app/models/huggingface")),
            Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/app/models/sentence_transformers")),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

        for cache_path in cache_paths:
            model_path = cache_path / model_folder
            logger.debug(f"Recherche du modèle dans: {model_path}")
            if model_path.exists():
                # Format standard HF : snapshots/<hash>/
                snapshots_dir = model_path / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        snapshot_path = snapshots[0]
                        logger.info(f"Modèle trouvé (format snapshots): {snapshot_path}")
                        return str(snapshot_path)
                # Format flat : fichiers directement dans models--<org>--<name>/
                if (model_path / "config.json").exists():
                    logger.info(f"Modèle trouvé (format flat): {model_path}")
                    return str(model_path)
            # Format flat sans préfixe models-- : cherche directement le dossier du modèle
            flat_path = cache_path / model_name.split("/")[-1]
            if flat_path.exists() and (flat_path / "config.json").exists():
                logger.info(f"Modèle trouvé (format flat sans préfixe): {flat_path}")
                return str(flat_path)

        logger.warning(f"Modèle {model_name} non trouvé dans les caches: {cache_paths}")
        return None

    # Budget de caracteres traites simultanement dans un lot d'encodage.
    # Correspond au regime nominal observe : 32 fragments d'environ mille
    # caracteres passent sans difficulte sur une carte de douze gigaoctets.
    BUDGET_CARACTERES_PAR_LOT = 32000

    # Au-dela, un fragment temoigne d'un decoupage defaillant en amont et
    # merite une trace explicite dans le journal.
    SEUIL_ALERTE_CARACTERES = 4000

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size if self.device == "cuda" else min(batch_size, 8)

        logger.info(f"Loading embedding model: {model_name} on {self.device} (batch_size={self.batch_size})")

        local_path = self._get_local_model_path(model_name)

        # Ne pas forcer use_safetensors : bge-m3 utilise pytorch_model.bin
        model_kwargs = {}

        if local_path:
            logger.info(f"Chargement en mode offline depuis: {local_path}")
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers indisponible : le paquet est peut-etre "
                    "installe mais son import a echoue (cause d'origine ci-dessous) "
                    "-- verifier en priorite une incompatibilite de version de "
                    "torch, plutot que supposer le paquet absent. "
                    f"Erreur d'origine : {_IMPORT_ERROR!r}"
                )
            self.model = SentenceTransformer(local_path, device=self.device, model_kwargs=model_kwargs)
        else:
            hf_cache = Path(os.environ.get("HF_HOME", "/app/models/huggingface")) / "hub"
            logger.info(f"Chargement avec local_files_only=True (cache: {hf_cache})")
            try:
                self.model = SentenceTransformer(
                    model_name,
                    device=self.device,
                    cache_folder=str(hf_cache),
                    model_kwargs=model_kwargs,
                )
            except Exception as e:
                logger.error(f"Impossible de charger le modèle en mode offline: {e}")
                raise RuntimeError(
                    f"Modèle {model_name} non trouvé dans le cache local ({hf_cache}). "
                    "Assurez-vous que le volume huggingface_cache est monté correctement."
                )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def _needs_prefix(self) -> bool:
        """Vérifie si le modèle nécessite un préfixe query:/passage: (E5)."""
        return "e5" in self.model_name.lower()

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        if self._needs_prefix():
            prefix = "query: " if is_query else "passage: "
            text = prefix + text

        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def _lot_adapte(self, longueur_max: int) -> int:
        """
        Reduit la taille du lot lorsque les textes sont longs.

        La bibliotheque trie les textes par longueur avant de les grouper : les
        fragments les plus longs se retrouvent donc ensemble dans un meme lot,
        et le remplissage se fait sur le plus long d'entre eux. Le cout memoire
        de l'attention croit avec le carre de la longueur de sequence. Un lot de
        32 sequences proches du maximum du modele demande bien plus que la
        memoire d'une carte de douze gigaoctets, ce qui se traduit non par une
        erreur franche mais par un ralentissement sans fin.

        On raisonne donc a budget de caracteres constant plutot qu'a nombre de
        textes constant.
        """
        if longueur_max <= 0:
            return self.batch_size
        estime = self.BUDGET_CARACTERES_PAR_LOT // longueur_max
        return max(1, min(self.batch_size, estime))

    def embed_chunks(self, chunks: List[Chunk]) -> List[dict]:
        if not chunks:
            return []

        texts = []
        for chunk in chunks:
            text = chunk.text_with_context
            if self._needs_prefix():
                text = "passage: " + text
            texts.append(text)

        longueur_max = max(len(t) for t in texts)
        lot = self._lot_adapte(longueur_max)

        logger.info(
            f"Generating embeddings for {len(texts)} chunks (with file context), "
            f"longueur max {longueur_max} caracteres, lot {lot}"
        )

        if longueur_max > self.SEUIL_ALERTE_CARACTERES:
            logger.warning(
                f"Fragment anormalement long : {longueur_max} caracteres pour une "
                f"cible de decoupage bien inferieure. Le cout de l'attention croit "
                f"avec le carre de la longueur : le lot est reduit a {lot} pour eviter "
                f"de saturer la memoire du peripherique. Verifiez la strategie de "
                f"decoupage appliquee a ce document."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=lot,
            show_progress_bar=True,
            convert_to_tensor=False
        )

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "text_with_context": chunk.text_with_context,
                "file_path": chunk.file_path,
                "file_name": chunk.file_name,
                "embedding": embedding.tolist(),
                "metadata": chunk.metadata
            })

        logger.info(f"Generated {len(results)} embeddings")
        return results

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query, is_query=True)
