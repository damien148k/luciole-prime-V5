"""
ChunkCleaner — Suppression des chunks dans Qdrant et OpenSearch.

Toutes les suppressions se font par filtre sur `document_id` (identifiant
stable du document, correspondant au filename utilisé par l'ingestion
comme clé dans les payloads — cf. rag-system/src/ingestion/chunker.py).

Le `file_path` présent dans les payloads est mis à jour séparément lors
des moves (via `update_file_path`), mais n'est jamais utilisé comme
clé de suppression.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .exceptions import CleanupError

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from opensearchpy import OpenSearch

# Les imports qdrant_client et opensearchpy sont différés dans les méthodes
# pour ne pas bloquer les tests unitaires (dépendances optionnelles en dev local).


class ChunkCleaner:
    """
    Gère la suppression et la mise à jour des chunks dans les stores vectoriels.

    Qdrant et OpenSearch sont traités en séquence. Si l'une des suppressions
    échoue, une CleanupError est levée pour que le worker marque le job en erreur.
    """

    def __init__(
        self,
        qdrant: "QdrantClient",
        opensearch: "OpenSearch",
    ) -> None:
        """
        Args:
            qdrant: Client Qdrant initialisé.
            opensearch: Client OpenSearch initialisé.
        """
        self._qdrant = qdrant
        self._opensearch = opensearch

    def delete_document_chunks(self, document_id: str, index_name: str) -> int:
        """
        Supprime tous les chunks d'un document dans Qdrant et OpenSearch.

        La suppression se fait par filtre sur le payload `document_id`,
        indépendamment du chemin du fichier.

        Args:
            document_id: Identifiant du document tel que stocké dans le
                payload Qdrant et le _source OpenSearch (filename, p.ex.
                "Compte-financier-unique-2024.pdf").
            index_name: Nom de la collection Qdrant / index OpenSearch.

        Returns:
            Nombre estimé de points supprimés dans Qdrant.

        Raises:
            CleanupError: Si la suppression échoue dans l'un des stores.
        """
        logger.debug(
            f"Suppression chunks : document_id={document_id}, index={index_name}"
        )

        deleted_count = self._delete_from_qdrant(document_id, index_name)
        os_deleted = self._delete_from_opensearch(document_id, index_name)

        logger.info(
            f"Chunks supprimés : document_id={document_id} | "
            f"Qdrant≈{deleted_count} points, OpenSearch={os_deleted} docs"
        )
        return deleted_count

    def _delete_from_qdrant(self, document_id: str, collection_name: str) -> int:
        """
        Supprime les points Qdrant filtrés par payload document_id.

        Returns:
            Nombre de points supprimés (approximatif — Qdrant retourne
            un statut, pas un compte exact avant v1.8).
        """
        from qdrant_client import models as qdrant_models

        try:
            result = self._qdrant.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            deleted = getattr(result, "deleted", 0) or 0
            logger.debug(
                f"Qdrant delete OK : document_id={document_id}, résultat={result}"
            )
            return deleted

        except Exception as exc:
            msg = f"Qdrant : échec suppression document_id={document_id} : {exc}"
            logger.error(msg)
            raise CleanupError(msg) from exc

    def _delete_from_opensearch(self, document_id: str, index_name: str) -> int:
        """
        Supprime les documents OpenSearch filtrés par document_id.

        Returns:
            Nombre de documents supprimés.
        """
        try:
            resp = self._opensearch.delete_by_query(
                index=index_name,
                body={"query": {"term": {"document_id": document_id}}},
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            logger.debug(
                f"OpenSearch delete OK : document_id={document_id}, deleted={deleted}"
            )
            return deleted

        except Exception as exc:
            msg = f"OpenSearch : échec suppression document_id={document_id} : {exc}"
            logger.error(msg)
            raise CleanupError(msg) from exc

    def update_file_path(
        self,
        document_id: str,
        new_path: str,
        index_name: str,
    ) -> None:
        """
        Met à jour le `file_path` dans les payloads/métadonnées après un move.

        Cette mise à jour est de la donnée d'affichage uniquement :
        elle ne modifie pas l'identité du document (document_id inchangé,
        sauf si le filename change — cas de rename traité par le worker).
        Un échec ici n'est pas bloquant — le move dans le StateStore est
        déjà appliqué.

        Args:
            document_id: Identifiant du document (filename d'origine).
            new_path: Nouveau chemin absolu.
            index_name: Nom de la collection / index.
        """
        self._update_qdrant_path(document_id, new_path, index_name)
        self._update_opensearch_path(document_id, new_path, index_name)

    def _update_qdrant_path(
        self,
        document_id: str,
        new_path: str,
        collection_name: str,
    ) -> None:
        """Met à jour file_path et file_name dans les payloads Qdrant."""
        from qdrant_client import models as qdrant_models

        try:
            self._qdrant.set_payload(
                collection_name=collection_name,
                payload={
                    "file_path": new_path,
                    "file_name": Path(new_path).name,
                },
                points=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.debug(
                f"Qdrant path mis à jour : document_id={document_id} → {new_path}"
            )

        except Exception as exc:
            # Non bloquant : le move est déjà acté dans le StateStore
            logger.warning(
                f"Qdrant : impossible de mettre à jour file_path "
                f"document_id={document_id} : {exc}"
            )

    def _update_opensearch_path(
        self,
        document_id: str,
        new_path: str,
        index_name: str,
    ) -> None:
        """Met à jour file_path et file_name dans les documents OpenSearch."""
        try:
            self._opensearch.update_by_query(
                index=index_name,
                body={
                    "script": {
                        "source": (
                            "ctx._source.file_path = params.fp; "
                            "ctx._source.file_name = params.fn"
                        ),
                        "params": {
                            "fp": new_path,
                            "fn": Path(new_path).name,
                        },
                    },
                    "query": {"term": {"document_id": document_id}},
                },
                refresh=True,
            )
            logger.debug(
                f"OpenSearch path mis à jour : document_id={document_id} → {new_path}"
            )

        except Exception as exc:
            logger.warning(
                f"OpenSearch : impossible de mettre à jour file_path "
                f"document_id={document_id} : {exc}"
            )
