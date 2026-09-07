"""
Dense Search - Vector similarity search using Qdrant
Compatible avec Qdrant server v1.7.x
"""

from typing import List, Dict
import httpx
from loguru import logger

from src.ingestion.embedder import Embedder


class DenseSearch:
    """
    Dense vector search using Qdrant via HTTP API
    Compatible avec Qdrant v1.7.x (Docker)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents_dense",
        embedder: Embedder = None
    ):
        """
        Initialize dense search
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Collection name to search
            embedder: Embedder instance for query embedding
        """
        self.collection_name = collection_name
        self.base_url = f"http://{host}:{port}"
        self.embedder = embedder
        
        logger.info(f"DenseSearch initialized: {host}:{port}/{collection_name}")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_conditions: Dict = None,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Search documents using dense vectors via HTTP API
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_conditions: Filtre Qdrant natif (format brut de l'API
                Qdrant, ex: {"must": [{"key": "editor", "match": {"value": "fortinet"}}]}).
                Prioritaire si fourni en plus de `filters`.
            filters: Filtres métier simplifiés (ex: {"editor": "fortinet",
                "severity": ["high", "medium"]}), traduits automatiquement
                vers le format Qdrant natif sur les champs payload indexés
                en keyword. Ignoré si `filter_conditions` est aussi fourni.
            
        Returns:
            List of search results with scores
        """
        if self.embedder is None:
            raise ValueError("Embedder not initialized")
        
        # Generate query embedding
        query_vector = self.embedder.embed_query(query)
        
        # Build search request for Qdrant v1.7 HTTP API
        search_request = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }
        
        effective_filter = filter_conditions or self._build_qdrant_filter(filters)
        if effective_filter:
            search_request["filter"] = effective_filter
        
        try:
            # Appel HTTP direct à l'API Qdrant v1.7
            url = f"{self.base_url}/collections/{self.collection_name}/points/search"
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=search_request)
                response.raise_for_status()
                data = response.json()
            
            results = []
            for hit in data.get("result", []):
                payload = hit.get("payload", {})
                results.append({
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "text": payload.get("text"),
                    "file_path": payload.get("file_path", ""),
                    "file_name": payload.get("file_name", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": hit.get("score", 0),
                    "search_type": "dense"
                })
            
            logger.debug(f"Dense search returned {len(results)} results for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []
    
    @staticmethod
    def _build_qdrant_filter(filters: dict = None) -> dict:
        """
        Traduit un dict de filtres metier simple vers le format de
        filtre natif Qdrant, sur les champs payload indexes en keyword
        (voir METADATA_FILTERABLE_FIELDS dans ingestion/pipeline.py).

        Args:
            filters: dict {champ: valeur} ou {champ: [valeurs]}. Une
                valeur liste applique un OR (should), une valeur simple
                un match exact. Toutes les cles sont combinees en AND (must).

        Returns:
            Filtre au format Qdrant ({"must": [...]})  ou None si vide.
        """
        if not filters:
            return None

        must_conditions = []
        for field, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [v for v in value if v is not None]
                if not values:
                    continue
                if len(values) == 1:
                    must_conditions.append({"key": field, "match": {"value": values[0]}})
                else:
                    must_conditions.append({
                        "key": field,
                        "match": {"any": list(values)}
                    })
            else:
                must_conditions.append({"key": field, "match": {"value": value}})

        if not must_conditions:
            return None
        return {"must": must_conditions}

    def health_check(self) -> bool:
        """Check if Qdrant is available"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/collections")
                return response.status_code == 200
        except Exception:
            return False

