"""
BM25 Search - Sparse retrieval using OpenSearch
"""

from typing import List, Dict
from loguru import logger
from opensearchpy import OpenSearch


class BM25Search:
    """
    BM25 sparse search using OpenSearch
    """
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "documents_bm25"):
        """
        Initialize BM25 search
        
        Args:
            host: OpenSearch host
            port: OpenSearch port
            index_name: Index name to search
        """
        self.index_name = index_name
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        logger.info(f"BM25Search initialized: {host}:{port}/{index_name}")
    
    def search(self, query: str, top_k: int = 20, filters: Dict = None) -> List[Dict]:
        """
        Search documents using BM25
        
        Recherche dans: texte, texte avec contexte, chemin du fichier, nom du fichier
        Les pondérations favorisent le contenu tout en valorisant le contexte fichier
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Filtres métier optionnels sur champs keyword
                (ex: {"editor": "fortinet", "severity": ["high", "medium"]}).
                Chaque clé correspond à un champ METADATA_FILTERABLE_FIELDS
                indexé en keyword à la racine du document OpenSearch.
                Une valeur liste applique un OR (terms), une valeur simple
                un match exact (term). Toutes les clés sont combinées en AND.
            
        Returns:
            List of search results with scores
        """
        query_clause = {
            "multi_match": {
                "query": query,
                "fields": [
                    "text^3",              # Contenu principal (priorité haute)
                    "text_with_context^2", # Texte avec contexte fichier
                    "file_name^2",         # Nom du fichier (souvent informatif)
                    "file_path^1.5",       # Chemin (contexte organisationnel)
                    "metadata.title^2",    # Titre du document si disponible
                    "metadata.author"      # Auteur
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }

        filter_clauses = self._build_filter_clauses(filters)
        if filter_clauses:
            final_query = {
                "bool": {
                    "must": [query_clause],
                    "filter": filter_clauses
                }
            }
        else:
            final_query = query_clause

        search_body = {
            "size": top_k,
            "query": final_query,
            "_source": ["chunk_id", "document_id", "text", "file_path", "file_name", "metadata"]
        }
        
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "chunk_id": source["chunk_id"],
                    "document_id": source["document_id"],
                    "text": source["text"],
                    "file_path": source.get("file_path", ""),
                    "file_name": source.get("file_name", ""),
                    "metadata": source.get("metadata", {}),
                    "score": hit["_score"],
                    "search_type": "bm25"
                })
            
            logger.debug(f"BM25 search returned {len(results)} results for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    @staticmethod
    def _build_filter_clauses(filters: Dict = None) -> List[Dict]:
        """
        Traduit un dict de filtres métier simple en clauses de filtre
        OpenSearch (term / terms), combinées en AND via bool.filter.

        Args:
            filters: dict {champ: valeur} ou {champ: [valeurs]}. Les clés
                avec valeur None ou liste vide sont ignorées.

        Returns:
            Liste de clauses prêtes à être placées dans bool.filter
        """
        if not filters:
            return []

        clauses = []
        for field, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [v for v in value if v is not None]
                if not values:
                    continue
                clauses.append({"terms": {field: list(values)}})
            else:
                clauses.append({"term": {field: value}})
        return clauses

    def health_check(self) -> bool:
        """Check if OpenSearch is available"""
        try:
            return self.client.ping()
        except Exception:
            return False

