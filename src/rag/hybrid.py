"""
Hybrid Search - Combines BM25 and Dense search with RRF fusion
"""

from typing import List, Dict, Tuple
from loguru import logger
from collections import defaultdict


class HybridSearch:
    """
    Hybrid search combining BM25 (sparse) and Dense (vector) search
    Uses Reciprocal Rank Fusion (RRF) for result combination
    """
    
    def __init__(
        self,
        bm25_search,
        dense_search,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        bm25_top_k: int = 50,
        dense_top_k: int = 50
    ):
        """
        Initialize hybrid search
        
        Args:
            bm25_search: BM25Search instance
            dense_search: DenseSearch instance
            bm25_weight: Weight for BM25 results (0-1)
            dense_weight: Weight for dense results (0-1)
            rrf_k: RRF constant (default 60)
            bm25_top_k: Number of results from BM25 search
            dense_top_k: Number of results from Dense search
        """
        self.bm25_search = bm25_search
        self.dense_search = dense_search
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        
        logger.info(f"HybridSearch initialized: bm25_weight={bm25_weight}, dense_weight={dense_weight}, bm25_top_k={bm25_top_k}, dense_top_k={dense_top_k}")
    
    def search(
        self,
        query: str,
        top_k: int = 30,
        filters: Dict = None,
        bm25_top_k: int = None,
        dense_top_k: int = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and dense search

        Args:
            query: Search query
            top_k: Number of final results to return after fusion
            filters: Filtres métier optionnels sur les champs YAML front
                matter propagés à la racine des documents/payloads
                (client, editor, technology, product, version,
                support_type, severity, ticket_id, projet, phase,
                thematique, departement). Format simple :
                {"editor": "fortinet"} ou {"severity": ["high", "medium"]}.
                Appliqué identiquement sur BM25 (OpenSearch) et Dense
                (Qdrant) avant la fusion RRF, donc les deux classements
                ne portent que sur le sous-ensemble filtré.
            bm25_top_k: Surcharge ponctuelle du pool brut BM25 (défaut :
                self.bm25_top_k, issu de settings.yaml). Utilisé par le
                mode recherche approfondie (deep_search) du pipeline
                query2 — la valeur d'instance reste inchangée.
            dense_top_k: Idem pour le pool brut dense.

        Returns:
            List of fused search results
        """
        # Get results from both search methods using configured top_k
        bm25_results = self.bm25_search.search(query, top_k=bm25_top_k or self.bm25_top_k, filters=filters)
        dense_results = self.dense_search.search(query, top_k=dense_top_k or self.dense_top_k, filters=filters)
        
        # Apply RRF fusion
        fused_results = self._rrf_fusion(bm25_results, dense_results, top_k)
        
        filter_info = f", filters={filters}" if filters else ""
        logger.info(f"Hybrid search: BM25={len(bm25_results)}, Dense={len(dense_results)}, Fused={len(fused_results)}{filter_info}")
        return fused_results
    
    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) algorithm
        
        RRF score = sum(1 / (k + rank_i)) for each ranking
        
        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search
            top_k: Number of results to return
            
        Returns:
            Fused and sorted results
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        chunk_data = {}
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = self.bm25_weight * (1 / (self.rrf_k + rank))
            rrf_scores[chunk_id] += rrf_score
            chunk_data[chunk_id] = result
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = self.dense_weight * (1 / (self.rrf_k + rank))
            rrf_scores[chunk_id] += rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        results = []
        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = rrf_score
            result["search_type"] = "hybrid"
            results.append(result)
        
        return results
    
    def search_multi(
        self,
        queries: List[str],
        top_k: int = 30,
        filters: Dict = None,
        bm25_top_k: int = None,
        dense_top_k: int = None
    ) -> List[Dict]:
        """
        Perform hybrid search with multiple query variants.
        Runs search for each query, deduplicates by chunk_id keeping best score.

        Args:
            queries: List of query variants
            top_k: Number of final results to return
            filters: Filtres métier optionnels, appliqués identiquement à
                chaque variante de requête (voir `search` pour le format)
            bm25_top_k / dense_top_k: surcharges ponctuelles des pools
                bruts, propagées à chaque variante (voir `search`).

        Returns:
            List of fused search results (deduplicated, best score per chunk)
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.search(queries[0], top_k=top_k, filters=filters,
                               bm25_top_k=bm25_top_k, dense_top_k=dense_top_k)

        # Run hybrid search for each query variant
        all_results = {}  # chunk_id -> best result

        for i, query in enumerate(queries):
            results = self.search(query, top_k=top_k, filters=filters,
                                  bm25_top_k=bm25_top_k, dense_top_k=dense_top_k)
            for result in results:
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue
                existing_score = all_results.get(chunk_id, {}).get("rrf_score", 0)
                new_score = result.get("rrf_score", 0)
                if chunk_id not in all_results or new_score > existing_score:
                    all_results[chunk_id] = result
        
        # Sort by rrf_score descending and return top_k
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x.get("rrf_score", 0), 
            reverse=True
        )
        
        logger.info(
            f"Multi-query search ({len(queries)} queries): "
            f"{len(sorted_results)} unique results, returning top {top_k}"
        )
        return sorted_results[:top_k]
    
    def search_with_details(self, query: str, top_k: int = 30, filters: Dict = None) -> Tuple[List[Dict], Dict]:
        """
        Search with detailed breakdown of results
        
        Args:
            query: Search query
            top_k: Number of final results to return after fusion
            filters: Filtres métier optionnels (voir `search` pour le format)
        
        Returns:
            Tuple of (fused_results, details_dict)
        """
        bm25_results = self.bm25_search.search(query, top_k=self.bm25_top_k, filters=filters)
        dense_results = self.dense_search.search(query, top_k=self.dense_top_k, filters=filters)
        fused_results = self._rrf_fusion(bm25_results, dense_results, top_k)
        
        details = {
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "bm25_count": len(bm25_results),
            "dense_count": len(dense_results),
            "fused_count": len(fused_results)
        }
        
        return fused_results, details

