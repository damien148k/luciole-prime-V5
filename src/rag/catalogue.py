"""Catalogue des titres du corpus pour l'analyse de couverture (query2).

Problème : l'analyse de couverture ne voit que les passages récupérés.
Sur un corpus multi-tomes (études d'impact...), un tome de synthèse
(RNT) peut donner une réponse d'apparence complète : verdict COUVERT,
pas de seconde passe, alors que le tome spécialisé n'a jamais été
récupéré. Mesuré le 15 août 2026 sur la Panière du Fort : « quels sont
les enjeux paysagers ? » restitué sans le « Volet paysager et
patrimonial », avec les confusions de synthèse que le tome dédié
aurait évitées.

Solution : un inventaire des titres du corpus, construit par agrégation
OpenSearch sur file_name.keyword, mis en cache (TTL), puis injecté dans
le prompt de couverture. Titres seuls : aucun résumé n'est généré à
l'ingestion. Le module ne connaît de BM25Search que `.client` et
`.index_name` (duck-typing) pour rester testable sans la pile
opensearchpy.
"""

import time
from typing import List, Optional

from loguru import logger

# Durée de vie du cache : le catalogue ne change qu'à l'ingestion, mais
# un TTL court évite de dépendre d'un signal d'invalidation du watcher.
CATALOGUE_TTL_S = 300

# Plafond de l'agrégation : au-delà, le corpus n'est plus un corpus de
# tomes nommés et l'inventaire perd son sens dans le prompt.
CATALOGUE_MAX_DOCS = 500


class CatalogueDocuments:
    """Inventaire des titres de documents indexés, avec cache TTL.

    Un échec de lecture (OpenSearch indisponible...) n'est jamais mis en
    cache : l'appel suivant retente, et le dernier inventaire connu est
    servi en attendant. Retour vide = fonctionnalité silencieusement
    désactivée, le prompt de couverture reste alors identique à la
    version sans catalogue.
    """

    def __init__(self, bm25_search=None, ttl_seconds: int = CATALOGUE_TTL_S):
        self._bm25 = bm25_search
        self._ttl = ttl_seconds
        self._cache_titres: Optional[List[str]] = None
        self._cache_ts = 0.0

    def titres(self) -> List[str]:
        """Titres triés des documents du corpus (cache TTL, repli souple)."""
        now = time.monotonic()
        if self._cache_titres is not None and (now - self._cache_ts) < self._ttl:
            return self._cache_titres
        titres = self._charger()
        if titres:
            self._cache_titres = titres
            self._cache_ts = now
            return titres
        # Échec ou corpus vide : servir le dernier inventaire connu.
        return self._cache_titres or []

    def _charger(self) -> List[str]:
        if self._bm25 is None:
            return []
        try:
            reponse = self._bm25.client.search(
                index=self._bm25.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "titres": {
                            "terms": {
                                "field": "file_name.keyword",
                                "size": CATALOGUE_MAX_DOCS,
                            }
                        }
                    },
                },
            )
            buckets = reponse["aggregations"]["titres"]["buckets"]
            return sorted(b["key"] for b in buckets)
        except Exception as e:
            logger.warning(f"catalogue: lecture des titres impossible ({e})")
            return []
