"""Interfaces web de Luciole V4 (couche 3 de la feuille de route).

Chaque UI est un service FastAPI autonome qui NE parle au coeur RAG que
par son contrat public (POST /api/rag/query). Aucune UI n'importe le
pipeline : elles restent remplacables sans toucher a L1/L2.
"""
