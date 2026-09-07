# Plan de construction V4 — validé le 7 septembre 2026

Cible prioritaire : réponses aux avis MRAe (wpd), une remarque à la fois,
validation humaine des sources. Machine de référence : RTX A5000 24 Go,
Qwen2.5 14B sous Ollama, fenêtre 32k.

## Principe

Reprendre le fonctionnement **mesuré** de V3 (watcher, parsers, chunker
paginé, recherche hybride, `query2`) et corriger les erreurs et
incohérences relevées dans l'analyse du 7 septembre. Pas de harnais
agentique libre : le code possède la boucle, le LLM exécute des tâches
bornées à sortie typée (voir `agent-harness-architecture.md`).

## Corrections retenues

| # | Défaut V3 | Correction V4 | Où |
|---|---|---|---|
| 1 | Identifiants Qdrant via `hash()` : doublons à la réingestion | UUID v5 du `chunk_id`, réingérer = remplacer | `src/ingestion/pipeline.py` |
| 1b | Deux voies d'ingestion (admin + watcher) | Watcher seul ; l'admin affiche son flux | `src/watcher`, admin |
| 2 | Détecteur d'esquive sans accents, texte non normalisé | Motif ASCII conservé, texte désaccentué avant comparaison (`unicodedata`) | `src/rag/esquive.py`, `evaluation/` |
| 3 | Bloc `<analyse>` du prompt MRAe cassait le garde-fou | Bloc retiré du prompt ; les stratégies restent en consigne de rédaction | `configs/prompts/` |
| 4 | `top_k` du chat sans effet | Retiré de l'UI et de l'API | chat |
| 5 | `rerank_top_n` défini deux fois (reranker 10, analyzer 15) | Une seule définition dans `settings.yaml`, passée en argument à chaque appel, jamais copiée | `src/config/settings.py`, `src/rag` |
| 6 | Passages réservés placés en tête sans explication | Ordre conservé, étiquette « passage ciblé : {requête} » dans la trace et l'UI | `src/rag/query2.py`, chat |

## Tranches, chacune avec ses tests

| Tranche | Contenu | Porte de passage |
|---|---|---|
| T1 Ingestion (faite, validée 7 sept.) | `src/config/settings.py`, `src/ingestion/*` (parsers, chunker, embedder repris), `pipeline.py` V4, watcher repris | `pytest tests/ingestion tests/watcher` verts ; ingestion d'un tome de Beaumont Sud sans doublon après réingestion |
| T2 RAG (code fait 7 sept., 167 tests ; porte de passage = campagne sur A5000) | `src/rag/` : bm25, dense, hybrid (RRF), reranker sans état, `llm.py`, `query2.py` avec esquive corrigée et étiquette « ciblé », API `POST /api/rag/query` renvoyant `{answer, citations, verdict, confidence, trace_id, passages}` | `campagne_reference.py` sur Beaumont Sud et Brissy : pas de régression d'esquives ni de bon tome par rapport à V3 |
| T3 Chat | Une remarque → réponse, verdict visible, passages étiquetés, citations douteuses surlignées | Test manuel par le chargé d'études sur 5 remarques |
| T4 Admin | Config par instance lue/écrite (formulaire sur `settings.yaml` + reload), flux du watcher, onglet campagne | Modifier `rerank_top_n` depuis l'UI change le nombre de passages à la requête suivante |

## Ce qui est repris tel quel de V3

`parsers.py`, `chunker.py` (pagination mesurée sur le corpus wpd),
`embedder.py`, `excel_parser.py`, `ocr.py`, le module `watcher/` complet
et ses tests. Ces fichiers ne sont modifiés qu'à la marge (lecture de la
configuration).

## Ce qui n'est pas repris

Le tracker SQLite du pipeline (doublon du StateStore du watcher),
l'ingestion manuelle de l'admin, l'ancien harnais `AgentOrchestrator`,
les routes `/api/analyze` modes files/folder/cross (non utilisées par le
cas MRAe), le module mail (réservé au cas Belacom, plus tard).
