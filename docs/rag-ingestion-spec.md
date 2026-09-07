# Spec technique — Fondation RAG & Ingestion (L1 + L2)

Statut : fondation à construire from scratch. Ne réutilise pas le code `query2` existant — celui-ci a validé des concepts (analyse de couverture, second passage ciblé) mais son implémentation est trop spécifique aux expériences MRAe passées pour servir de base neutre.

## 1. Objectifs et contraintes

- Fonctionnement 100% offline/on-premise : embeddings et reranker exécutés localement, aucun appel API cloud dans le chemin de production.
- Multi-tenant : chaque instance client a sa propre configuration, ses propres collections, son propre journal d'audit.
- Aucun paramètre de chunking ou de reranking figé dans le code — tout est lu depuis une configuration par instance, modifiable à terme via l'UI admin.
- Traçabilité complète : chaque chunk et chaque réponse sont audités (source, hash, scores).
- Support prioritaire du français, corpus multilingue possible.

## 2. Pipeline d'ingestion

### 2.1 Connecteurs de source
- Dossier local / upload manuel au lancement ; extensible plus tard (partages réseau, boîtes mail).
- Chaque source a : un identifiant, une cadence de rafraîchissement configurable, un hash de contenu pour détecter les changements.

### 2.2 Parsing par type de document
| Type | Traitement |
|---|---|
| PDF | Extraction texte + tables + hiérarchie de titres/sections |
| DOCX | Extraction structurée (styles de titres préservés) |
| HTML / Markdown | Découpe respectant la hiérarchie de titres, suppression du bruit (nav, boilerplate) |
| Code source | Découpe par fonction/classe via AST, pas de découpe par caractères |

Le pipeline de parsing est conçu en plugins par type de document, pour pouvoir en ajouter sans toucher au reste du pipeline.

### 2.3 Chunking (configurable, jamais figé)
Valeurs par défaut recommandées, surchargeables par instance et par type de document :

| Type de document | Taille par défaut | Overlap | Stratégie |
|---|---|---|---|
| Prose générale | 512–1024 tokens | 10–20% | Recursive / structure-aware |
| Documentation structurée | 400–512 tokens | 10–20% | Découpe par section (titres) |
| Code | Variable | — | Par fonction/classe (AST) |
| Juridique / contrats | Variable | — | Par clause, avec paragraphe complet |

Règles communes :
- Préfixer chaque chunk avec le titre du document et sa section avant embedding (gain de rappel significatif).
- Filtrer les chunks vides ou dupliqués.
- Aucun chunk ne dépasse une limite haute configurable (ex. 2500 tokens) — sous-découpage automatique si dépassement.

### 2.4 Métadonnées par chunk
`chunk_id`, `document_id`, `document_hash` (SHA-256), `source_path`, `section/titre`, `page`, `parent_chunk_id`, `document_type`, `tenant_id`, `ingestion_timestamp`, `langue_detectee`.

### 2.5 Embedding
- Modèle local multilingue auto-hébergeable (famille type BGE-m3 ou équivalent), dimension configurable.
- Vecteurs normalisés avant indexation.

### 2.6 Indexation
- Index vectoriel auto-hébergeable (Qdrant, Weaviate ou pgvector).
- Index lexical (BM25) en parallèle — indexation double systématique pour permettre la recherche hybride.

### 2.7 Journal d'audit d'ingestion
Append-only, horodaté : document, hash, statut, erreurs éventuelles.

## 3. Pipeline de récupération et génération

1. Requête utilisateur → embedding + requête lexicale en parallèle.
2. Récupération hybride : top-N dense (50–100) + top-N BM25 (50–100), N configurable.
3. Fusion par Reciprocal Rank Fusion (k=60 par défaut, configurable).
4. Reranking par cross-encoder local sur les candidats fusionnés → top-K configurable (5–12).
5. Assemblage du contexte : dédoublonnage, ordre, respect de la limite de contexte du LLM, citation attachée à chaque passage.
6. Génération strictement contrainte au contexte fourni, avec instruction explicite de signaler l'absence d'information plutôt que d'extrapoler.
7. **Verdict structuré généralisé** : `SUFFISANT / PARTIEL / INSUFFISANT` + score de confiance.
8. Réponse API : `{answer, citations[], verdict, confidence, trace_id, sources_used[]}`.

## 4. Configuration par instance (exemple)

```yaml
instance: nom-client
ingestion:
  chunking:
    prose: { size: 512, overlap_pct: 15 }
    structured_doc: { size: 450, overlap_pct: 10, respect_headings: true }
    code: { strategy: ast }
  embedding_model: bge-m3-fr
retrieval:
  dense_top_n: 75
  bm25_top_n: 75
  fusion: rrf
  rrf_k: 60
  rerank_top_k: 8
  reranker_model: bge-reranker-local
security:
  tenant_isolation: true
  pii_redaction: false
refresh:
  default_cadence: weekly
```

Ce fichier est la surface exacte que l'UI admin (L4) devra pouvoir lire et modifier — aucun de ces paramètres ne doit vivre ailleurs que dans cette configuration.

## 5. Sécurité et conformité

- Hash SHA-256 par document, vérifié avant chaque récupération.
- Contrôle d'accès par tenant et par document/chunk (RBAC/ABAC).
- Chiffrement au repos (AES-256) et en transit (TLS).
- Journal d'audit signé pour chaque requête : question, chunks retournés (hash + source + scores), métadonnées modèle, réponse, décision de verdict.
- Politique de rétention configurable par tenant.

## 6. Évaluation et non-régression

- Jeu de test golden set par instance (questions, réponses de référence, sources attendues) — généralisation de la méthodologie déjà appliquée sur les corpus MRAe/Beaumont Sud, mais indépendante de tout cas d'usage spécifique.
- Métriques suivies : recall@k, précision de citation, taux de faux `INSUFFISANT`, latence p50/p95.
- Rejoué en intégration continue à chaque changement de configuration ou de code.

## 7. Contrat d'API exposé (pour la future couche agent L3)

- `POST /api/rag/query` — `{tenant_id, question, filters?, overrides?}` → `{answer, citations, verdict, confidence, trace_id}`
- `POST /api/ingestion/documents` — `{tenant_id, source}` → `{document_id, status}`
- `GET /PUT /api/admin/config/{tenant_id}` — lecture/écriture de la configuration décrite en section 4

Ce contrat doit rester stable même quand L3 (agent/harnais/boucle/graph) sera construit par-dessus : L3 doit pouvoir appeler ce RAG comme un outil parmi d'autres, sans connaître son fonctionnement interne.
