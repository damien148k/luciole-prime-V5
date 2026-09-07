# Architecture cible — Luciole Prime V4

## Principes directeurs

- **Neutralité et adaptabilité** : aucune logique métier (MRAe ou autre) n'est codée en dur. Tout ce qui varie selon le cas d'usage (chunking, reranking, prompts, seuils de verdict) est de la configuration par instance, pilotable depuis l'UI admin.
- **RAG-only** : le système ne répond qu'à partir de documents ingérés et sourcés. Aucune réponse ne doit être générée sans passage par la couche de récupération.
- **Souveraineté** : inférence (embeddings, reranker, génération) exécutée on-premise/offline. Aucune dépendance à une API cloud dans la boucle de production.
- **Construction par couches vérifiées** : chaque couche est validée par un jeu de test avant que la couche suivante ne s'appuie dessus. Pas de génération de code non supervisée sur l'ensemble de la pile.
- **Multi-tenant** : chaque client a sa propre configuration, ses propres collections et son propre journal d'audit, déployés en instances Docker isolées comme aujourd'hui.

## Vue d'ensemble des couches

```
┌─────────────────────────────────────────────┐
│ L4 — Interfaces                              │
│  Chat UI (utilisateur final)                 │
│  Admin UI (config RAG, agents, audit)        │
├─────────────────────────────────────────────┤
│ L3 — Agent / Harnais / Boucle / Graph         │
│  (phase future — reconstruction complète,     │
│   aucune reprise du code Belacom)             │
├─────────────────────────────────────────────┤
│ L2 — Cœur RAG (retrieval + génération)        │
│  Hybrid search → fusion RRF → reranking       │
│  → génération contrainte → verdict structuré  │
├─────────────────────────────────────────────┤
│ L1 — Ingestion                                │
│  Parsing par type → chunking configurable     │
│  → embedding local → indexation hybride       │
├─────────────────────────────────────────────┤
│ L0 — Infrastructure                           │
│  Docker Compose par instance, GPU/CPU local   │
└─────────────────────────────────────────────┘
```

## L0 — Infrastructure

Inchangé dans son principe par rapport à l'existant : une instance Docker Compose par client, réseau isolé, modèles d'inférence locaux (LLM, embeddings, reranker). Point d'attention pour V4 : préparer dès le départ un fichier de configuration par instance (`configs/<tenant>.yaml`) au lieu de variables d'environnement dispersées, pour que l'admin UI (L4) puisse le lire/écrire directement.

## L1 — Ingestion

Voir `docs/rag-ingestion-spec.md` pour le détail complet. Résumé :

- Connecteurs par source, avec hash de contenu et cadence de rafraîchissement configurable.
- Parsing spécifique par type de document (PDF, DOCX, HTML/Markdown, code).
- Chunking structure-aware, avec taille/overlap par défaut selon le type de document, mais toujours surchargeable en configuration — jamais figé dans le code.
- Métadonnées complètes par chunk (source, section, page, chunk parent, hash document, tenant).
- Indexation double : vectorielle (dense) + lexicale (BM25), pour permettre la recherche hybride en L2.
- Journal d'audit d'ingestion append-only.

## L2 — Cœur RAG

- Recherche hybride (dense + BM25) fusionnée par Reciprocal Rank Fusion.
- Reranking par cross-encoder local, réduction à un top-K configurable.
- Génération strictement contrainte au contexte récupéré, avec instruction explicite de signaler l'absence d'information plutôt que d'extrapoler.
- **Verdict structuré généralisé** : `SUFFISANT / PARTIEL / INSUFFISANT` + score de confiance — généralisation du triptyque COUVERT/PARTIEL/NON_COUVERT de l'ancien query2, mais pensé comme un signal générique exploitable par n'importe quel cas d'usage, pas seulement MRAe.
- Réponse API structurée avec citations, verdict, confiance et identifiant de trace — c'est ce contrat qui permettra à la couche agent (L3) de prendre des décisions sans connaître les détails internes du RAG.

## L3 — Agent / Harnais / Boucle / Graph (phase future)

Conception détaillée disponible dans `docs/postmortem-belacom.md` (post-mortem du harnais Belacom, cause racine confirmée) et `docs/agent-harness-architecture.md` (architecture proposée). Résumé :

- **Reconstruction complète depuis zéro.** Le harnais testé sur Belacom a révélé un problème de conception architecturale profond (pas de bug isolé) : il ne doit pas servir de référence, y compris pour éviter de reproduire son pattern par réflexe.
- **Routeur déterministe, pas un agent libre.** Le verdict L2 (`SUFFISANT/PARTIEL/INSUFFISANT` + confidence) pilote une machine à états en code : appel L2 unique par défaut, un seul cycle de raffinement borné si `PARTIEL`, nœuds terminaux non réentrants. Le LLM n'intervient que pour reformuler une requête ou synthétiser, jamais pour décider d'arrêter.
- **Moteur de règles métier réellement exécuté.** Les règles de routage (ex. escalade sur sévérité critique) sont interprétées en code à chaque requête via un évaluateur à liste blanche, pas seulement documentées comme sur Belacom.
- **Pattern superviseur + spécialistes (optionnel, phase 2)** : un orchestrateur au-dessus de plusieurs instances RAG, construit uniquement si un tenant en a un besoin réel et mesuré.
- **Porte de mesure obligatoire à chaque phase** : toute extension de L3 doit être comparée à L2 seul sur un jeu d'évaluation golden set avant généralisation — c'est l'absence de cette mesure préalable qui a permis au harnais Belacom d'aller en production avant d'être invalidé.

## L4 — Interfaces

- **Chat UI** : consomme uniquement l'API de L2 (ou de L3 une fois disponible) — aucune logique RAG côté interface.
- **Admin UI** : lecture/écriture de la configuration par instance (chunking, reranking, hybrid weights, cadence de rafraîchissement, contrôle d'accès), visualisation du journal d'audit, et à terme pilotage des profils d'agents et traces d'exécution (reprise de l'idée de l'onglet "Agents" déjà esquissé côté Luciole Prime v3, mais sur la nouvelle fondation).

## Transversal

- **Sécurité** : chiffrement au repos/en transit, contrôle d'accès par tenant jusqu'au niveau chunk, hash de document vérifié à chaque ingestion.
- **Observabilité** : trace structurée par requête (candidats récupérés, scores, décision de fusion/reranking, réponse) — condition nécessaire pour que L3 puisse un jour s'appuyer sur ces signaux.
- **Évaluation** : jeu de test golden set par instance, rejoué en CI à chaque changement, sur le modèle des benchmarks déjà pratiqués (MRAe, Beaumont Sud) mais généralisé à toute instance.
