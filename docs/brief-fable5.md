# Brief — Fondation RAG & Ingestion, Luciole Prime V4

À soumettre à Claude Fable 5 (idéalement via un harnais agentique type Claude Code, avec accès direct au dépôt, pour exploiter sa fenêtre de contexte de 1M tokens).

## Contexte projet

Luciole est une plateforme d'assistant IA/RAG souveraine, offline, destinée à des clients professionnels, avec pour principe central de répondre uniquement à partir de documents ingérés et sourcés — afin de limiter les hallucinations et de garder le contrôle de la donnée. Elle doit rester neutre et adaptable à tout cas d'usage métier (le premier déploiement testé était centré sur les avis MRAe, mais ce n'est qu'un cas d'usage parmi d'autres).

Le projet est repris dans un nouveau dépôt, **luciole-prime-V4** ([github.com/damien148k/luciole-prime-V4](https://github.com/damien148k/luciole-prime-V4)), qui part d'une feuille blanche pour la fondation technique.

## Ce qui existe et ne doit PAS être réutilisé

- Le pipeline `query2` (générations précédentes) a validé des concepts utiles — analyse de couverture, verdicts de couverture, second passage ciblé — mais son code est trop spécifique aux expériences MRAe passées. Ne pas l'importer ; s'en inspirer uniquement pour les concepts génériques déjà repris dans la spec ci-dessous.
- Le harnais agentique testé sur l'instance Belacom souffrait d'un **problème de conception architecturale profond** (pas un bug isolé). Ce code ne doit servir à aucune reprise, même partielle. Cette couche (agent/harnais/boucle/graph) est une **phase future, hors périmètre de cette demande**.

## Périmètre de cette demande

Construire uniquement la fondation **RAG + ingestion** (couches L1 et L2 de l'architecture cible), from scratch, selon les bonnes pratiques actuelles. Les spécifications complètes sont dans le dépôt :

- `docs/architecture.md` — vue d'ensemble des couches et leur articulation
- `docs/rag-ingestion-spec.md` — spec technique détaillée de la fondation demandée ici

## Contraintes non négociables

1. **Offline/on-premise** : embeddings et reranker exécutés localement, aucune dépendance API cloud dans le chemin de production (Docker Compose par instance client, comme l'existant).
2. **Rien de figé en dur** : chunking, reranking, poids de fusion hybride doivent être lus depuis une configuration par instance (schéma YAML fourni dans la spec), jamais codés en constantes.
3. **Traçabilité complète** : hash de document, métadonnées par chunk, journal d'audit signé par requête.
4. **Contrat d'API stable** : respecter exactement le contrat décrit en section 7 de `docs/rag-ingestion-spec.md`, pour que la future couche agent puisse s'y brancher sans changement de contrat.
5. **Multi-tenant dès la conception**, même si le premier déploiement réel ne concerne qu'un seul client.

## Ce qui est attendu comme livrable

1. Une revue critique de la spec fournie (`docs/rag-ingestion-spec.md`) : incohérences, angles morts, choix à challenger — avant toute écriture de code.
2. Une proposition d'implémentation (structure de code dans `src/ingestion/` et `src/rag/`, stack technique précise : bibliothèques de parsing, store vectoriel, moteur BM25, modèle d'embedding et de reranking auto-hébergeables recommandés).
3. Un plan d'implémentation par tranches, chacune accompagnée de tests, pour permettre une validation humaine à chaque étape plutôt qu'une génération de code massive et non supervisée.
4. Un jeu de test golden set minimal (format, méthodologie) pour valider la fondation avant de passer à la couche agent.

## Notes opérationnelles

- Il s'agit d'une tâche de conception/développement sur du code propriétaire (pas de documents clients confidentiels) — l'usage de Fable 5 est adapté à ce périmètre, en gardant en tête la rétention de 30 jours appliquée par Anthropic sur ce modèle.
- Ne pas faire exécuter la phase agent/harnais/boucle/graph dans le même brief : elle nécessite un post-mortem préalable du harnais Belacom, à traiter séparément.
