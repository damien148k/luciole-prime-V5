# Post-mortem — harnais agentique « Belacom » (Luciole Prime v3)

**Statut : archivé, à titre d'enseignement pour Luciole Prime V4.**
**Périmètre : la couche agent/harnais/boucle (`AgentOrchestrator`) qui appelait le RAG comme un outil. `query2` (le pipeline itératif classique : recherche → analyse de couverture → recherche ciblée → génération) reste validé et n'est pas concerné par ce post-mortem.**

Ce document reconstitue les faits à partir de l'historique réel du dépôt [damien148k/luciole-prime-v3](https://github.com/damien148k/luciole-prime-v3) (code, commits, pull requests fusionnées). Il ne s'appuie sur aucun souvenir approximatif : chaque affirmation est sourcée par un lien vers le commit ou la PR correspondante.

## Résumé exécutif

Entre le 31 juillet et le 22 août 2026, un harnais agentique borné (boucle plan/act/observe, outils fixes, conditions d'arrêt explicites) a été construit pour Luciole, testé en conditions réelles sur une instance pilote de support technique (dossier interne désigné « Belacom »), puis comparé à un pipeline procédural simple sur le jeu de test MRAe. Le harnais agentique a systématiquement sous-performé un pipeline non agentique plus simple, pour des raisons qui tiennent à la conception même de la boucle et non à des bugs isolés. Le 22 août 2026, le harnais a été entièrement retiré du dépôt ([PR #41](https://github.com/damien148k/luciole-prime-v3/pull/41)) au profit de `query2`, devenu le chemin unique.

La cause racine n'est pas une instabilité ponctuelle : c'est un problème de conception plus profond, confirmé par l'utilisateur. Le harnais confiait à un unique appel LLM, à chaque étape, à la fois le choix de l'outil à appeler et la décision d'arrêt (`final_answer` / `no_answer` / `escalate_to_human`), sans séparation entre extraction du besoin, recherche et rédaction, et sans aucune règle déterministe pour les décisions critiques (les `routing_rules` du profil étaient explicitement documentées comme non exécutées, à la charge du jugement du LLM).

## Chronologie factuelle

| Date | Événement | Référence |
|---|---|---|
| 31/07/2026 | Création de la boucle agentique bornée (`AgentOrchestrator`, `ToolRegistry`, profils YAML `generic` et `belacom_support`) | [PR #3](https://github.com/damien148k/luciole-prime-v3/pull/3) |
| 31/07/2026 | Premier test réel sur l'instance pilote de support : question « dernière panne VPN Fortinet ? », le ticket pertinent (`ticket-BLM-2025-0314.md`) est trouvé dès l'étape 1, mais l'agent boucle jusqu'à `max_steps` sans conclure | [PR #4](https://github.com/damien148k/luciole-prime-v3/pull/4) |
| 31/07/2026 | Correctif : `min_sources` assoupli de 2 à 1, et blocage de la répétition mot pour mot d'une réponse déjà refusée | [PR #4](https://github.com/damien148k/luciole-prime-v3/pull/4) |
| 31/07/2026 | Ajout de l'onglet Admin « Agents » (profils, traces, escalades) | [PR #5](https://github.com/damien148k/luciole-prime-v3/pull/5) |
| 31/07/2026 | Le reranker cross-encoder n'était pas branché sur les recherches de l'agent (seul le tri RRF s'appliquait) | [PR #6](https://github.com/damien148k/luciole-prime-v3/pull/6) |
| 01/08/2026 | Découverte que le planificateur ne recevait que des noms de fichiers, pas le contenu des extraits : il planifiait « à l'aveugle » et a produit du contenu inventé attribué à de vrais tickets cités sur l'instance pilote | [PR #10](https://github.com/damien148k/luciole-prime-v3/pull/10) |
| 01/08/2026 | Trois défauts sur le chemin négatif : une escalade volontaire du LLM ne terminait pas la boucle (5 appels `escalate_to_human` consécutifs jusqu'à `max_steps`) ; aucune distinction entre lacune documentaire et urgence | [PR #11](https://github.com/damien148k/luciole-prime-v3/pull/11) |
| 02/08/2026 | Sur le jeu de test MRAe (20 remarques), le rappel documentaire de la chaîne agentique complète plafonnait à 2/12, alors qu'une recherche seule sur les mêmes cas obtenait 7/12 à 8/12 avec le bon document en rang 1-2 : le moteur de recherche n'était pas en cause, l'agent rejouait la même requête à l'identique jusqu'à 5 fois | [PR #15](https://github.com/damien148k/luciole-prime-v3/pull/15) |
| 03/08/2026 | Le pipeline procédural historique (`src/api/main.py`, non démarré par aucun service) est supprimé comme code mort, l'agent devenant le seul chemin de production | [PR #20](https://github.com/damien148k/luciole-prime-v3/pull/20) |
| 04/08/2026 | Plafond dur du nombre de recherches par question (`max_searches`), car une consigne textuelle « n'émets pas plus de trois requêtes » n'était pas respectée par le modèle | [PR #28](https://github.com/damien148k/luciole-prime-v3/pull/28) |
| 05-06/08/2026 | Mesure comparative sur 20 cas MRAe : pipeline procédural simple = 0/20 esquives, 9/10 bon tome, 14,3 min ; pipeline avec déduction de question = 1/20 esquives, 10/10 bon tome, 29,1 min ; **boucle agentique complète = 4 sans réponse, 8 bons/2 mauvais** | [PR #31](https://github.com/damien148k/luciole-prime-v3/pull/31) |
| 06/08/2026 | Introduction du commutateur `CHAT_ROUTE` pour permettre de comparer la route agentique à la route procédurale, la route agentique restant la moins bonne des trois mesurées | [PR #31](https://github.com/damien148k/luciole-prime-v3/pull/31), [PR #32](https://github.com/damien148k/luciole-prime-v3/pull/32) |
| 08/08/2026 | Création de `query2`, pipeline itératif non agentique (recherche classique → analyse de couverture → extraction de sujet → recherche ciblée sous quota réservé → génération) | commits [6f0af66](https://github.com/damien148k/luciole-prime-v3/commit/6f0af66) / [06fbdde](https://github.com/damien148k/luciole-prime-v3/commit/06fbdde) |
| 11/08/2026 | Le profil `belacom_support.yaml` est renommé `support_technique.yaml` pour ne plus exposer le nom d'un client dans le produit livré | [commit b556f62](https://github.com/damien148k/luciole-prime-v3/commit/b556f62856f479a8ec4fa67908111d7888afd686) |
| 15/08/2026 | `query2` atteint 14/16 réponses substantielles et sourcées sur le jeu MRAe (contre 7/16 pour la route classique) ; devient la route de production par défaut | commits `query2 v2.9`/`v2.10` |
| 22/08/2026 | Suppression complète de `orchestrator.py`, `tools.py`, du système `AGENT_PROFILE`, de la route `/api/agent/run` et des profils YAML. `/api/query2` devient l'unique chemin RAG du service | [PR #41](https://github.com/damien148k/luciole-prime-v3/pull/41) |

## Architecture du harnais retiré

Le harnais suivait un schéma ReAct classique, borné :

```
pour step in 1..max_steps :
    prompt = question + outils disponibles + observations accumulées
    decision = LLM.call(system_prompt, prompt)   # un seul appel, un seul JSON {"tool", "args"}
    si decision.tool == "final_answer" :
        si conditions_arret non satisfaites (min_sources, require_citation) :
            observation = réponse refusée, retente
            continue
        retourner réponse
    si decision.tool in ("no_answer", "escalate_to_human") :
        retourner (sortie terminale)
    sinon :
        exécuter l'outil, ajouter l'observation, continue
```

Outils disponibles : `search_documents`, `search_multi`, `get_document`, `escalate_to_human`, `final_answer`, `no_answer` (registre fixe, filtré par profil métier YAML).

Point de conception central, documenté dans le code lui-même : la même décision LLM porte à la fois le choix de l'outil et le jugement d'arrêt. Il n'existe pas de séparation entre une étape d'extraction du besoin, une étape de recherche et une étape de rédaction — contrairement à `query2`, qui sépare explicitement recherche classique, analyse de couverture, extraction de sujet et génération.

Autre point structurel : le profil YAML prévoyait des `routing_rules` (ex. « sévérité critique → escalade automatique ») mais la documentation du fichier précise explicitement que ces règles ne sont *pas* exécutées par l'orchestrateur v1 et ne servent que de documentation pour une future version — la décision d'escalade dépendait donc entièrement du jugement du LLM à chaque étape, sans filet déterministe.

## Symptômes observés (preuves factuelles)

1. **Boucle improductive sur réponse répétée** — l'agent retentait mot pour mot une `final_answer` déjà refusée, épuisant `max_steps` sans jamais varier de stratégie ([PR #4](https://github.com/damien148k/luciole-prime-v3/pull/4)).
2. **Escalade volontaire qui ne termine pas la boucle** — un appel LLM à `escalate_to_human` était traité comme un outil intermédiaire ordinaire ; reproduit en test : 5 appels consécutifs jusqu'à `max_steps` ([PR #11](https://github.com/damien148k/luciole-prime-v3/pull/11)).
3. **Hallucination attribuée à de vraies sources** — le planificateur ne recevant que des noms de fichiers (pas le contenu), il a rédigé une réponse inventée en citant deux tickets réels non pertinents ([PR #10](https://github.com/damien148k/luciole-prime-v3/pull/10)).
4. **Requêtes répétées à l'identique** — sur un cas mesuré, l'agent a rejoué cinq fois la même requête de recherche, aboutissant à `no_answer` alors que le document correct était en rang 1-2 d'une recherche directe ([PR #15](https://github.com/damien148k/luciole-prime-v3/pull/15)).
5. **Sous-performance quantifiée face à un pipeline non agentique** — 2/12 de rappel documentaire pour la chaîne agentique complète contre 7/12 à 8/12 pour la recherche seule sur les mêmes cas ([PR #15](https://github.com/damien148k/luciole-prime-v3/pull/15)) ; puis, en comparaison contrôlée sur 20 cas, l'agent complet obtient 4 réponses sans résultat et un score de justesse documentaire inférieur aux deux variantes procédurales testées ([PR #31](https://github.com/damien148k/luciole-prime-v3/pull/31)).

Chaque symptôme a fait l'objet d'un correctif ponctuel mergé (reranker branché, contenu transmis, sorties négatives nettoyées, recherches répétées bloquées, plafond de recherches). Après une dizaine de correctifs de ce type entre le 31 juillet et le 4 août, la mesure comparative du 5-6 août a montré que même corrigée, la boucle agentique restait la route la moins bonne des trois testées. C'est cette mesure, et non un bug résiduel, qui a motivé le basculement vers `query2` puis le retrait complet du harnais le 22 août.

## Cause racine architecturale

Le problème n'est pas une suite de bugs d'intégration : c'est le choix de conception d'un agent ReAct libre pour une tâche qui, empiriquement, se résout mieux par un pipeline itératif à étapes séparées et déterministes.

Trois défauts structurels se combinent :

- **Décision d'outil et décision d'arrêt confondues dans un seul appel LLM.** Chaque étape demande au modèle de choisir *et* de juger simultanément s'il en sait assez — sans étape de vérification indépendante de la couverture documentaire (ce que `query2` fait explicitement via son analyse de couverture).
- **Aucun garde-fou déterministe sur les décisions critiques.** Les `routing_rules` (ex. escalade automatique sur sévérité critique) existaient dans la configuration mais n'étaient jamais exécutées : la robustesse du système dépendait entièrement de la constance d'un LLM à choisir le bon outil au bon moment, y compris sur des cas d'urgence.
- **Absence de mécanisme de variation de stratégie de recherche.** Rien n'empêchait le modèle de rejouer une requête identique ou proche ; chaque garde-fou ajouté (répétition bloquée, plafond de recherches) a traité un symptôme sans changer le principe : c'est toujours le LLM, à chaque étape, qui décide seul de la prochaine requête, sans logique de repli structurée.

Ces trois points expliquent pourquoi les correctifs successifs (PR #4, #6, #10, #11, #15, #28) ont amélioré des cas isolés sans jamais rattraper la performance d'un pipeline procédural plus simple : ils traitaient des symptômes d'un principe de boucle libre plutôt que la cause.

## Enseignements pour Luciole Prime V4

1. **Ne pas reconstruire un ReAct libre sur le modèle Belacom.** La couche agent/harnais de V4 (L3, prévue comme un développement futur d'après `docs/architecture.md`) doit séparer explicitement les décisions déterministes (couverture suffisante ou non, nombre de sources, escalade sur critère métier) des décisions laissées au jugement du LLM (formulation de requête, synthèse). Les critères déterministes doivent être évalués en code, jamais uniquement par un jugement LLM à chaque étape.
2. **Toute règle de routage métier critique (escalade, sévérité) doit être exécutée en code, jamais laissée en documentation non appliquée.** C'est exactement l'écart identifié dans le profil `belacom_support.yaml`.
3. **Le contenu des extraits doit toujours être transmis au composant qui décide**, jamais uniquement des métadonnées ou noms de fichiers — sous peine de fabrication de contenu attribué à de vraies sources.
4. **Toute sortie terminale explicite (escalade, absence de réponse) doit interrompre immédiatement le traitement**, sans laisser un mécanisme de boucle la retraiter comme une étape intermédiaire.
5. **Mesurer avant de généraliser.** La décision de retrait n'est venue ni d'une intuition ni d'un ressenti utilisateur, mais d'une mesure comparative chiffrée sur un jeu de cas fixe. V4 doit conserver ce principe : tout composant agentique candidat pour L3 devra être comparé à une base de référence (le futur pipeline RAG generique de L2) sur un jeu d'évaluation, avant toute généralisation en production.
6. **`query2` n'est pas le modèle à copier tel quel pour V4** (il reste volontairement spécifique aux cas d'usage passés), mais son principe de séparation explicite des étapes — recherche, analyse de couverture, complément ciblé, génération — est la leçon de conception à retenir face à l'échec du harnais Belacom.

## Sources

- [damien148k/luciole-prime-v3](https://github.com/damien148k/luciole-prime-v3) — dépôt source
- [PR #3](https://github.com/damien148k/luciole-prime-v3/pull/3) — création du harnais agentique
- [PR #4](https://github.com/damien148k/luciole-prime-v3/pull/4) — premier test réel, `min_sources` et boucle de réponse répétée
- [PR #5](https://github.com/damien148k/luciole-prime-v3/pull/5) — onglet Admin Agents
- [PR #6](https://github.com/damien148k/luciole-prime-v3/pull/6) — reranker non branché
- [PR #10](https://github.com/damien148k/luciole-prime-v3/pull/10) — contenu non transmis au planificateur, hallucination
- [PR #11](https://github.com/damien148k/luciole-prime-v3/pull/11) — sorties négatives et escalade volontaire
- [PR #15](https://github.com/damien148k/luciole-prime-v3/pull/15) — recherches répétées, rappel documentaire 2/12
- [PR #20](https://github.com/damien148k/luciole-prime-v3/pull/20) — suppression du pipeline procédural historique
- [PR #28](https://github.com/damien148k/luciole-prime-v3/pull/28) — plafond dur du nombre de recherches
- [PR #31](https://github.com/damien148k/luciole-prime-v3/pull/31) — mesure comparative agent vs procédural
- [PR #32](https://github.com/damien148k/luciole-prime-v3/pull/32) — déterminisme et outils de mesure
- [PR #41](https://github.com/damien148k/luciole-prime-v3/pull/41) — retrait complet du harnais, `query2` seul chemin
- [Commit b556f62](https://github.com/damien148k/luciole-prime-v3/commit/b556f62856f479a8ec4fa67908111d7888afd686) — neutralisation du nom client dans le profil
- [Commit 6fce794](https://github.com/damien148k/luciole-prime-v3/commit/6fce7940bb8a56dc2ada0e22e08caa6cab7728f6) — suppression finale du code de la boucle agentique
