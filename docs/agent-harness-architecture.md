# Architecture de la couche agentique (L3) — Luciole Prime V4

Statut : proposition de conception, à construire après stabilisation de L1 (ingestion) et L2 (cœur RAG). Reconstruction complète depuis zéro — aucune reprise du code ou des hypothèses du harnais Belacom (voir `docs/postmortem-belacom.md`).

## 0. Ce que cette conception corrige explicitement

Chaque choix ci-dessous répond à une cause racine identifiée dans le post-mortem Belacom :

| Défaut constaté (Belacom) | Correction dans cette conception |
|---|---|
| Un seul appel LLM par étape décidait à la fois l'outil et l'arrêt | Décision d'arrêt et de routage évaluée en code, à partir du verdict structuré renvoyé par L2 — jamais laissée au jugement libre du LLM |
| `routing_rules` déclarées en YAML mais jamais exécutées | Moteur de règles réellement interprété en code à chaque requête, avant et après l'appel RAG |
| Boucle non bornée avec requêtes répétées à l'identique | Un seul cycle de raffinement autorisé par défaut, compteur décrémenté en code, jamais délégué au LLM |
| Escalade volontaire qui ne terminait pas la boucle | Nœuds terminaux structurellement non réentrants |
| Planificateur ne recevant que des noms de fichiers | L3 consomme directement `answer` + `citations` déjà assemblées par L2, jamais des métadonnées seules |
| Généralisation sans mesure préalable | Obligation de comparer contre L2 seul sur un jeu d'évaluation avant toute mise en production, à chaque phase |

## 1. Principe directeur

L3 n'est **pas** un agent autonome libre. C'est un **routeur déterministe** au-dessus de L2, avec un unique cycle de raffinement borné, et un LLM utilisé pour des tâches ponctuelles et non pour piloter le flux de contrôle lui-même. Le LLM ne choisit jamais s'il faut s'arrêter — il ne fait que reformuler une requête ou rédiger une synthèse quand le code lui a déjà confié cette tâche précise.

Règle de conception : **le chemin le plus simple qui fonctionne est le chemin par défaut**. Un appel unique à L2 (`POST /api/rag/query`) suffit pour la grande majorité des questions. L3 n'intervient que pour les cas où le verdict L2 dit explicitement que ce n'est pas suffisant.

## 2. Vue d'ensemble — machine à états, pas une boucle ReAct libre

```
0. Garde métier (code) — règles pré-requête (ex: severity=critical)
   |
   |-- une règle matche une action terminale --------------> TERMINAL : escalade immédiate
   |
   `-- aucune règle ne matche
      |
      v
   1. Appel RAG (outil L2) — POST /api/rag/query
      |
      v
   2. Évaluateur de verdict (table de décision, code)
      |
      |-- SUFFISANT, confidence >= seuil ------------------> TERMINAL : réponse directe
      |
      |-- INSUFFISANT ou PARTIEL persistant après raffinement
      |     |
      |     v
      |   Garde métier (code) — règles post-verdict
      |     |-- règle matche (ex. sévérité) ---------------> TERMINAL : escalade
      |     `-- aucune règle ne matche --------------------> TERMINAL : no_answer
      |
      `-- PARTIEL, et budget de raffinement > 0
            |
            v
      3. Raffinement borné (1 itération par défaut)
         reformulation LLM ciblée (question + verdict + extraits déjà récupérés)
         -> retour à l'étape 1 avec la requête reformulée
         (compteur de raffinements décrémenté en code)
            |
            `-- budget de raffinement épuisé --------------> TERMINAL : réponse partielle + avertissement
```

## 3. Le moteur de règles métier (corrige le défaut central de Belacom)

Le profil `belacom_support.yaml` documentait des `routing_rules` (ex. « sévérité critique → escalade ») explicitement marquées comme non exécutées. Ici, ces règles sont interprétées à l'exécution par un évaluateur restreint (liste blanche de champs et d'opérateurs, pas d'`eval` Python libre) :

```yaml
# config/agent_profiles/<tenant>.yaml — édité depuis l'UI admin (L4)
agent_profile:
  tenant_id: nom-client

  verdict_thresholds:
    suffisant_confidence_min: 0.70   # sous ce seuil, SUFFISANT est traité comme PARTIEL

  max_refinements: 1                  # nombre de cycles de raffinement autorisés, borné en code

  routing_rules:                      # évaluées dans l'ordre, la première qui matche gagne
    - when: "metadata.severity == 'critical'"
      then: escalate_to_human
      priority: 1
      reason: "Sévérité critique déclarée sur la question ou les documents retrouvés"
    - when: "verdict == 'INSUFFISANT' and confidence < 0.3"
      then: no_answer
      priority: 2
    - when: "verdict == 'INSUFFISANT'"
      then: escalate_to_human
      priority: 3
      reason: "Corpus insuffisant après cycle de raffinement"

  escalation:
    channel: webhook            # ou email, ticket
    endpoint: "https://..."

  supervisor:
    enabled: false               # activé seulement si plusieurs collections doivent être comparées
    instances: []
```

- Champs autorisés dans `when` : uniquement les champs exposés par la réponse L2 (`verdict`, `confidence`, `metadata.*` provenant des chunks retrouvés) et par le contexte de requête (`history_length`, `custom_prompt_present`). Aucune exécution de code arbitraire.
- Ce fichier est strictement de la configuration — le moteur qui l'interprète, lui, vit dans le code de L3 et est couvert par des tests (mêmes garanties que L1/L2 : rien de figé, mais tout ce qui est déclaré est réellement exécuté).
- Toute modification de ce profil depuis l'UI admin doit être validée (schéma + syntaxe des règles) avant sauvegarde, exactement comme l'était déjà `REQUIRED_KEYS` sur l'ancien système de profils — mais avec en plus la validation de la grammaire des règles.

## 4. Rôle du LLM — restreint à des tâches ponctuelles, jamais au contrôle de flux

Le LLM n'est appelé qu'à deux endroits, chacun avec une sortie strictement typée (schéma JSON validé, pas de parsing par expression régulière comme dans l'ancien `_parse_decision`) :

1. **Reformulation ciblée (nœud 3)** — reçoit la question originale, le verdict, la confiance, et les extraits déjà récupérés par L2 (contenu complet, jamais uniquement des noms de fichiers). Produit une requête reformulée et, si pertinent, des filtres de métadonnées additionnels. Ne décide jamais d'arrêter ou de continuer — cette décision reste dans l'évaluateur de verdict (nœud 2).
2. **Synthèse finale en cas de raffinement (optionnel)** — si le second appel L2 ramène des passages complémentaires, une synthèse peut fusionner les deux réponses citées. Sinon, la réponse L2 est renvoyée telle quelle : pas de réécriture systématique qui risquerait d'introduire du contenu non sourcé.

Aucun tool générique type `search_documents`/`get_document` n'est exposé à un LLM en L3. Le seul outil de récupération est l'appel à L2 via son contrat d'API stable (`POST /api/rag/query`), qui encapsule déjà la recherche hybride, le reranking et le calcul de couverture — L3 n'a pas à réinventer une logique de recherche.

## 5. Contrat d'outils exposés à L3

| Outil | Rôle | Type de décision |
|---|---|---|
| `rag_query(tenant_id, question, filters?)` | Appelle `POST /api/rag/query` (L2). Unique point d'accès aux documents. | Aucune — passthrough |
| `reformulate_query(question, verdict, excerpts)` | LLM à sortie structurée, produit une requête + filtres | LLM, sans contrôle de flux |
| `notify_escalation(reason, metadata)` | Effet de bord déterministe (webhook/ticket) | Déclenché uniquement par le moteur de règles ou un nœud terminal, jamais choisi librement par un LLM |
| `compare_instances(tenant_ids[], question)` *(optionnel, superviseur)* | Appelle plusieurs instances L2 en parallèle, compare par règle déterministe (ex. confidence la plus haute, verdict le plus favorable) | Code, pas d'heuristique de sous-chaînes type ancien `no_info_patterns` |

## 6. Contrat d'API de L3

```
POST /api/agent/query
  {tenant_id, question, history?, custom_prompt?}
  →
  {
    answer,
    verdict,              # SUFFISANT | PARTIEL | INSUFFISANT (verdict final après raffinement éventuel)
    confidence,
    sources[],
    trace_id,
    stopped_reason,        # reponse_directe | reponse_partielle | no_answer | escalade
    steps_used,            # 1 ou 2 (appel initial + raffinement éventuel)
    escalated: bool,
    escalation_reason?,
    matched_rules[]        # liste des règles métier qui ont matché, pour l'audit
  }
```

`stopped_reason` correspond toujours à un nœud terminal structurel du graphe, jamais à une chaîne produite librement par un LLM — c'est directement vérifiable dans le code du routeur, contrairement à l'ancien `stopped_reason` de `AgentOrchestrator` qui pouvait aussi valoir `final_answer_fallback` sur une réponse non parsable.

## 7. Superviseur multi-instances (optionnel, activé seulement si justifié)

Construit uniquement si un tenant a effectivement plusieurs collections à comparer (ex. précise/large/guides déjà évoqué pour Luciole-MRAe). Dans ce cas, `compare_instances` appelle chaque instance L2 en parallèle et retient la meilleure réponse selon une règle déterministe explicite (verdict le plus favorable puis confidence la plus élevée, à égalité le plus grand nombre de sources), jamais selon une détection de motifs textuels comme le faisait `_pick_deep_search_result` dans l'ancien harnais Belacom (recherche de sous-chaînes « pas d'information », fragile et non traçable).

## 8. Observabilité et audit

- Chaque exécution produit une trace complète : nœuds traversés, verdict à chaque étape, règles évaluées et laquelle a matché, requête de raffinement utilisée le cas échéant, durée par étape.
- Cette trace est stockée (équivalent de l'ancienne table `agent_runs`, reconstruite) et consultable depuis l'onglet Admin « Agents » (L4), avec un filtre par `stopped_reason` et par règle déclenchée — ce qui permet de repérer immédiatement une règle mal configurée, plutôt que de le découvrir en test manuel comme sur Belacom.

## 9. Plan de mise en œuvre par phases, avec porte de mesure obligatoire

Chaque phase ne démarre que si la précédente est mesurée sur un jeu d'évaluation golden set et ne régresse pas par rapport à la phase précédente — c'est la mesure du 5-6 août 2026 (comparaison agent vs pipeline procédural) qui a révélé le problème Belacom trop tard ; ici la mesure est une porte de passage, pas une vérification a posteriori.

| Phase | Contenu | Condition de passage à la phase suivante |
|---|---|---|
| Phase 0 | L2 seul, sans L3. Sert de ligne de base. | Golden set mesuré et versionné par tenant |
| Phase 1 | Garde métier (règles pré-requête) + appel L2 + évaluateur de verdict + 1 cycle de raffinement borné + nœuds terminaux | Sur le golden set : aucune régression de recall/précision par rapport à la Phase 0, et amélioration mesurable du taux de résolution des cas `PARTIEL` |
| Phase 2 | Superviseur multi-instances (`compare_instances`) | Uniquement si un tenant a un besoin réel de comparaison multi-collections, et seulement après stabilité de la Phase 1 en production sur au moins une instance |
| Phase 3 (non planifiée) | Tout enrichissement agentique supplémentaire (plans multi-étapes, outils métier additionnels) | À justifier individuellement par une mesure, jamais ajouté par défaut |

## 10. Ce qui n'est délibérément pas repris de Belacom

- Pas de boucle `max_steps` avec un LLM qui choisit un outil à chaque tour parmi un registre ouvert.
- Pas de parsing de décision par expression régulière sur une sortie JSON libre — toute sortie LLM utilisée pour une décision passe par un schéma structuré validé.
- Pas de `routing_rules` documentées mais non interprétées : le moteur de règles de la section 3 est un composant testé, pas un commentaire YAML.
- Pas de tools génériques `search_documents`/`get_document` exposés directement à un LLM de contrôle : l'unique porte d'entrée documentaire est `rag_query`, qui encapsule déjà la logique de couverture calculée par L2.
