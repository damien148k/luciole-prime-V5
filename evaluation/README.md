# Campagne de référence — avis MRAe (Beaumont Sud, Brissy)

## Ce qu'on mesure

Chaque remarque MRAe est posée à Luciole, et la réponse est comparée à la
réponse réellement rédigée par wpd dans son mémoire en réponse.

| mesure | calculée par | ce qu'elle dit |
|---|---|---|
| esquive | code (motif sans accents) | Luciole a-t-elle répondu sur le fond ? |
| bon tome | code | le bon document a-t-il été soumis au modèle ? |
| couverture, recherche B | trace query2 | le pipeline a-t-il détecté un manque et cherché à nouveau ? |
| faits / sources / invention | juge LLM (0 à 2) | proximité avec la réponse wpd |

Le juge LLM ne connaît pas le corpus. Il compare deux textes. Une note
« faits » basse peut donc signifier que wpd a utilisé une information hors
dossier (réunion, étude complémentaire). Ces cas sont listés en fin de
rapport pour relecture humaine.

## Préparer le jeu

Un fichier JSONL, une ligne par remarque. Voir `jeu_reference.exemple.jsonl`.

- `question` : la remarque MRAe, copiée telle quelle.
- `reponse_reference` : la réponse wpd, copiée telle quelle.
- `sources_reference` : les documents/pages que wpd cite. Laisser vide si
  wpd répond sans citer le dossier (esquive légitime attendue).

Ce fichier contient des données réelles de projet : il ne va pas dans le
dépôt. Le déposer dans le dossier `evaluation/` de l'instance (monté
dans le conteneur `luciole-rag-<instance>` sous `/app/evaluation/`).
Ne jamais déposer le mémoire en réponse wpd ni l'avis MRAe dans `data/` :
le watcher les indexerait et Luciole répondrait en recopiant wpd.

## Lancer

Un projet = une instance = un index. Lancer la campagne par projet, deux
fois de suite pour mesurer le plancher de bruit. Le script tourne dans le
conteneur `luciole-rag-<instance>` (V4 : plus de conteneur agent) et
appelle `POST /api/rag/query` en local ; le juge passe par Ollama.
Arrêter le watcher avant (`docker compose stop watcher`) pour libérer
la VRAM sur une carte 12 Go :

```bash
docker exec -e JEU=/app/evaluation/jeu_brissy.jsonl -e LABEL=brissy-1 -e CONSIGNE=/app/config/prompts/consigne_mrae.md luciole-rag-brissy python /app/evaluation/campagne_reference.py
```

```bash
docker exec -e JEU=/app/evaluation/jeu_brissy.jsonl -e LABEL=brissy-2 -e CONSIGNE=/app/config/prompts/consigne_mrae.md luciole-rag-brissy python /app/evaluation/campagne_reference.py
```

Sorties dans `evaluation/` : `campagne_<label>.jsonl` (détail par cas) et
`rapport_<label>.txt` (synthèse).

## Lire le résultat

1. Deux passages identiques au caractère près sur au moins 18 cas sur 20 :
   la mesure est fiable, sinon vérifier `temperature: 0` et `seed: 42`.
2. Esquives et bon tome d'abord : ce sont les défauts du retrieval.
3. Juge ensuite : « invention » à 0 ou 1 est le cas grave, à relire.
4. Le même jeu rejoué sur V4 donne l'écart V3 → V4, cas par cas.
