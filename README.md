# Luciole Prime V5

Refonte complète de Luciole : plateforme d'assistant IA/RAG souveraine, offline et adaptable à tout cas d'usage métier, conçue pour interroger uniquement une base documentaire contrôlée afin de limiter les hallucinations et protéger la donnée.

## Philosophie de conception

- **Neutralité et adaptabilité** : aucun paramètre métier n'est figé dans le code. Le chunking, le reranking, et les autres réglages du pipeline RAG sont exposés comme configuration par instance/cas d'usage (via l'UI admin, à venir), pas comme constantes codées en dur.
- **RAG-only par défaut** : le système répond en s'appuyant exclusivement sur les documents ingérés et sourcés, pour limiter les hallucinations et garder le contrôle sur la donnée.
- **Construction incrémentale et vérifiée** : chaque couche est validée par un jeu de test avant que la couche suivante ne s'appuie dessus. On ne repart pas d'une génération de code non supervisée pour l'ensemble de la pile.

## Feuille de route architecturale

Les couches sont construites dans cet ordre, chacune sur la précédente :

1. **RAG & ingestion** (fondation, en cours) — construit from scratch selon les bonnes pratiques actuelles : ingestion structurée par type de document, chunking et reranking configurables, recherche hybride (dense + BM25) avec fusion RRF, métadonnées de traçabilité par chunk, journal d'audit.
2. **Agent / harnais / boucle / graph** — couche de décision et d'orchestration construite au-dessus du RAG validé. Reconstruite entièrement, avec un graphe d'état pensé comme pilote de la boucle agentique plutôt qu'une boucle implicite seule.
3. **Interface de chat** — interface utilisateur finale, une fois la pile agent+RAG opérationnelle et testée.
4. **Interface d'administration** — pilotage des paramètres RAG (chunking, reranking, sources, etc.), des profils d'agents, et supervision/traçabilité.

## Démarrage rapide (installation en ligne)

Prérequis : Docker Desktop installé et démarré, PC connecté à internet.
Modèle V4 : un dossier de travail = une instance = un index. Pour tester
un second projet, relancer `INSTALL.ps1` avec un autre nom d'instance
dans le même dossier (les données du précédent sont alors effacées).

```powershell
.\INSTALL.ps1
```

Le script détecte la VRAM disponible (`nvidia-smi`) et choisit la
fenêtre de contexte en conséquence (32768 sur une carte 20 Go et plus,
16384 en dessous). Forcer un profil : `.\INSTALL.ps1 -GpuProfile 3080ti`
(ou `a5000`, `cpu`).

Ensuite, déposer les documents dans `data\` : le watcher les indexe
automatiquement. Gestion courante : `.\MANAGE.ps1 -Action status|logs|health`.
Campagne de test : voir `evaluation\README.md`.

Une fois l'installation terminée, les interfaces sont servies par
l'instance : chat, recueil d'avis et administration (supervision du
watcher, gestion des index, évaluation RAGAS). Leurs adresses sont
rappelées à la fin de l'installation et par `.\MANAGE.ps1 -Action urls`.

## Installation hors-ligne (air-gap)

`INSTALL.ps1` télécharge images et modèles depuis internet. Pour une
machine isolée, le package se prépare en deux temps.

Sur une machine connectée, avec Docker démarré :

```powershell
.\PREPARE_OFFLINE.ps1 -GpuProfile a5000
```

Le script produit `offline_package\` (environ 22 Go, mesuré) : images
Docker exportées et compressées, image applicative construite, caches
Ollama et HuggingFace, code source et scripts. Prévoir 30 Go libres
pendant la préparation. Les modèles déjà présents dans `models\` sont
réutilisés tels quels, sans nouveau téléchargement. Le profil GPU est
celui de la machine **cible**, car il détermine le modèle LLM embarqué.

Copier le dossier sur la machine isolée, de préférence sur un disque
interne, puis :

```powershell
.\INSTALL_OFFLINE.ps1
```

Ce script vérifie le package, charge les images, puis appelle
`INSTALL.ps1 -Offline`, qui crée l'instance sans rien construire ni
télécharger. Le chargement des images prend une dizaine de minutes,
l'essentiel pour l'image applicative. Si une pièce manque, il s'arrête en le disant : sur une
machine isolée, rien ne pourra la rattraper ensuite.

## Structure du dépôt

```
src/
  ingestion/   # Pipeline d'ingestion documentaire (parsing, chunking, embedding)
  rag/         # Récupération, reranking, fusion, génération
docs/          # Documentation d'architecture et de décisions
configs/       # Configurations par instance/cas d'usage
tests/         # Jeux d'évaluation et tests de non-régression
```

## État actuel

Squelette initial du dépôt. Aucune logique métier n'est encore implémentée — la phase en cours est la conception et l'implémentation de la fondation RAG/ingestion.
