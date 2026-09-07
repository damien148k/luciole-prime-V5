"""
Watcher — Service de surveillance de fichiers pour Luciole Prime.

Détecte les changements dans les dossiers surveillés et déclenche
la mise à jour incrémentale de l'index RAG (Qdrant + OpenSearch).

Composants principaux :
- FileWatcher   : surveillance du filesystem (watchdog PollingObserver)
- JobQueue      : file d'attente persistante (SQLite)
- IndexWorker   : traitement des jobs d'indexation
- StateStore    : état des documents indexés
- Reconciler    : scan périodique de réconciliation
- WatcherService: orchestrateur principal
"""
