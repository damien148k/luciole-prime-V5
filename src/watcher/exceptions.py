"""
Exceptions métier du service watcher.
"""


class WatcherError(Exception):
    """Exception de base pour tous les erreurs du watcher."""


class FileNotStableError(WatcherError):
    """
    Levée quand un fichier n'a pas atteint une taille stable après les
    vérifications de stabilité (fichier en cours de copie ou d'écriture).
    """


class IndexingError(WatcherError):
    """
    Levée quand l'indexation d'un document dans Qdrant ou OpenSearch échoue.
    Encapsule l'erreur sous-jacente du pipeline d'ingestion.
    """


class CleanupError(WatcherError):
    """
    Levée quand la suppression des chunks d'un document dans Qdrant
    ou OpenSearch échoue.
    """


class ConfigurationError(WatcherError):
    """
    Levée quand la configuration du watcher est invalide ou incohérente
    (chemin surveillé inexistant, extension mal formée, etc.).
    """


class QueueError(WatcherError):
    """
    Levée quand une opération sur la JobQueue échoue (corruption SQLite,
    contrainte unique violée, etc.).
    """


class StateStoreError(WatcherError):
    """
    Levée quand une opération sur le StateStore échoue.
    """


class FileTooLargeError(WatcherError):
    """
    Levée quand un fichier dépasse la taille maximale configurée.
    """

    def __init__(self, path: str, size_bytes: int, max_bytes: int) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"Fichier trop volumineux : {path} "
            f"({size_bytes / 1024 / 1024:.1f} Mo > {max_bytes / 1024 / 1024:.0f} Mo max)"
        )
