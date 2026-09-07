"""
Constantes du service watcher.

Regroupe les extensions supportées, les timeouts, les limites et
les valeurs d'énumération utilisées dans tout le module.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Extensions de fichiers autorisées par défaut
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({
    # Texte structuré
    ".pdf", ".docx", ".doc", ".odt",
    # Présentations
    ".pptx", ".ppt", ".odp",
    # Tableurs
    ".xlsx", ".xls", ".ods",
    # Texte brut et balisage
    ".txt", ".md", ".rst", ".csv", ".rtf",
    # Web / structuré
    ".html", ".htm", ".xml", ".json",
    # Messagerie
    ".msg", ".eml",
})

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".tif", ".tiff", ".svg",
    ".webp", ".ico",
})

# Extensions ignorées silencieusement (fichiers temporaires, systèmes, locks)
IGNORED_EXTENSIONS: frozenset[str] = frozenset({
    ".tmp", ".temp", ".swp", ".swo", ".swn",
    ".lock", ".lck", ".pid",
    ".bak", ".bak2", ".orig",
    ".ds_store", ".thumbs",
    "~",              # fichiers temporaires LibreOffice / Word
})

# Préfixes de noms de fichiers ignorés (fichiers temporaires)
IGNORED_FILENAME_PREFIXES: tuple[str, ...] = (
    "~$",    # Word / Excel temporaires
    ".~",    # LibreOffice temporaires
    "._",    # macOS metadata
)

# ─────────────────────────────────────────────────────────────────────────────
# Répertoires exclus par défaut
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache",
    "node_modules",
    ".venv", "venv", "env",
    "Thumbs.db", ".DS_Store",
})

# ─────────────────────────────────────────────────────────────────────────────
# Limites opérationnelles
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB: int = 9999         # Taille maximale d'un fichier à indexer (Mo)
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

MAX_QUEUE_DEPTH: int = 50_000        # Alerte si la queue dépasse ce seuil
MAX_SCAN_FILES: int = 10_000         # Limite par scan de réconciliation

# ─────────────────────────────────────────────────────────────────────────────
# Timeouts et intervalles (secondes)
# ─────────────────────────────────────────────────────────────────────────────

DEBOUNCE_SECONDS: float = 3.0           # Délai de debounce événements FS
STABILITY_CHECKS: int = 3               # Nombre de vérifications de stabilité fichier
STABILITY_INTERVAL: float = 2.0         # Intervalle entre vérifications (s)
WORKER_POLL_INTERVAL: float = 1.0       # Fréquence de poll de la queue par le worker
POLLING_OBSERVER_TIMEOUT: float = 5.0   # Intervalle watchdog PollingObserver
RECONCILE_INTERVAL: int = 300           # Scan de réconciliation toutes les 5 min

# ─────────────────────────────────────────────────────────────────────────────
# Retry et backoff
# ─────────────────────────────────────────────────────────────────────────────

RETRY_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_BASE: float = 60.0        # Délai initial entre tentatives (s)
RETRY_BACKOFF_MULTIPLIER: float = 2.0   # Facteur exponentiel
RETRY_MAX_BACKOFF: float = 3600.0       # Délai maximum (1 heure)

# ─────────────────────────────────────────────────────────────────────────────
# Base de données
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH: str = "/app/backups/watcher/watcher.db"
CONTENT_HASH_BLOCK_SIZE: int = 8 * 1024 * 1024   # Blocs de 8 Mo pour SHA-256

# ─────────────────────────────────────────────────────────────────────────────
# Statuts (miroir des enums dans models.py — pour usage en SQL direct)
# ─────────────────────────────────────────────────────────────────────────────

DOC_STATUS_PENDING: str = "pending"
DOC_STATUS_INDEXED: str = "indexed"
DOC_STATUS_ERROR: str = "error"
DOC_STATUS_DELETED: str = "deleted"

JOB_STATUS_PENDING: str = "pending"
JOB_STATUS_IN_PROGRESS: str = "in_progress"
JOB_STATUS_COMPLETED: str = "completed"
JOB_STATUS_FAILED: str = "failed"
JOB_STATUS_DEAD: str = "dead"

JOB_ACTION_UPSERT: str = "upsert"
JOB_ACTION_DELETE: str = "delete"
JOB_ACTION_MOVE: str = "move"

JOB_SOURCE_WATCHER: str = "watcher"
JOB_SOURCE_RESCAN: str = "rescan"
JOB_SOURCE_MANUAL: str = "manual"
JOB_SOURCE_STARTUP: str = "startup"

# Nombre de verrous SQLite consécutifs subis par le worker avant de passer le
# journal en WARNING et de signaler le service comme dégradé. Un worker vivant
# qui n'écrit plus doit être visible de l'extérieur.
LOCK_ERRORS_WARN_THRESHOLD: int = 5

DELETION_REASON_FILE_REMOVED: str = "file_removed"
DELETION_REASON_MANUAL: str = "manual"
DELETION_REASON_RECONCILE: str = "reconcile"
