"""
Initialisation et gestion de la base SQLite du watcher.

La base est stockée dans un volume Docker persistant (`./backups/watcher/`).
Elle utilise le mode WAL et des PRAGMA optimisés pour un accès concurrent
lecture/écriture entre le worker, le reconciler et l'API FastAPI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Ouvre une connexion SQLite avec les PRAGMA de production.

    À appeler une fois par thread : les connexions SQLite ne sont pas
    thread-safe. Chaque composant (worker, API, reconciler) maintient
    sa propre connexion.

    Args:
        db_path: Chemin absolu vers le fichier watcher.db.

    Returns:
        Connexion SQLite configurée, avec sqlite3.Row comme row_factory.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """
    Applique les PRAGMA SQLite pour un usage production.

    Durabilité + concurrence + performances mémoire.
    """
    pragmas: list[tuple[str, str]] = [
        # ── Durabilité et concurrence ──────────────────────────────────
        ("journal_mode",       "WAL"),
        # WAL : les lectures ne bloquent pas les écritures (worker ≠ API).

        ("synchronous",        "NORMAL"),
        # fsync à chaque checkpoint WAL uniquement — équilibre perf/durabilité.
        # Pas de perte de données en cas de crash OS (mais possible si coupure
        # électrique brutale sans UPS — acceptable en contexte entreprise).

        ("busy_timeout",       "5000"),
        # Attendre jusqu'à 5 s si la base est verrouillée avant de lever
        # sqlite3.OperationalError. Évite les échecs immédiats sous contention.

        ("foreign_keys",       "ON"),
        # Intégrité référentielle : document_chunks → documents (ON DELETE CASCADE).

        # ── Performances mémoire ───────────────────────────────────────
        ("cache_size",         "-64000"),
        # Cache de pages : valeur négative = kilo-octets → 64 Mo.

        ("temp_store",         "MEMORY"),
        # Tables temporaires en RAM (tris, sous-requêtes) — évite les fichiers tmp.

        ("mmap_size",          "268435456"),
        # Memory-mapped I/O : 256 Mo. Accélère les lectures séquentielles.

        # ── Contrôle taille WAL ────────────────────────────────────────
        ("journal_size_limit", "67108864"),
        # Auto-checkpoint quand le WAL dépasse 64 Mo.
    ]

    for pragma, value in pragmas:
        conn.execute(f"PRAGMA {pragma}={value}")
    conn.commit()


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialise la base watcher : crée le répertoire, le fichier SQLite,
    applique les PRAGMA de production, puis crée les tables si elles
    n'existent pas.

    Args:
        db_path: Chemin absolu vers watcher.db.

    Returns:
        Connexion SQLite initialisée et prête à l'emploi.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    _create_schema(conn)
    logger.info(f"Watcher DB initialisée : {db_path}")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Crée les tables et index si ils n'existent pas encore."""
    conn.executescript("""
        -- ─────────────────────────────────────────────────────────────────
        -- Documents indexés
        --
        -- source_id : UUID stable, généré à la première découverte.
        --             Ne change jamais, même après rename/move.
        -- current_path : chemin actuel du fichier (modifiable).
        -- deleted_at / deletion_reason : soft-delete avec audit.
        -- ─────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS documents (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id        TEXT    UNIQUE NOT NULL,
            current_path     TEXT    NOT NULL,
            file_name        TEXT    NOT NULL,
            file_extension   TEXT    NOT NULL DEFAULT '',
            file_size_bytes  INTEGER,
            quick_hash       TEXT,
            content_hash     TEXT,
            chunk_count      INTEGER NOT NULL DEFAULT 0,
            index_name       TEXT    NOT NULL DEFAULT 'documents',
            status           TEXT    NOT NULL DEFAULT 'pending',
            deleted_at       TEXT,
            deletion_reason  TEXT,
            first_indexed_at TEXT,
            last_indexed_at  TEXT,
            last_error       TEXT,
            retry_count      INTEGER NOT NULL DEFAULT 0,
            version          INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_id
            ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_documents_current_path
            ON documents(current_path);
        CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents(content_hash)
            WHERE content_hash IS NOT NULL;

        -- ─────────────────────────────────────────────────────────────────
        -- Audit des chunks (optionnel — non source de vérité)
        --
        -- La suppression cascade automatiquement quand le document parent
        -- est purgé via purge_tombstones().
        -- ─────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS document_chunks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id  TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
            chunk_id   TEXT NOT NULL,
            index_name TEXT NOT NULL,
            indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX        IF NOT EXISTS idx_chunks_source_id
            ON document_chunks(source_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_chunk_id
            ON document_chunks(chunk_id, index_name);

        -- ─────────────────────────────────────────────────────────────────
        -- File d'attente de jobs
        -- ─────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS jobs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id        TEXT    UNIQUE NOT NULL,
            file_path     TEXT    NOT NULL,
            action        TEXT    NOT NULL,
            old_path      TEXT,
            status        TEXT    NOT NULL DEFAULT 'pending',
            priority      INTEGER NOT NULL DEFAULT 0,
            source        TEXT             DEFAULT 'watcher',
            attempts      INTEGER NOT NULL DEFAULT 0,
            max_attempts  INTEGER NOT NULL DEFAULT 3,
            next_retry_at TEXT,
            error_message TEXT,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
            started_at    TEXT,
            completed_at  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_file_path
            ON jobs(file_path);
        CREATE INDEX IF NOT EXISTS idx_jobs_next_retry
            ON jobs(next_retry_at)
            WHERE next_retry_at IS NOT NULL;

        -- ─────────────────────────────────────────────────────────────────
        -- Journal des événements filesystem
        -- ─────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS watcher_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            file_path  TEXT NOT NULL,
            old_path   TEXT,
            timestamp  TEXT NOT NULL DEFAULT (datetime('now')),
            debounced  INTEGER      NOT NULL DEFAULT 0,
            job_id     TEXT,
            ignored    INTEGER      NOT NULL DEFAULT 0,
            reason     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON watcher_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_file_path
            ON watcher_events(file_path);

        -- ─────────────────────────────────────────────────────────────────
        -- Journal des erreurs et audit
        -- ─────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS audit_errors (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id        TEXT,
            file_path     TEXT,
            error_type    TEXT NOT NULL,
            error_message TEXT NOT NULL,
            stack_trace   TEXT,
            timestamp     TEXT NOT NULL DEFAULT (datetime('now')),
            resolved      INTEGER      NOT NULL DEFAULT 0,
            resolved_at   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_errors_timestamp
            ON audit_errors(timestamp);
        CREATE INDEX IF NOT EXISTS idx_errors_job_id
            ON audit_errors(job_id)
            WHERE job_id IS NOT NULL;
    """)
    conn.commit()


def purge_tombstones(conn: sqlite3.Connection, older_than_days: int = 90) -> int:
    """
    Supprime les documents soft-deleted datant de plus de `older_than_days` jours.

    La suppression CASCADE s'applique automatiquement sur `document_chunks`
    grâce à la contrainte FOREIGN KEY ON DELETE CASCADE.

    Args:
        conn: Connexion SQLite active.
        older_than_days: Âge minimum en jours pour purger un tombstone.

    Returns:
        Nombre de lignes supprimées.
    """
    cursor = conn.execute(
        """
        DELETE FROM documents
        WHERE status = 'deleted'
          AND deleted_at < datetime('now', :offset)
        """,
        {"offset": f"-{older_than_days} days"},
    )
    conn.commit()
    count = cursor.rowcount
    if count:
        logger.info(f"Tombstones purgés : {count} document(s) (> {older_than_days} jours)")
    return count


def log_audit_error(
    conn: sqlite3.Connection,
    error_type: str,
    error_message: str,
    job_id: str | None = None,
    file_path: str | None = None,
    stack_trace: str | None = None,
) -> None:
    """
    Insère une entrée dans `audit_errors`.

    Args:
        conn: Connexion SQLite active.
        error_type: Catégorie d'erreur (parse_error, embed_error, etc.).
        error_message: Message d'erreur lisible.
        job_id: Identifiant du job concerné (optionnel).
        file_path: Chemin du fichier concerné (optionnel).
        stack_trace: Traceback complet (optionnel).
    """
    conn.execute(
        """
        INSERT INTO audit_errors (job_id, file_path, error_type, error_message, stack_trace)
        VALUES (:job_id, :file_path, :error_type, :error_message, :stack_trace)
        """,
        {
            "job_id": job_id,
            "file_path": file_path,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
        },
    )
    conn.commit()
