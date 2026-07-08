"""
database/database.py

Low-level SQLite persistence layer for DataPilot AI.

Responsibilities:
- Manage the SQLite connection lifecycle (creation, safe commit,
  safe rollback, closing) via a context manager.
- Ensure the database file and its parent directory exist.
- Create the required schema automatically if it does not yet exist.

This module has NO knowledge of Slack, AI providers, or any specific
business feature (SQL explanation, cleaning, generation, etc.). It is
a generic, reusable persistence primitive that any future feature can
depend on. Domain-specific query logic (e.g. saving/reading command
history) belongs in higher-level repository modules such as
database/history.py, not here.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/datapilot.db"

_CREATE_COMMAND_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS command_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    command TEXT NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    execution_time REAL NOT NULL,
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT,
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEX_USER_ID = """
CREATE INDEX IF NOT EXISTS idx_command_history_user
ON command_history (user_id);
"""

_CREATE_INDEX_COMMAND = """
CREATE INDEX IF NOT EXISTS idx_command_history_command
ON command_history (command);
"""

_CREATE_INDEX_CREATED_AT = """
CREATE INDEX IF NOT EXISTS idx_command_history_created_at
ON command_history (created_at);
"""


class DatabaseError(Exception):
    """
    Raised when a database operation fails (connection errors, schema
    initialization failures, or query execution errors).

    This is the single exception type that callers of DatabaseManager
    need to catch; they do not need to know about sqlite3's internal
    exception hierarchy.
    """


class DatabaseManager:
    """
    Manages the SQLite connection lifecycle and schema for DataPilot
    AI's persistence layer.

    This class is intentionally generic and reusable: it knows how to
    open connections and ensure tables exist, but it has no knowledge
    of what the data means. Higher-level repository classes (e.g.
    HistoryRepository) use an instance of this class to perform their
    domain-specific reads and writes.

    Usage:
        db_manager = DatabaseManager()
        with db_manager.get_connection() as conn:
            conn.execute("SELECT 1")
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the database manager and ensure the schema exists.

        Args:
            db_path: Filesystem path to the SQLite database file.
                Defaults to "data/datapilot.db" relative to the
                current working directory. The parent directory is
                created automatically if it does not exist.

        Raises:
            DatabaseError: If the database file's parent directory
                cannot be created, or if schema initialization fails.
        """
        self._db_path: Path = Path(db_path or _DEFAULT_DB_PATH)

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception(
                "Failed to create database directory: %s", self._db_path.parent
            )
            raise DatabaseError(
                f"Could not create database directory '{self._db_path.parent}'."
            ) from exc

        logger.debug("DatabaseManager initialized with db_path=%s", self._db_path)
        self._initialize_schema()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Provide a managed SQLite connection as a context manager.

        Opens a new connection configured with row_factory=sqlite3.Row
        (so result rows support column-name access) and foreign key
        enforcement enabled. On successful exit from the `with` block,
        the transaction is committed. If an exception propagates out
        of the `with` block, the transaction is rolled back before the
        exception is re-raised. The connection is always closed,
        whether the block succeeded or failed.

        Yields:
            An open sqlite3.Connection ready for use.

        Raises:
            DatabaseError: If the connection cannot be opened, or if
                commit/rollback fails.
        """
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(str(self._db_path))
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as exc:
            logger.exception("Failed to open SQLite connection at %s", self._db_path)
            raise DatabaseError(
                f"Could not open database connection at '{self._db_path}'."
            ) from exc

        try:
            yield connection
            connection.commit()
        except Exception:
            logger.exception(
                "Error during database transaction; rolling back."
            )
            try:
                connection.rollback()
            except sqlite3.Error:
                logger.exception("Rollback failed after transaction error.")
            raise
        finally:
            try:
                connection.close()
            except sqlite3.Error:
                logger.exception("Failed to close database connection cleanly.")

    def _initialize_schema(self) -> None:
        """
        Create all required tables and indexes if they do not already
        exist.

        This is called automatically once, during __init__, so the
        database is always ready to use immediately after
        DatabaseManager is constructed. Table and index creation use
        "IF NOT EXISTS", so this is safe to run on every application
        startup.

        Raises:
            DatabaseError: If table or index creation fails.
        """
        try:
            with self.get_connection() as conn:
                conn.execute(_CREATE_COMMAND_HISTORY_TABLE)
                conn.execute(_CREATE_INDEX_USER_ID)
                conn.execute(_CREATE_INDEX_COMMAND)
                conn.execute(_CREATE_INDEX_CREATED_AT)
            logger.info(
                "Database schema initialized successfully at %s", self._db_path
            )
        except DatabaseError:
            # Already logged and wrapped by get_connection(); re-raise as-is.
            raise
        except sqlite3.Error as exc:
            logger.exception("Failed to initialize database schema.")
            raise DatabaseError("Could not initialize database schema.") from exc