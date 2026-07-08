"""
database/history.py

Repository layer for DataPilot AI's command execution history.

Responsibilities:
- Provide the single public API for reading and writing rows in the
  command_history table.
- Translate raw sqlite3 rows into plain Python dictionaries so no
  other part of the application ever needs to know sqlite3 exists.
- Wrap all sqlite3-level failures into a single, domain-specific
  exception type.

HistoryRepository is the ONLY layer in the project allowed to contain
SQL statements. Business services (SQLAnalyzer, SQLCleaner,
SQLGenerator, SQLOptimizer, SQLValidator) and app.py must never
construct SQL themselves — they call methods on this repository.

This module depends only on database.DatabaseManager, supplied via
constructor injection. It has NO knowledge of Slack or any AI
provider, and is designed to remain the single persistence entry
point as future features (CSV uploads, dataset reports, analytics,
dashboards) are added.
"""

import logging
import sqlite3
from typing import Any, Final

from database.database import DatabaseError, DatabaseManager

logger = logging.getLogger(__name__)

_DEFAULT_RECENT_LIMIT: Final[int] = 10
_DEFAULT_USER_LIMIT: Final[int] = 20
_DEFAULT_COMMAND_LIMIT: Final[int] = 20
_DEFAULT_FAILURES_LIMIT: Final[int] = 10

_INSERT_COMMAND_HISTORY = """
INSERT INTO command_history (
    user_id,
    command,
    input_text,
    output_text,
    provider,
    model,
    execution_time,
    success,
    error_message,
    duration_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_HISTORY = """
SELECT *
FROM command_history
ORDER BY created_at DESC
LIMIT ?;
"""

_SELECT_USER_HISTORY = """
SELECT *
FROM command_history
WHERE user_id = ?
ORDER BY created_at DESC
LIMIT ?;
"""

_SELECT_COMMAND_HISTORY = """
SELECT *
FROM command_history
WHERE command = ?
ORDER BY created_at DESC
LIMIT ?;
"""

_SELECT_RECENT_FAILURES = """
SELECT *
FROM command_history
WHERE success = 0
ORDER BY created_at DESC
LIMIT ?;
"""

_SELECT_TOTAL_COUNT = "SELECT COUNT(*) AS total FROM command_history;"

_SELECT_SUCCESS_COUNT = (
    "SELECT COUNT(*) AS successful FROM command_history WHERE success = 1;"
)

_SELECT_FAILURE_COUNT = (
    "SELECT COUNT(*) AS failed FROM command_history WHERE success = 0;"
)

_SELECT_AVERAGES = """
SELECT
    AVG(execution_time) AS avg_execution_time,
    AVG(duration_ms) AS avg_duration_ms
FROM command_history;
"""

_SELECT_COMMAND_COUNTS = """
SELECT command, COUNT(*) AS count
FROM command_history
GROUP BY command
ORDER BY count DESC;
"""

_DELETE_ALL_HISTORY = "DELETE FROM command_history;"

_DELETE_USER_HISTORY = "DELETE FROM command_history WHERE user_id = ?;"

_SELECT_HISTORY_EXISTS = "SELECT 1 FROM command_history LIMIT 1;"


class HistoryRepositoryError(Exception):
    """
    Raised when a HistoryRepository operation fails.

    This is the single exception type that callers (business services,
    app.py, or future features) need to catch; they never need to know
    about sqlite3's internal exception hierarchy or DatabaseError.
    """


class HistoryRepository:
    """
    Repository providing the single public API for persisting and
    querying command execution history.

    All SQL statements in the project are confined to this class.
    Every method returns plain Python types (dicts, lists, primitives)
    — never sqlite3.Row objects — so callers remain fully decoupled
    from the underlying storage engine.

    This repository is intentionally designed to remain the single
    persistence entry point as DataPilot AI grows to support CSV
    uploads, dataset cleaning reports, analytics, and dashboards: those
    features can either reuse command_history (e.g. logging a "clean_csv"
    command) or extend this repository with additional methods, without
    requiring any change to how DatabaseManager itself is used.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """
        Initialize the repository with an injected DatabaseManager.

        Args:
            db_manager: A configured DatabaseManager instance
                responsible for connection lifecycle and schema
                management. This repository never constructs its own
                DatabaseManager.
        """
        self._db: DatabaseManager = db_manager
        logger.debug("HistoryRepository initialized.")

    def save_command(
        self,
        user_id: str,
        command: str,
        input_text: str,
        output_text: str,
        provider: str,
        model: str,
        execution_time: float,
        success: bool,
        error_message: str | None,
        duration_ms: int,
    ) -> int:
        """
        Insert one command execution record into history.

        Args:
            user_id: Slack user ID (or other caller-supplied identity)
                who invoked the command.
            command: Command name (e.g. "explain", "clean", "generate",
                "optimize", "validate").
            input_text: The raw input submitted for the command.
            output_text: The result produced by the command. Should be
                an empty string (not None) if the command failed
                before producing output.
            provider: Name of the AI provider used (e.g. "Gemini").
            model: Model identifier used (e.g. "gemini-2.5-flash").
            execution_time: Business-level execution time, in seconds,
                as measured by the caller.
            success: Whether the command completed successfully.
            error_message: Error message if the command failed, or
                None if it succeeded.
            duration_ms: Total duration of the operation in
                milliseconds.

        Returns:
            The autoincremented integer ID of the newly inserted row.

        Raises:
            HistoryRepositoryError: If the insert fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                cursor = conn.execute(
                    _INSERT_COMMAND_HISTORY,
                    (
                        user_id,
                        command,
                        input_text,
                        output_text,
                        provider,
                        model,
                        execution_time,
                        1 if success else 0,
                        error_message,
                        duration_ms,
                    ),
                )
                record_id = cursor.lastrowid
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception(
                "Failed to save command history. user_id=%s, command=%s",
                user_id,
                command,
            )
            raise HistoryRepositoryError(
                "Could not save command history record."
            ) from exc

        logger.info(
            "Saved command history record. id=%d, user_id=%s, command=%s, success=%s",
            record_id,
            user_id,
            command,
            success,
        )
        return record_id

    def get_recent_history(self, limit: int = _DEFAULT_RECENT_LIMIT) -> list[dict[str, Any]]:
        """
        Retrieve the most recent command history records, across all
        users, ordered newest first.

        Args:
            limit: Maximum number of records to return. Defaults to 10.

        Returns:
            A list of dictionaries, each representing one history
            record, ordered by created_at descending.

        Raises:
            HistoryRepositoryError: If the query fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                rows = conn.execute(_SELECT_RECENT_HISTORY, (limit,)).fetchall()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Failed to fetch recent history. limit=%d", limit)
            raise HistoryRepositoryError(
                "Could not retrieve recent command history."
            ) from exc

        logger.debug("Fetched %d recent history record(s).", len(rows))
        return [self._row_to_dict(row) for row in rows]

    def get_user_history(
        self, user_id: str, limit: int = _DEFAULT_USER_LIMIT
    ) -> list[dict[str, Any]]:
        """
        Retrieve command history for a specific user, ordered newest
        first.

        Args:
            user_id: The user identifier to filter by.
            limit: Maximum number of records to return. Defaults to 20.

        Returns:
            A list of dictionaries, each representing one history
            record belonging to the given user, ordered by created_at
            descending.

        Raises:
            HistoryRepositoryError: If the query fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                rows = conn.execute(_SELECT_USER_HISTORY, (user_id, limit)).fetchall()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception(
                "Failed to fetch user history. user_id=%s, limit=%d", user_id, limit
            )
            raise HistoryRepositoryError(
                "Could not retrieve history for this user."
            ) from exc

        logger.debug(
            "Fetched %d history record(s) for user_id=%s.", len(rows), user_id
        )
        return [self._row_to_dict(row) for row in rows]

    def get_command_history(
        self, command: str, limit: int = _DEFAULT_COMMAND_LIMIT
    ) -> list[dict[str, Any]]:
        """
        Retrieve history for a specific command type, ordered newest
        first.

        Args:
            command: Command name to filter by (e.g. "explain",
                "clean", "generate", "optimize", "validate").
            limit: Maximum number of records to return. Defaults to 20.

        Returns:
            A list of dictionaries, each representing one history
            record for the given command, ordered by created_at
            descending.

        Raises:
            HistoryRepositoryError: If the query fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                rows = conn.execute(_SELECT_COMMAND_HISTORY, (command, limit)).fetchall()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception(
                "Failed to fetch command history. command=%s, limit=%d",
                command,
                limit,
            )
            raise HistoryRepositoryError(
                "Could not retrieve history for this command."
            ) from exc

        logger.debug(
            "Fetched %d history record(s) for command=%s.", len(rows), command
        )
        return [self._row_to_dict(row) for row in rows]

    def get_recent_failures(
        self, limit: int = _DEFAULT_FAILURES_LIMIT
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most recent failed command executions, ordered
        newest first. Useful for debugging and monitoring.

        Args:
            limit: Maximum number of records to return. Defaults to 10.

        Returns:
            A list of dictionaries, each representing one failed
            history record, ordered by created_at descending.

        Raises:
            HistoryRepositoryError: If the query fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                rows = conn.execute(_SELECT_RECENT_FAILURES, (limit,)).fetchall()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Failed to fetch recent failures. limit=%d", limit)
            raise HistoryRepositoryError(
                "Could not retrieve recent failed commands."
            ) from exc

        logger.debug("Fetched %d recent failure record(s).", len(rows))
        return [self._row_to_dict(row) for row in rows]

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute aggregate statistics over the entire command history.

        Returns:
            A dictionary with the following keys:
                total_commands (int): Total number of history records.
                successful_commands (int): Number of successful
                    executions.
                failed_commands (int): Number of failed executions.
                average_execution_time (float): Average execution_time
                    across all records, in seconds. 0.0 if there is no
                    history.
                average_duration_ms (float): Average duration_ms across
                    all records. 0.0 if there is no history.
                commands_by_type (dict[str, int]): Mapping of command
                    name to how many times it has been executed,
                    ordered from most to least frequent.
                most_used_command (str | None): The single most
                    frequently executed command name, or None if there
                    is no history yet.

        Raises:
            HistoryRepositoryError: If any statistics query fails.
        """
        try:
            with self._db.get_connection() as conn:
                total_row = conn.execute(_SELECT_TOTAL_COUNT).fetchone()
                success_row = conn.execute(_SELECT_SUCCESS_COUNT).fetchone()
                failure_row = conn.execute(_SELECT_FAILURE_COUNT).fetchone()
                averages_row = conn.execute(_SELECT_AVERAGES).fetchone()
                command_rows = conn.execute(_SELECT_COMMAND_COUNTS).fetchall()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Failed to compute history statistics.")
            raise HistoryRepositoryError(
                "Could not compute command history statistics."
            ) from exc

        commands_by_type: dict[str, int] = {
            row["command"]: row["count"] for row in command_rows
        }
        most_used_command = command_rows[0]["command"] if command_rows else None

        statistics: dict[str, Any] = {
            "total_commands": total_row["total"] if total_row else 0,
            "successful_commands": success_row["successful"] if success_row else 0,
            "failed_commands": failure_row["failed"] if failure_row else 0,
            "average_execution_time": (
                averages_row["avg_execution_time"] or 0.0 if averages_row else 0.0
            ),
            "average_duration_ms": (
                averages_row["avg_duration_ms"] or 0.0 if averages_row else 0.0
            ),
            "commands_by_type": commands_by_type,
            "most_used_command": most_used_command,
        }

        logger.info(
            "Computed history statistics. total=%d, successful=%d, failed=%d",
            statistics["total_commands"],
            statistics["successful_commands"],
            statistics["failed_commands"],
        )
        return statistics

    def history_exists(self) -> bool:
        """
        Check whether at least one command history record exists.

        Returns:
            True if the command_history table contains at least one
            row, False otherwise.

        Raises:
            HistoryRepositoryError: If the query fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                row = conn.execute(_SELECT_HISTORY_EXISTS).fetchone()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Failed to check whether history exists.")
            raise HistoryRepositoryError(
                "Could not determine whether command history exists."
            ) from exc

        exists = row is not None
        logger.debug("History exists check result: %s", exists)
        return exists

    def delete_history(self) -> int:
        """
        Delete every command history record.

        Returns:
            The number of rows deleted.

        Raises:
            HistoryRepositoryError: If the delete fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                cursor = conn.execute(_DELETE_ALL_HISTORY)
                deleted_count = cursor.rowcount
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Failed to delete all command history.")
            raise HistoryRepositoryError(
                "Could not delete command history."
            ) from exc

        logger.info("Deleted all command history. rows_deleted=%d", deleted_count)
        return deleted_count

    def delete_user_history(self, user_id: str) -> int:
        """
        Delete all command history records belonging to a specific
        user.

        Args:
            user_id: The user identifier whose history should be
                deleted.

        Returns:
            The number of rows deleted.

        Raises:
            HistoryRepositoryError: If the delete fails for any reason.
        """
        try:
            with self._db.get_connection() as conn:
                cursor = conn.execute(_DELETE_USER_HISTORY, (user_id,))
                deleted_count = cursor.rowcount
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception(
                "Failed to delete history for user_id=%s.", user_id
            )
            raise HistoryRepositoryError(
                "Could not delete history for this user."
            ) from exc

        logger.info(
            "Deleted command history for user_id=%s. rows_deleted=%d",
            user_id,
            deleted_count,
        )
        return deleted_count

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """
        Convert a sqlite3.Row into a plain Python dictionary.

        Args:
            row: A row object produced by a query executed on a
                connection configured with row_factory=sqlite3.Row.

        Returns:
            A dictionary mapping column names to their values,
            decoupling all callers from sqlite3-specific types.
        """
        return dict(row)
