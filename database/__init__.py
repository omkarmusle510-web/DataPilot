"""
database/__init__.py

Public interface for DataPilot AI's database package.

This package provides the persistence layer used by the rest of the
application:

- DatabaseManager (database.py): Low-level SQLite connection lifecycle
  management and automatic schema initialization. Knows nothing about
  what the stored data means.
- HistoryRepository (history.py): The single public API for reading
  and writing command execution history. All SQL statements in the
  project are confined to this class.

Callers outside this package should import exclusively from
`database` (this module), never from `database.database` or
`database.history` directly, so the internal module layout can evolve
without breaking consumers. For example:

    from database import DatabaseManager, HistoryRepository

    db_manager = DatabaseManager()
    history = HistoryRepository(db_manager)

This package has NO knowledge of Slack or any AI provider, and
contains no business logic — it is a generic, reusable persistence
layer that future features (CSV uploads, dataset reports, analytics,
dashboards) can build on without redesign.
"""

from .database import DatabaseError, DatabaseManager
from .history import HistoryRepository, HistoryRepositoryError

__all__ = [
    "DatabaseManager",
    "DatabaseError",
    "HistoryRepository",
    "HistoryRepositoryError",
]