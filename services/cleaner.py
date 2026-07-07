"""
services/cleaner.py

Business logic for DataPilot AI's SQL cleaning/formatting feature.

Responsibilities:
- Validate incoming SQL query text.
- Construct the system/user prompts that instruct an AI model to
  reformat SQL according to common style conventions, without
  altering the query's semantics.
- Delegate the actual text generation to an injected AIProvider.
- Translate provider-level failures into a single, domain-specific
  exception type.

This module has NO knowledge of any concrete AI vendor (Gemini,
OpenAI, Claude, Groq, etc.) and NO knowledge of Slack. It depends only
on the AIProvider abstraction, which is supplied via constructor
injection. This satisfies the Dependency Inversion Principle: high-level
business logic (this module) depends on an abstraction, not on a
concrete implementation.
"""

import logging
from typing import Final

from services.ai import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: Final[str] = (
    "You are a senior data engineer who formats SQL queries according "
    "to common SQL style conventions. Rewrite the given SQL query so "
    "that: all SQL keywords (SELECT, FROM, WHERE, JOIN, GROUP BY, "
    "ORDER BY, etc.) are uppercase; each major clause starts on its "
    "own line; nested subqueries and joined tables are properly "
    "indented; spacing around commas, operators, and parentheses is "
    "consistent. You must preserve the exact semantics of the query — "
    "do not add, remove, or reorder any logic, filters, joins, or "
    "columns. Return ONLY the cleaned SQL query, with no explanation, "
    "no commentary, and no markdown code fences."
)


class SQLCleanerError(Exception):
    """
    Raised whenever the SQL cleaning service cannot produce a result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    AIProviderError or any underlying AI vendor's exception hierarchy.
    """


class SQLCleaner:
    """
    Encapsulates the business logic needed to reformat a raw SQL query
    string according to common SQL style conventions.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the cleaner with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, OpenAIProvider) responsible for actual
                text generation. This cleaner never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "SQLCleaner initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def clean_sql(self, sql_query: str) -> str:
        """
        Reformat a SQL query according to common SQL style conventions.

        Args:
            sql_query: The raw SQL query text to clean/format.

        Returns:
            The cleaned SQL query, with uppercase keywords, consistent
            indentation, and consistent spacing, while preserving the
            original query's semantics.

        Raises:
            SQLCleanerError: If the query is empty/invalid, or if the
                underlying AI provider fails to generate a valid
                cleaned query for any reason.
        """
        cleaned_input = sql_query.strip()
        if not cleaned_input:
            logger.warning("SQL cleaning requested with empty query text.")
            raise SQLCleanerError("No SQL query was provided to clean.")

        logger.info(
            "Requesting SQL cleaning via provider=%s. query_length=%d chars",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            len(cleaned_input),
        )
        logger.debug("Full SQL query to clean: %s", cleaned_input)

        user_prompt = f"Clean and format this SQL query:\n\n{cleaned_input}"

        try:
            cleaned_output = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to clean SQL query: %s", exc)
            raise SQLCleanerError(
                "The AI service was unable to clean this query. "
                "Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while cleaning SQL query via provider."
            )
            raise SQLCleanerError(
                "An unexpected error occurred while cleaning the query."
            ) from exc

        cleaned_result = cleaned_output.strip()
        if not cleaned_result:
            logger.error("AI provider returned an empty cleaned query string.")
            raise SQLCleanerError("The AI service returned an empty cleaned query.")

        logger.info(
            "SQL cleaning completed successfully. result_length=%d chars",
            len(cleaned_result),
        )
        return cleaned_result