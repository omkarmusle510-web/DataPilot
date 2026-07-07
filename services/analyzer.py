"""
services/analyzer.py

Business logic for DataPilot AI's SQL explanation feature.

Responsibilities:
- Validate incoming SQL query text.
- Construct the system/user prompts that instruct an AI model to
  explain SQL queries in plain English.
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
    "You are a senior data engineer. Explain SQL queries in plain, "
    "simple English for someone who may not know SQL well. "
    "Describe what tables/columns are involved, what filtering or "
    "joining happens, and what the query ultimately returns. Keep it "
    "concise (3-6 sentences) and do not simply restate the raw SQL."
)


class SQLExplanationError(Exception):
    """
    Raised whenever the SQL explanation service cannot produce a result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    AIProviderError or any underlying AI vendor's exception hierarchy.
    """


class SQLAnalyzer:
    """
    Encapsulates the business logic needed to turn a raw SQL query
    string into a plain-English explanation.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the analyzer with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, OpenAIProvider) responsible for actual
                text generation. This analyzer never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "SQLAnalyzer initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def explain_sql(self, sql_query: str) -> str:
        """
        Generate a plain-English explanation of a SQL query.

        Args:
            sql_query: The raw SQL query text to explain.

        Returns:
            A plain-English explanation of the query.

        Raises:
            SQLExplanationError: If the query is empty/invalid, or if
                the underlying AI provider fails to generate a valid
                explanation for any reason.
        """
        cleaned_query = sql_query.strip()
        if not cleaned_query:
            logger.warning("SQL explanation requested with empty query text.")
            raise SQLExplanationError("No SQL query was provided to explain.")

        logger.info(
            "Requesting SQL explanation via provider=%s. query_length=%d chars",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            len(cleaned_query),
        )
        logger.debug("Full SQL query to explain: %s", cleaned_query)

        user_prompt = f"Explain this SQL query:\n\n{cleaned_query}"

        try:
            explanation = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to explain SQL query: %s", exc)
            raise SQLExplanationError(
                "The AI service was unable to explain this query. "
                "Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while explaining SQL query via provider."
            )
            raise SQLExplanationError(
                "An unexpected error occurred while explaining the query."
            ) from exc

        cleaned_explanation = explanation.strip()
        if not cleaned_explanation:
            logger.error("AI provider returned an empty explanation string.")
            raise SQLExplanationError("The AI service returned an empty explanation.")

        logger.info(
            "SQL explanation generated successfully. explanation_length=%d chars",
            len(cleaned_explanation),
        )
        return cleaned_explanation