"""
services/optimizer.py

Business logic for DataPilot AI's SQL performance optimization
feature.

Responsibilities:
- Validate incoming SQL query text.
- Construct the system/user prompts that instruct an AI model to act
  as a senior database performance engineer and produce a structured
  optimization report (issues, reasoning, improved SQL, expected
  benefit) for a given query.
- Delegate the actual text generation to an injected AIProvider.
- Translate provider-level failures into a single, domain-specific
  exception type.

This module has NO knowledge of any concrete AI vendor (Gemini,
OpenAI, Claude, Groq, etc.) and NO knowledge of Slack. It depends only
on the AIProvider abstraction, which is supplied via constructor
injection. This satisfies the Dependency Inversion Principle: high-level
business logic (this module) depends on an abstraction, not on a
concrete implementation.

This service is NOT a SQL formatter (see services/cleaner.py) and NOT
a SQL generator (see services/generator.py). It is a SQL performance
advisor: given an existing query, it returns a structured analysis of
performance issues and recommended improvements.
"""

import logging
from typing import Final

from services.ai import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: Final[str] = (
    "You are a senior database performance engineer who reviews SQL "
    "queries for performance issues and produces a structured "
    "optimization report. Inspect the given SQL query for problems "
    "such as: SELECT *; missing WHERE clauses; inefficient or "
    "unnecessary JOINs; cartesian joins; nested subqueries that could "
    "be simplified; missing indexes; functions applied to indexed "
    "columns (non-sargable predicates); ORDER BY without LIMIT; "
    "DISTINCT misuse; inefficient GROUP BY usage; leading-wildcard "
    "LIKE '%value%' patterns; unnecessary selected columns; duplicate "
    "or redundant filtering; and any other performance bottlenecks.\n\n"
    "For every issue found, provide:\n"
    "- Issue: a short description of the specific problem.\n"
    "- Reason: why it is a problem for performance.\n"
    "- Improved Query: a corrected or improved version of the "
    "relevant SQL.\n"
    "- Expected Benefit: the expected performance improvement.\n\n"
    "If no meaningful optimization opportunities exist, state exactly: "
    "\"No significant optimization opportunities found.\"\n\n"
    "Return your analysis as plain text only. Do not use markdown, "
    "do not use JSON, do not use code fences, do not explain that you "
    "are an AI, do not greet the user, and do not add any other "
    "commentary. Structure your entire response using exactly this "
    "layout:\n\n"
    "SQL Performance Analysis\n\n"
    "Overall Rating:\n"
    "<rating>/10\n\n"
    "Optimization Opportunities\n\n"
    "1.\n\n"
    "Issue:\n"
    "<issue text>\n\n"
    "Reason:\n"
    "<reason text>\n\n"
    "Improved Query:\n"
    "<improved SQL>\n\n"
    "Expected Benefit:\n"
    "<expected benefit text>\n\n"
    "(repeat numbered entries for each additional issue found)\n\n"
    "Final Recommendation:\n"
    "<final recommendation text>"
)


class SQLOptimizerError(Exception):
    """
    Raised whenever the SQL optimization service cannot produce a
    result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    AIProviderError or any underlying AI vendor's exception hierarchy.
    """


class SQLOptimizer(object):
    """
    Encapsulates the business logic needed to analyze a SQL query for
    performance issues and produce a structured optimization report.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.

    This class is a performance advisor, not a formatter or generator:
    it inspects an existing query and reports on its efficiency, it
    does not rewrite the query's style (see SQLCleaner) and does not
    produce new queries from natural language (see SQLGenerator).
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the optimizer with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, OpenAIProvider) responsible for actual
                text generation. This optimizer never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "SQLOptimizer initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def optimize_sql(self, sql_query: str) -> str:
        """
        Analyze a SQL query for performance issues and produce a
        structured optimization report.

        Args:
            sql_query: The raw SQL query text to analyze.

        Returns:
            A plain-text optimization report following the fixed
            "SQL Performance Analysis" structure: an overall rating,
            a numbered list of optimization opportunities (each with
            an issue, reason, improved query, and expected benefit),
            and a final recommendation. If no issues are found, the
            report states that no significant optimization
            opportunities were found.

        Raises:
            SQLOptimizerError: If the query is empty/invalid, or if
                the underlying AI provider fails to generate a valid
                optimization report for any reason.
        """
        cleaned_query = sql_query.strip()
        if not cleaned_query:
            logger.warning("SQL optimization requested with empty query text.")
            raise SQLOptimizerError("No SQL query was provided to optimize.")

        logger.info(
            "Requesting SQL optimization via provider=%s. query_length=%d chars",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            len(cleaned_query),
        )
        logger.debug("Full SQL query to optimize: %s", cleaned_query)

        user_prompt = f"Analyze this SQL query for performance issues:\n\n{cleaned_query}"

        try:
            report = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to optimize SQL query: %s", exc)
            raise SQLOptimizerError(
                "The AI service was unable to analyze this query. "
                "Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while optimizing SQL query via provider."
            )
            raise SQLOptimizerError(
                "An unexpected error occurred while analyzing the query."
            ) from exc

        cleaned_report = report.strip()
        if not cleaned_report:
            logger.error("AI provider returned an empty optimization report.")
            raise SQLOptimizerError(
                "The AI service returned an empty optimization report."
            )

        logger.info(
            "SQL optimization completed successfully. report_length=%d chars",
            len(cleaned_report),
        )
        return cleaned_report