"""
services/validator.py

Business logic for DataPilot AI's SQL validation feature.

Responsibilities:
- Validate incoming SQL query text for basic presence (non-empty).
- Construct the system/user prompts that instruct an AI model to act
  as a senior SQL reviewer and produce a structured validation report
  (syntax errors, warnings, best practices, and a final verdict) for
  a given query.
- Delegate the actual text generation to an injected AIProvider.
- Translate provider-level failures into a single, domain-specific
  exception type.

This module has NO knowledge of any concrete AI vendor (Gemini,
OpenAI, Claude, Groq, etc.) and NO knowledge of Slack. It depends only
on the AIProvider abstraction, which is supplied via constructor
injection. This satisfies the Dependency Inversion Principle: high-level
business logic (this module) depends on an abstraction, not on a
concrete implementation.

This service is NOT a SQL formatter (see services/cleaner.py), NOT a
SQL generator (see services/generator.py), and NOT a performance
advisor (see services/optimizer.py). It is a correctness reviewer: it
inspects a query for syntax errors, semantic mistakes, unsafe
operations, and best-practice violations, and reports on them.
"""

import logging
from typing import Final

from services.ai import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: Final[str] = (
    "You are a senior SQL reviewer who validates SQL queries for "
    "correctness and reports problems. You do not optimize queries "
    "for performance, you do not reformat their style, and you do not "
    "explain what a query does in plain English — you validate "
    "correctness only.\n\n"
    "Inspect the given SQL query for issues such as: syntax errors; "
    "missing commas; a missing FROM clause; missing or malformed WHERE "
    "operators; unclosed parentheses; reserved keyword misuse; unknown "
    "table or column aliases; duplicate aliases; ambiguous column "
    "references; incorrect GROUP BY usage; HAVING misuse; ORDER BY "
    "misuse; JOIN mistakes; missing ON clauses; NULL comparison "
    "mistakes (e.g. using = NULL instead of IS NULL); data type "
    "mismatches; unsafe DELETE statements (e.g. without a WHERE "
    "clause); unsafe UPDATE statements (e.g. without a WHERE clause); "
    "unsafe TRUNCATE statements; unsafe DROP statements; possible SQL "
    "injection patterns; use of SELECT *; vendor-specific syntax that "
    "may not be portable; ANSI SQL compatibility issues; and other "
    "best-practice violations.\n\n"
    "Every issue you report (whether an Error or a Warning) must "
    "include a severity level of exactly one of: High, Medium, or "
    "Low.\n\n"
    "Return your analysis as plain text only. Do not use markdown, do "
    "not use JSON, do not use code fences, do not explain that you are "
    "an AI, do not greet the user, and do not add any other "
    "commentary. Structure your entire response using exactly this "
    "layout:\n\n"
    "SQL Validation Report\n\n"
    "Overall Status\n"
    "<✅ VALID or ❌ INVALID>\n\n"
    "Overall Score\n"
    "<score>/10\n\n"
    "Errors\n\n"
    "1.\n\n"
    "Problem:\n"
    "<problem text>\n\n"
    "Severity:\n"
    "<High, Medium, or Low>\n\n"
    "Explanation:\n"
    "<explanation text>\n\n"
    "Suggested Fix:\n"
    "<suggested fix text>\n\n"
    "(repeat numbered entries for each additional error found; if the "
    "query is valid, write \"No Errors Found\" instead of a numbered "
    "list)\n\n"
    "Warnings\n\n"
    "1.\n\n"
    "Warning:\n"
    "<warning text>\n\n"
    "Reason:\n"
    "<reason text>\n\n"
    "Recommendation:\n"
    "<recommendation text>\n\n"
    "(repeat numbered entries for each additional warning found; if "
    "there are none, write \"No Warnings Found\")\n\n"
    "Best Practices\n"
    "• <best practice observation>\n"
    "• <best practice observation>\n"
    "• <best practice observation>\n\n"
    "Final Verdict\n"
    "<final verdict text>\n\n"
    "If the query is fully valid with no errors, still include the "
    "Warnings, Best Practices, and Final Verdict sections in full."
)


class SQLValidatorError(Exception):
    """
    Raised whenever the SQL validation service cannot produce a
    result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    AIProviderError or any underlying AI vendor's exception hierarchy.
    """


class SQLValidator:
    """
    Encapsulates the business logic needed to validate a SQL query for
    correctness and produce a structured validation report.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.

    This class is a correctness reviewer, not a formatter, generator,
    or performance advisor: it inspects a query for syntax errors,
    semantic mistakes, unsafe operations, and best-practice
    violations, and reports on them (see SQLCleaner, SQLGenerator, and
    SQLOptimizer for those other concerns).
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the validator with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, OpenAIProvider) responsible for actual
                text generation. This validator never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "SQLValidator initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def validate_sql(self, sql_query: str) -> str:
        """
        Validate a SQL query for correctness and produce a structured
        validation report.

        Args:
            sql_query: The raw SQL query text to validate.

        Returns:
            A plain-text validation report following the fixed
            "SQL Validation Report" structure: an overall status, an
            overall score, a numbered list of errors (each with a
            problem, severity, explanation, and suggested fix), a
            numbered list of warnings (each with a warning, reason,
            and recommendation), a list of best practices, and a
            final verdict.

        Raises:
            SQLValidatorError: If the query is empty/invalid, or if
                the underlying AI provider fails to generate a valid
                validation report for any reason.
        """
        cleaned_query = sql_query.strip()
        if not cleaned_query:
            logger.warning("SQL validation requested with empty query text.")
            raise SQLValidatorError("No SQL query was provided to validate.")

        logger.info(
            "Requesting SQL validation via provider=%s. query_length=%d chars",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            len(cleaned_query),
        )
        logger.debug("Full SQL query to validate: %s", cleaned_query)

        user_prompt = f"Validate this SQL query:\n\n{cleaned_query}"

        try:
            report = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to validate SQL query: %s", exc)
            raise SQLValidatorError(
                "The AI service was unable to validate this query. "
                "Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while validating SQL query via provider."
            )
            raise SQLValidatorError(
                "An unexpected error occurred while validating the query."
            ) from exc

        cleaned_report = report.strip()
        if not cleaned_report:
            logger.error("AI provider returned an empty validation report.")
            raise SQLValidatorError(
                "The AI service returned an empty validation report."
            )

        logger.info(
            "SQL validation completed successfully. report_length=%d chars",
            len(cleaned_report),
        )
        return cleaned_report