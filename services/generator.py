"""
services/generator.py

Business logic for DataPilot AI's natural-language-to-SQL generation
feature.

Responsibilities:
- Validate incoming natural language requests.
- Construct the system/user prompts that instruct an AI model to
  generate syntactically correct ANSI SQL from a plain-English
  description, using reasonable generic schema assumptions when no
  schema information is provided.
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
    "You are a senior data engineer who converts natural language "
    "requests into syntactically correct ANSI SQL. Follow these rules "
    "strictly:\n"
    "1. Generate only syntactically correct, standard ANSI SQL. Never "
    "invent unsupported syntax or vendor-specific extensions unless "
    "the request explicitly names a specific database system.\n"
    "2. Prefer readability: use uppercase SQL keywords (SELECT, FROM, "
    "WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, etc.), and use proper, "
    "consistent indentation with each major clause on its own line.\n"
    "3. If the request does not specify table or column names, assume "
    "a reasonable, generic relational schema using common, "
    "conventional names such as employees, customers, orders, "
    "products, and their obvious columns (e.g. employee_name, salary, "
    "order_date, product_name, price). Do not invent unusual, overly "
    "specific, or bizarre schema names.\n"
    "4. Return ONLY the SQL query itself. Do not include markdown "
    "formatting, do not wrap the output in ``` code fences, do not "
    "explain the query, do not include comments inside or around the "
    "SQL, do not apologize, do not greet the user, and do not add any "
    "prose before or after the query.\n"
    "5. Terminate the query with a semicolon."
)


class SQLGeneratorError(Exception):
    """
    Raised whenever the SQL generation service cannot produce a result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    AIProviderError or any underlying AI vendor's exception hierarchy.
    """


class SQLGenerator:
    """
    Encapsulates the business logic needed to convert a natural
    language request into a syntactically correct ANSI SQL query.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the generator with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, OpenAIProvider) responsible for actual
                text generation. This generator never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "SQLGenerator initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def generate_sql(self, user_request: str) -> str:
        """
        Generate a syntactically correct ANSI SQL query from a natural
        language request.

        Args:
            user_request: The plain-English description of the desired
                query (e.g. "Show the top 10 highest paid employees").

        Returns:
            A syntactically correct ANSI SQL query string, with
            uppercase keywords and consistent indentation, containing
            no markdown, commentary, or prose.

        Raises:
            SQLGeneratorError: If the request is empty/whitespace-only,
                or if the underlying AI provider fails to generate a
                valid SQL query for any reason.
        """
        cleaned_request = user_request.strip()
        if not cleaned_request:
            logger.warning("SQL generation requested with empty request text.")
            raise SQLGeneratorError("No request was provided to generate SQL from.")

        logger.info(
            "Requesting SQL generation via provider=%s. request_length=%d chars",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            len(cleaned_request),
        )
        logger.debug("Full natural language request: %s", cleaned_request)

        user_prompt = f"Generate a SQL query for this request:\n\n{cleaned_request}"

        try:
            generated_sql = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to generate SQL query: %s", exc)
            raise SQLGeneratorError(
                "The AI service was unable to generate a query for this "
                "request. Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while generating SQL query via provider."
            )
            raise SQLGeneratorError(
                "An unexpected error occurred while generating the query."
            ) from exc

        cleaned_sql = generated_sql.strip()
        if not cleaned_sql:
            logger.error("AI provider returned an empty generated SQL string.")
            raise SQLGeneratorError("The AI service returned an empty SQL query.")

        logger.info(
            "SQL generation completed successfully. generated_sql_length=%d chars",
            len(cleaned_sql),
        )
        return cleaned_sql