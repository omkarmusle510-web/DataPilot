"""
services/analyzer.py

Business logic for DataPilot AI's SQL explanation feature.

Responsibilities:
- Communicate with the OpenAI API.
- Construct prompts that instruct the model to explain SQL queries in
  plain English.
- Handle all OpenAI-related failure modes and translate them into a
  single, safe, user-facing exception type.

This module has NO knowledge of Slack. It is a pure service layer that
could be reused by any interface (Slack, CLI, HTTP API, etc.) without
modification.
"""

import logging
import os
from typing import Final

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
)
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

_DEFAULT_MODEL: Final[str] = "gpt-4o-mini"

_SYSTEM_PROMPT: Final[str] = ("""
You are DataPilot AI, a Senior Data Engineer.

Your job is to explain SQL queries clearly.

For every SQL query provide:

1. Purpose (one sentence)

2. Step-by-step execution

3. Tables involved

4. Columns used

5. Filters

6. Joins

7. Aggregations

8. Sorting

9. Final output

10. Performance notes if applicable.

Explain in beginner-friendly English using bullet points.

Never simply repeat the SQL.
"""
)


class SQLExplanationError(Exception):
    """
    Raised whenever the SQL explanation service cannot produce a result.

    Callers (e.g. Slack event handlers) only need to catch this single
    exception type; they do not need to know about OpenAI's internal
    exception hierarchy.
    """


class SQLAnalyzer:
    """
    Encapsulates all logic needed to turn a raw SQL query string into a
    plain-English explanation using an OpenAI chat model.

    This class is intentionally Slack-agnostic: it accepts a string in
    and returns a string out (or raises SQLExplanationError), so it can
    be unit-tested or reused outside of the Slack context entirely.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """
        Initialize the analyzer with a configured OpenAI client.

        Args:
            api_key: OpenAI API key. Falls back to the OPENAI_API_KEY
                environment variable if not provided.
            model: OpenAI chat model name. Falls back to the OPENAI_MODEL
                environment variable, then to a hardcoded default.

        Raises:
            ValueError: If no API key is available from either the
                argument or the environment.
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your "
                "environment or pass api_key explicitly."
            )

        self._model: str = model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
        self._client: OpenAI = OpenAI(api_key=resolved_key)

        logger.debug("SQLAnalyzer initialized with model=%s", self._model)

    def explain_sql(self, sql_query: str) -> str:
        """
        Generate a plain-English explanation of a SQL query.

        Args:
            sql_query: The raw SQL query text to explain.

        Returns:
            A plain-English explanation of the query.

        Raises:
            SQLExplanationError: If the query is empty/invalid, or if the
                OpenAI API call fails for any reason (authentication,
                rate limiting, connection issues, or an unexpected API
                error), or if the API response is empty/malformed.
        """
        cleaned_query = sql_query.strip()
        if not cleaned_query:
            raise SQLExplanationError("No SQL query was provided to explain.")

        logger.info("Requesting SQL explanation. Query length=%d chars", len(cleaned_query))

        try:
            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Explain this SQL query:\n\n{cleaned_query}",
                    },
                ],
                temperature=0.3,
                max_tokens=400,
            )
        except AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", exc)
            raise SQLExplanationError(
                "Authentication with the AI service failed. Please check "
                "the API key configuration."
            ) from exc
        except RateLimitError as exc:
            logger.error("OpenAI rate limit exceeded: %s", exc)
            raise SQLExplanationError(
                "The AI service is currently rate-limited. Please try "
                "again shortly."
            ) from exc
        except APIConnectionError as exc:
            logger.error("Failed to connect to OpenAI API: %s", exc)
            raise SQLExplanationError(
                "Could not connect to the AI service. Please check your "
                "network connection."
            ) from exc
        except APIError as exc:
            logger.error("OpenAI API returned an error: %s", exc)
            raise SQLExplanationError(
                "The AI service returned an error while explaining the query."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception("Unexpected error while explaining SQL query.")
            raise SQLExplanationError(
                "An unexpected error occurred while explaining the query."
            ) from exc

        return self._extract_explanation(response)

    @staticmethod
    def _extract_explanation(response: ChatCompletion) -> str:
        """
        Safely extract the explanation text from an OpenAI ChatCompletion.

        Args:
            response: The raw ChatCompletion object returned by the
                OpenAI SDK.

        Returns:
            The explanation text, stripped of leading/trailing whitespace.

        Raises:
            SQLExplanationError: If the response contains no usable
                content (missing choices, missing message, or empty text).
        """
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            logger.error("OpenAI response missing expected structure: %r", response)
            raise SQLExplanationError(
                "The AI service returned an empty or malformed response."
            ) from exc

        if not content or not content.strip():
            raise SQLExplanationError("The AI service returned an empty explanation.")

        return content.strip()