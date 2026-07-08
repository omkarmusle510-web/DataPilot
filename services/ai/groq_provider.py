"""
services/ai/groq_provider.py

Concrete AIProvider implementation backed by Groq's hosted LLMs, using
the official `groq` Python SDK.

This is the ONLY file in the project that imports the groq SDK. No
other module (analyzer.py, app.py, or any other provider) should ever
import `groq` directly.
"""

import logging
import os

from groq import (
    Groq,
    APIError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from services.ai.base import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_TIMEOUT_SECONDS = 30.0


class GroqProvider(AIProvider):
    """
    AIProvider implementation using Groq's hosted LLMs via the
    official groq Python SDK.

    Configuration is read from environment variables:
        GROQ_API_KEY: Required. API key for the Groq API.
        GROQ_MODEL: Optional. Model name. Defaults to
            "llama-3.3-70b-versatile".
        AI_TEMPERATURE: Optional. Sampling temperature, shared with
            other providers. Defaults to 0.3.
        AI_TIMEOUT: Optional. Request timeout in seconds, shared with
            other providers. Defaults to 30.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Initialize the Groq provider with a configured Groq client.

        Args:
            api_key: Groq API key. Falls back to the GROQ_API_KEY
                environment variable if not provided.
            model: Groq model name. Falls back to the GROQ_MODEL
                environment variable, then to
                "llama-3.3-70b-versatile".
            temperature: Sampling temperature. Falls back to the
                AI_TEMPERATURE environment variable, then to 0.3.
            timeout_seconds: Request timeout in seconds. Falls back to
                the AI_TIMEOUT environment variable, then to 30
                seconds.

        Raises:
            ValueError: If no API key is available from either the
                argument or the environment.
        """
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in your "
                "environment or pass api_key explicitly."
            )

        self._model: str = model or os.getenv("GROQ_MODEL", _DEFAULT_MODEL)
        self._temperature: float = (
            temperature
            if temperature is not None
            else float(os.getenv("AI_TEMPERATURE", _DEFAULT_TEMPERATURE))
        )
        self._timeout_seconds: float = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("AI_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS))
        )

        # --- Public provider metadata (part of the AIProvider contract) ---
        self.name: str = "Groq"
        self.vendor: str = "Groq"
        self.model: str = self._model

        self._client: Groq = Groq(
            api_key=resolved_key,
            timeout=self._timeout_seconds,
        )

        logger.debug(
            "GroqProvider initialized. model=%s, temperature=%s, timeout_seconds=%s",
            self._model,
            self._temperature,
            self._timeout_seconds,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a text completion using the configured Groq model.

        Args:
            system_prompt: Instructions defining the model's role and
                behavior, passed as the system message.
            user_prompt: The user-facing content the model should
                respond to.

        Returns:
            The generated text, stripped of leading/trailing whitespace.

        Raises:
            AIProviderError: If the Groq API call fails for any reason
                (authentication, connection, rate limiting, a
                non-2xx status response, or an unexpected error), or
                if the model returns no usable text.
        """
        logger.info(
            "Requesting Groq generation. model=%s, prompt_length=%d",
            self._model,
            len(user_prompt),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_completion_tokens=1024,
            )
        except AuthenticationError as exc:
            logger.error("Groq authentication failed: %s", exc)
            raise AIProviderError(
                "Authentication with the Groq AI service failed. Please "
                "check the API key configuration."
            ) from exc
        except RateLimitError as exc:
            logger.error("Groq rate limit exceeded: %s", exc)
            raise AIProviderError(
                "The Groq AI service is currently rate-limited. Please "
                "try again shortly."
            ) from exc
        except APIConnectionError as exc:
            logger.error("Failed to connect to Groq API: %s", exc)
            raise AIProviderError(
                "Could not connect to the Groq AI service. Please check "
                "your network connection."
            ) from exc
        except APIStatusError as exc:
            logger.error(
                "Groq API returned a non-success status: status_code=%s, %s",
                getattr(exc, "status_code", "unknown"),
                exc,
            )
            raise AIProviderError(
                "The Groq AI service returned an error while generating "
                "a response."
            ) from exc
        except APIError as exc:
            logger.error("Groq API returned an error: %s", exc)
            raise AIProviderError(
                "The Groq AI service returned an error while generating "
                "a response."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception("Unexpected error while calling Groq API.")
            raise AIProviderError(
                "An unexpected error occurred while contacting the Groq "
                "AI service."
            ) from exc

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: object) -> str:
        """
        Safely extract generated text from a Groq chat completion
        response.

        Args:
            response: The raw ChatCompletion object returned by the
                groq SDK.

        Returns:
            The response text, stripped of leading/trailing whitespace.

        Raises:
            AIProviderError: If the response contains no usable text
                (e.g. empty choices, missing message, or a malformed
                response object).
        """
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            logger.error("Groq response missing expected structure: %r", response)
            raise AIProviderError(
                "The Groq AI service returned an empty or malformed "
                "response."
            ) from exc

        if not content or not content.strip():
            logger.error("Groq response contained no usable text: %r", response)
            raise AIProviderError(
                "The Groq AI service returned an empty response."
            )

        return content.strip()