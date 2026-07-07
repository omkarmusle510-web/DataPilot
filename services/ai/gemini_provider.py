"""
services/ai/gemini_provider.py

Concrete AIProvider implementation backed by Google's Gemini models,
using the official `google-genai` SDK.

This is the ONLY file in the project that imports the google-genai
SDK. No other module (analyzer.py, app.py, or any other provider)
should ever import `google.genai` directly.
"""

import logging
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from services.ai.base import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_TIMEOUT_MS = 30_000  # 30 seconds, expressed in milliseconds per SDK convention


class GeminiProvider(AIProvider):
    """
    AIProvider implementation using Google's Gemini models via the
    official google-genai SDK.

    Configuration is read from environment variables:
        GOOGLE_API_KEY: Required. API key for the Gemini Developer API.
        GEMINI_MODEL: Optional. Model name. Defaults to "gemini-2.5-flash".
        AI_TEMPERATURE: Optional. Sampling temperature. Defaults to 0.3.
        AI_TIMEOUT: Optional. Request timeout in seconds. Defaults to 30.

    Known limitation (as of the current google-genai SDK):
        The SDK accepts a timeout via types.HttpOptions(timeout=...),
        which this class applies at the client level. However, there
        are open upstream issues where this timeout is not always
        reliably enforced by the underlying HTTP transport in every
        SDK release. AI_TIMEOUT is therefore best-effort today. Should
        Google fix or change this behavior, only this file needs to be
        updated — the AIProvider interface and all callers are
        unaffected.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Initialize the Gemini provider with a configured genai Client.

        Args:
            api_key: Google AI Studio API key. Falls back to the
                GOOGLE_API_KEY environment variable if not provided.
            model: Gemini model name. Falls back to the GEMINI_MODEL
                environment variable, then to "gemini-2.5-flash".
            temperature: Sampling temperature. Falls back to the
                AI_TEMPERATURE environment variable, then to 0.3.
            timeout_seconds: Request timeout in seconds. Falls back to
                the AI_TIMEOUT environment variable, then to 30 seconds.
                See the class docstring for a known SDK limitation.

        Raises:
            ValueError: If no API key is available from either the
                argument or the environment.
        """
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in your "
                "environment or pass api_key explicitly."
            )

        self._model: str = model or os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)
        self._temperature: float = (
            temperature
            if temperature is not None
            else float(os.getenv("AI_TEMPERATURE", _DEFAULT_TEMPERATURE))
        )
        timeout_s = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("AI_TIMEOUT", _DEFAULT_TIMEOUT_MS / 1000))
        )
        self._timeout_ms: int = int(timeout_s * 1000)

        # --- Public provider metadata (part of the AIProvider contract) ---
        self.name: str = "Gemini"
        self.vendor: str = "Google"
        self.model: str = self._model

        self._client: genai.Client = genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )

        logger.debug(
            "GeminiProvider initialized. model=%s, temperature=%s, timeout_ms=%d",
            self._model,
            self._temperature,
            self._timeout_ms,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a text completion using the configured Gemini model.

        Args:
            system_prompt: Instructions defining the model's role and
                behavior, passed as the Gemini system instruction.
            user_prompt: The user-facing content the model should
                respond to.

        Returns:
            The generated text, stripped of leading/trailing whitespace.

        Raises:
            AIProviderError: If the Gemini API call fails for any reason
                (authentication, connection, rate limiting, malformed
                response, or an unexpected error), or if the model
                returns no usable text.
        """
        logger.info(
            "Requesting Gemini generation. model=%s, prompt_length=%d",
            self._model,
            len(user_prompt),
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._temperature,
                    max_output_tokens=1024,
                ),
            )
        except APIError as exc:
            logger.error("Gemini API returned an error: %s", exc)
            raise AIProviderError(
                "The Gemini AI service returned an error while generating "
                "a response."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception("Unexpected error while calling Gemini API.")
            raise AIProviderError(
                "An unexpected error occurred while contacting the Gemini "
                "AI service."
            ) from exc

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: "types.GenerateContentResponse") -> str:
        """
        Safely extract generated text from a Gemini response object.

        Args:
            response: The raw GenerateContentResponse returned by the
                google-genai SDK.

        Returns:
            The response text, stripped of leading/trailing whitespace.

        Raises:
            AIProviderError: If the response contains no usable text
                (e.g. blocked by safety filters, empty candidates, or a
                malformed response object).
        """
        text = getattr(response, "text", None)

        if not text or not text.strip():
            logger.error("Gemini response contained no usable text: %r", response)
            raise AIProviderError(
                "The Gemini AI service returned an empty response. This "
                "may be due to safety filtering or an internal error."
            )

        return text.strip()