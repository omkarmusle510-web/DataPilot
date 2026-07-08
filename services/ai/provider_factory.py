"""
services/ai/provider_factory.py

Factory responsible for selecting and constructing the active
AIProvider implementation based on environment configuration.

This is the ONLY module in the project (besides the concrete provider
files themselves) that needs to change when a new AI provider is
added. Callers (app.py, analyzer.py) never import concrete provider
classes directly — they call get_provider() and receive back an
AIProvider, remaining fully decoupled from which vendor is active.
"""

import logging
import os
from typing import Callable

from services.ai.base import AIProvider
from services.ai.gemini_provider import GeminiProvider
from services.ai.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "gemini"

# Registry mapping a provider name (as configured via AI_PROVIDER) to a
# zero-argument callable that constructs the corresponding AIProvider.
# To add a new provider (OpenAI, Claude, OpenRouter, DeepSeek, Ollama,
# etc.):
#   1. Create services/ai/<name>_provider.py implementing AIProvider.
#   2. Import its class above.
#   3. Add one entry to this dict.
# No other file in the project needs to change.
_PROVIDER_REGISTRY: dict[str, Callable[[], AIProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def get_provider(provider_name: str | None = None) -> AIProvider:
    """
    Construct and return the configured AIProvider implementation.

    Args:
        provider_name: Explicit provider name to use (e.g. "gemini",
            "groq"). If not provided, falls back to the AI_PROVIDER
            environment variable, then to "gemini" if that is also
            unset. Matching is case-insensitive.

    Returns:
        An instantiated AIProvider implementation ready for use.

    Raises:
        ValueError: If the resolved provider name does not match any
            entry in the provider registry.
        ValueError: If the selected provider's constructor fails due to
            missing configuration (e.g. a missing API key). This is
            intentionally not caught here, since a misconfigured
            provider is a startup-time error that should surface
            immediately and clearly, not be masked by the factory.
    """
    resolved_name = (provider_name or os.getenv("AI_PROVIDER", _DEFAULT_PROVIDER)).strip().lower()

    provider_constructor = _PROVIDER_REGISTRY.get(resolved_name)
    if provider_constructor is None:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        logger.error(
            "Unsupported AI_PROVIDER '%s'. Supported providers: %s",
            resolved_name,
            supported,
        )
        raise ValueError(
            f"Unsupported AI provider '{resolved_name}'. "
            f"Supported providers are: {supported}."
        )

    logger.info("Resolved AI provider: %s", resolved_name)
    return provider_constructor()