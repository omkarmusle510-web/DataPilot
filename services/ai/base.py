"""
services/ai/base.py

Defines the abstract contract that every AI provider in DataPilot AI
must implement, plus the shared exception type used to report provider
failures.

This module has NO dependency on any specific AI SDK (no OpenAI, no
google-genai, etc.). It exists purely to decouple business logic
(services/analyzer.py) from any single AI vendor, so providers can be
added or swapped by only touching files inside services/ai/.
"""

from abc import ABC, abstractmethod


class AIProviderError(Exception):
    """
    Raised by any AIProvider implementation when it cannot fulfill a
    generation request.

    This is the single exception type that provider-agnostic callers
    (e.g. services/analyzer.py) need to catch. Each concrete provider
    is responsible for catching its own SDK-specific exceptions and
    re-raising them as AIProviderError, so callers never need to know
    about OpenAI's, Google's, or any other vendor's exception hierarchy.
    """


class AIProvider(ABC):
    """
    Abstract base class defining the contract for all AI text-generation
    providers used by DataPilot AI.

    Concrete implementations (GeminiProvider, OpenAIProvider,
    ClaudeProvider, GroqProvider, OpenRouterProvider, OllamaProvider,
    etc.) must live under services/ai/ and implement `generate()`.

    Business logic modules (e.g. services/analyzer.py) depend only on
    this abstraction, never on a concrete provider class, satisfying the
    Dependency Inversion Principle.

    Required interface attributes:
        Every concrete implementation MUST set the following instance
        attributes in its `__init__`, in addition to implementing
        `generate()`. These are part of the provider contract, not
        optional metadata — callers (e.g. logging, diagnostics, Slack
        footer messages) may rely on them being present for ANY
        provider:

        name (str): A short, human-readable provider name
            (e.g. "Gemini", "OpenAI", "Claude", "Groq").
        vendor (str): The company/organization behind the provider
            (e.g. "Google", "OpenAI", "Anthropic", "Groq Inc.").
        model (str): The specific model identifier in use
            (e.g. "gemini-2.5-flash", "gpt-4o-mini").

    These are declared as class-level annotations below purely for
    documentation and static-analysis purposes (e.g. mypy). Python's
    ABC mechanism does not enforce instance-attribute presence the way
    it enforces abstract methods, so code review / testing should
    verify each new provider sets all three in __init__.
    """

    name: str
    vendor: str
    model: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a text completion from the underlying AI model.

        Args:
            system_prompt: Instructions defining the model's role,
                tone, and behavior for this request.
            user_prompt: The actual content/question the model should
                respond to.

        Returns:
            The generated text response, stripped of leading/trailing
            whitespace.

        Raises:
            AIProviderError: If the underlying provider fails to
                produce a valid response for any reason (authentication,
                network, rate limiting, malformed response, etc.).
        """
        raise NotImplementedError