"""
app.py

Entry point for DataPilot AI's Slack bot (Socket Mode).

Responsibilities:
- Bootstrap the Slack Bolt app and environment configuration.
- Listen for @mentions and parse supported commands
  ("explain", "clean", "generate", "optimize", "validate").
- Delegate business logic to services.analyzer.SQLAnalyzer,
  services.cleaner.SQLCleaner, services.generator.SQLGenerator,
  services.optimizer.SQLOptimizer, and services.validator.SQLValidator,
  all constructed with a single shared AI provider obtained from the
  provider factory.
- Render responses as Slack Block Kit messages.

This module has NO knowledge of which AI vendor is active. It never
imports a concrete provider class (e.g. GeminiProvider) and never
reads vendor-specific environment variables (e.g. GOOGLE_API_KEY). It
only depends on services.ai.get_provider(), which resolves the active
provider based on the AI_PROVIDER environment variable.
"""

import logging
import os
import re
from typing import Callable, Final, NamedTuple

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from services.ai import get_provider
from services.analyzer import SQLAnalyzer, SQLExplanationError
from services.cleaner import SQLCleaner, SQLCleanerError
from services.generator import SQLGenerator, SQLGeneratorError
from services.optimizer import SQLOptimizer, SQLOptimizerError
from services.validator import SQLValidator, SQLValidatorError

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# A single provider instance is resolved once at startup via the
# factory, then shared across every business-logic service. app.py
# never knows or cares which concrete provider (Gemini, OpenAI,
# Claude, Groq, etc.) is active — that decision belongs entirely to
# services/ai/provider_factory.py, driven by the AI_PROVIDER env var.
provider = get_provider()
analyzer = SQLAnalyzer(provider)
cleaner = SQLCleaner(provider)
generator = SQLGenerator(provider)
optimizer = SQLOptimizer(provider)
validator = SQLValidator(provider)

logger.info(
    "DataPilot AI configured with AI provider: name=%s, vendor=%s, model=%s",
    getattr(provider, "name", "unknown"),
    getattr(provider, "vendor", "unknown"),
    getattr(provider, "model", "unknown"),
)

# Central source of truth for supported command keywords. The parsing
# regex is built dynamically from this tuple, so adding a new command
# only ever requires a new _COMMAND_HANDLERS entry plus a name here —
# no regex or parsing logic changes.
SUPPORTED_COMMANDS: Final[tuple[str, ...]] = (
    "explain",
    "clean",
    "generate",
    "optimize",
    "validate",
)

_MENTION_PATTERN = re.compile(r"<@[^>]+>")
_COMMAND_PATTERN = re.compile(
    rf"^\s*({'|'.join(SUPPORTED_COMMANDS)})\s*(.*)$", re.IGNORECASE | re.DOTALL
)


def _parse_command(raw_text: str) -> tuple[str, str] | None:
    """
    Parse a raw Slack mention message and extract the command name and
    its argument text.

    Strips the leading user-mention token (e.g. "<@U123ABC>"), then
    checks whether the remaining text starts with one of the keywords
    in SUPPORTED_COMMANDS.

    Args:
        raw_text: The full text of the Slack event, including the
            bot mention.

    Returns:
        A tuple of (command_name, argument_text) where command_name is
        lowercase and a member of SUPPORTED_COMMANDS, and argument_text
        may be an empty string if the user typed only the command with
        nothing after it. Returns None if the message does not use any
        supported command.
    """
    without_mention = _MENTION_PATTERN.sub("", raw_text).strip()
    match = _COMMAND_PATTERN.match(without_mention)
    if not match:
        return None
    command = match.group(1).lower()
    argument_text = match.group(2).strip()
    return command, argument_text


def _build_footer_context() -> dict:
    """
    Build the shared Block Kit context footer used by every command
    response, keeping the branding line defined in exactly one place.

    Returns:
        A single Block Kit "context" block dictionary.
    """
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "🤖 Powered by DataPilot AI"}],
    }


def _build_explanation_blocks(sql_query: str, explanation: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting a SQL explanation.

    Args:
        sql_query: The original SQL query submitted by the user.
        explanation: The plain-English explanation returned by the AI.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "🧠 SQL Explanation"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"```{sql_query}```"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": explanation}},
        _build_footer_context(),
    ]


def _build_cleaned_sql_blocks(original_sql: str, cleaned_sql: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting a cleaned/formatted
    SQL query alongside the original.

    Args:
        original_sql: The raw SQL query submitted by the user.
        cleaned_sql: The reformatted SQL query returned by the AI.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "🧹 Cleaned SQL"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original:*\n```{original_sql}```"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Formatted:*\n```{cleaned_sql}```"},
        },
        _build_footer_context(),
    ]


def _build_generated_sql_blocks(user_request: str, generated_sql: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting SQL generated from a
    natural language request.

    Args:
        user_request: The original natural language request submitted
            by the user.
        generated_sql: The ANSI SQL query generated by the AI.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "⚙️ Generated SQL"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Request:*\n{user_request}"},
        },
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"```{generated_sql}```"}},
        _build_footer_context(),
    ]


def _build_optimizer_blocks(sql_query: str, report: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting a SQL performance
    optimization report alongside the original query.

    Args:
        sql_query: The original SQL query submitted by the user.
        report: The structured optimization report returned by the AI.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "⚡ SQL Performance Analysis"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original SQL:*\n```{sql_query}```"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Optimization Report:*\n{report}"},
        },
        _build_footer_context(),
    ]


def _build_validation_blocks(sql_query: str, report: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting a SQL validation
    report alongside the original query.

    Args:
        sql_query: The original SQL query submitted by the user.
        report: The structured validation report returned by the AI.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "✅ SQL Validation Report"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original SQL:*\n```{sql_query}```"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Validation Report:*\n{report}"},
        },
        _build_footer_context(),
    ]


class _CommandSpec(NamedTuple):
    """
    Defines how a single supported command is processed end-to-end.

    Attributes:
        process: A callable that takes the raw argument text and
            returns the AI-generated result string. Raises the
            command's domain-specific error on failure.
        build_blocks: A callable that takes (original_argument, result)
            and returns the Slack Block Kit blocks to reply with.
        missing_input_message: The message to show the user when the
            command is invoked with no argument text attached.
    """

    process: Callable[[str], str]
    build_blocks: Callable[[str, str], list[dict]]
    missing_input_message: str


# Registry of supported commands. Adding a new command means adding
# one entry here and one keyword to SUPPORTED_COMMANDS — no changes to
# the parsing logic or the event handler's control flow are required.
_COMMAND_HANDLERS: dict[str, _CommandSpec] = {
    "explain": _CommandSpec(
        process=analyzer.explain_sql,
        build_blocks=_build_explanation_blocks,
        missing_input_message="Please paste a SQL query after the explain command.",
    ),
    "clean": _CommandSpec(
        process=cleaner.clean_sql,
        build_blocks=_build_cleaned_sql_blocks,
        missing_input_message="Please paste a SQL query after the clean command.",
    ),
    "generate": _CommandSpec(
        process=generator.generate_sql,
        build_blocks=_build_generated_sql_blocks,
        missing_input_message=(
            "Please describe what you'd like a SQL query for after the "
            "generate command."
        ),
    ),
    "optimize": _CommandSpec(
        process=optimizer.optimize_sql,
        build_blocks=_build_optimizer_blocks,
        missing_input_message="Please paste a SQL query after the optimize command.",
    ),
    "validate": _CommandSpec(
        process=validator.validate_sql,
        build_blocks=_build_validation_blocks,
        missing_input_message="Please paste a SQL query after the validate command.",
    ),
}

# All business-logic services raise their own domain-specific error on
# failure. They are handled identically at the Slack layer, so they
# are caught together as a single tuple.
_BUSINESS_ERRORS = (
    SQLExplanationError,
    SQLCleanerError,
    SQLGeneratorError,
    SQLOptimizerError,
    SQLValidatorError,
)


@app.event("app_mention")
def handle_app_mention(event: dict, say) -> None:
    """
    Handle @DataPilot mentions and process supported commands.

    Behavior:
        - Ignores any mention that does not use a supported command.
        - If a command is used with no argument text, asks the user
          to provide one.
        - If argument text is present, dispatches to the command's
          processor (SQLAnalyzer, SQLCleaner, SQLGenerator,
          SQLOptimizer, or SQLValidator) and replies with a Block Kit
          message containing the result.
        - Any failure (parsing, AI service, or unexpected) is caught
          and reported to the user safely, without crashing the app or
          leaking stack traces.

    Args:
        event: The Slack event payload for the app_mention event.
        say: Slack Bolt's helper for posting a message back to the channel.
    """
    try:
        raw_text = event.get("text", "")
        parsed = _parse_command(raw_text)

        if parsed is None:
            return  # Not a supported command — ignore per spec.

        command, argument_text = parsed
        spec = _COMMAND_HANDLERS[command]

        if not argument_text:
            say(text=spec.missing_input_message)
            return

        result = spec.process(argument_text)
        say(blocks=spec.build_blocks(argument_text, result), text=f"SQL {command.capitalize()}")

    except _BUSINESS_ERRORS as exc:
        logger.warning("Command processing failed: %s", exc)
        say(text=f":warning: {exc}")
    except Exception:  # noqa: BLE001 - final safety net, never crash the bot
        logger.exception("Unexpected error handling app_mention event.")
        say(text=":warning: Something went wrong while processing your request.")


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    logger.info("⚡️ DataPilot AI is starting in Socket Mode...")
    handler.start()