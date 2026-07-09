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
- Listen for CSV file uploads and run the Dataset Intelligence
  workflow via services.file_handler.SlackFileHandler,
  services.dataset_profiler.DatasetProfiler, and
  services.dataset_cleaner.DatasetCleaner.
- Render responses as Slack Block Kit messages.
- Persist every command execution (successful or failed) to SQLite via
  HistoryRepository, without ever letting a persistence failure affect
  Slack responsiveness.

This module has NO knowledge of which AI vendor is active. It never
imports a concrete provider class (e.g. GeminiProvider) and never
reads vendor-specific environment variables (e.g. GOOGLE_API_KEY). It
only depends on services.ai.get_provider(), which resolves the active
provider based on the AI_PROVIDER environment variable.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Final, NamedTuple

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from database import DatabaseManager, HistoryRepository, HistoryRepositoryError
from services.ai import get_provider
from services.analyzer import SQLAnalyzer, SQLExplanationError
from services.cleaner import SQLCleaner, SQLCleanerError
from services.generator import SQLGenerator, SQLGeneratorError
from services.optimizer import SQLOptimizer, SQLOptimizerError
from services.validator import SQLValidator, SQLValidatorError
from services.file_handler import SlackFileHandler, SlackFileHandlerError
from services.dataset_profiler import DatasetProfiler, DatasetProfilerError
from services.dataset_cleaner import DatasetCleaner, DatasetCleanerError

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

# Dataset Intelligence services. Reuse the same shared `provider`
# instance already constructed above — never instantiate a second
# provider.
file_handler = SlackFileHandler()
dataset_profiler = DatasetProfiler(provider)
dataset_cleaner = DatasetCleaner()

# A single DatabaseManager and HistoryRepository are constructed once
# at startup and reused for the lifetime of the application, mirroring
# how the shared AI `provider` instance is handled. Never construct
# these inside handle_app_mention() or handle_dataset_file_upload().
db_manager = DatabaseManager()
history_repository = HistoryRepository(db_manager)

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


def _build_dataset_intelligence_blocks(
    original_filename: str,
    original_profile,
    cleaning_result,
    cleaned_profile,
    execution_time: float,
) -> list[dict]:
    """
    Build a Block Kit message layout presenting a full Dataset
    Intelligence Report: original dataset info, before-cleaning stats,
    cleaning summary, after-cleaning stats, AI insights, and the
    cleaned dataset filename.

    Args:
        original_filename: The original uploaded CSV filename.
        original_profile: DatasetProfileResult for the dataset before
            cleaning.
        cleaning_result: CleaningResult describing the cleaning
            operation.
        cleaned_profile: DatasetProfileResult for the dataset after
            cleaning.
        execution_time: Total workflow execution time in seconds.

    Returns:
        A list of Slack Block Kit block dictionaries ready to send.
    """
    memory_usage_mb = original_profile.memory_usage_bytes / (1024 * 1024)

    renamed_columns_text = (
        ", ".join(f"{old} → {new}" for old, new in cleaning_result.renamed_columns.items())
        if cleaning_result.renamed_columns
        else "None"
    )
    columns_removed_text = ", ".join(cleaning_result.columns_removed) or "None"
    columns_converted_text = ", ".join(cleaning_result.columns_converted) or "None"
    cleaning_actions_text = "\n".join(
        f"• {action}" for action in cleaning_result.cleaning_actions
    ) or "None"

    return [
        {"type": "header", "text": {"type": "plain_text", "text": "📊 Dataset Intelligence Report"}},
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"📁 *Dataset Information*\n"
                    f"*Filename:* {original_filename}\n"
                    f"*Rows:* {original_profile.rows}\n"
                    f"*Columns:* {original_profile.columns}\n"
                    f"*Memory Usage:* {memory_usage_mb:.2f} MB"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"📈 *Before Cleaning*\n"
                    f"*Quality Score:* {original_profile.quality_score}/10\n"
                    f"*Duplicate Rows:* {original_profile.duplicate_rows}\n"
                    f"*Missing Values:* {original_profile.total_missing_values}"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"🧹 *Cleaning Summary*\n"
                    f"*Rows Removed:* {cleaning_result.rows_removed}\n"
                    f"*Duplicate Rows Removed:* {cleaning_result.duplicates_removed}\n"
                    f"*Empty Rows Removed:* {cleaning_result.empty_rows_removed}\n"
                    f"*Columns Removed:* {columns_removed_text}\n"
                    f"*Columns Converted:* {columns_converted_text}\n"
                    f"*Renamed Columns:* {renamed_columns_text}\n"
                    f"*Cleaning Actions:*\n{cleaning_actions_text}"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"📈 *After Cleaning*\n"
                    f"*Quality Score:* {cleaned_profile.quality_score}/10\n"
                    f"*Duplicate Rows:* {cleaned_profile.duplicate_rows}\n"
                    f"*Missing Values:* {cleaned_profile.total_missing_values}"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"💡 *AI Insights*\n{cleaned_profile.ai_report}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"📂 *Cleaned Dataset*\n*Filename:* {Path(cleaning_result.output_file).name}",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"🤖 DataPilot AI · provider={getattr(provider, 'name', 'unknown')} "
                        f"· {execution_time:.2f}s"
                    ),
                }
            ],
        },
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

# Errors raised by the Dataset Intelligence workflow. Handled
# identically to _BUSINESS_ERRORS, but kept as a separate tuple since
# they are raised by a completely independent event handler.
_DATASET_ERRORS = (
    SlackFileHandlerError,
    DatasetProfilerError,
    DatasetCleanerError,
)


def _save_history(
    user_id: str,
    command: str,
    input_text: str,
    output_text: str,
    execution_time: float,
    duration_ms: int,
    success: bool,
    error_message: str | None,
) -> None:
    """
    Persist a single command execution record to history.

    This is a best-effort operation: history is a secondary feature,
    so any failure while saving is logged and swallowed rather than
    propagated. A database outage must never prevent the Slack bot
    from responding to the user.

    Args:
        user_id: Slack user ID who invoked the command.
        command: Command name (e.g. "explain", "clean", "generate",
            "optimize", "validate", "dataset").
        input_text: The raw argument text submitted for the command.
        output_text: The result produced by the command, or an empty
            string if the command failed.
        execution_time: Execution time in seconds.
        duration_ms: Execution time in integer milliseconds.
        success: Whether the command completed successfully.
        error_message: The error message if the command failed, or
            None if it succeeded.
    """
    try:
        history_repository.save_command(
            user_id=user_id,
            command=command,
            input_text=input_text,
            output_text=output_text,
            provider=getattr(provider, "vendor", "unknown"),
            model=getattr(provider, "model", "unknown"),
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
        )
    except HistoryRepositoryError:
        logger.exception(
            "Failed to save command history. command=%s, user_id=%s",
            command,
            user_id,
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
        - Every successful or failed command execution is persisted to
          history via HistoryRepository. Persistence failures are
          logged but never prevent a Slack response.
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

        user_id = event.get("user", "unknown")
        start_time = time.perf_counter()

        try:
            result = spec.process(argument_text)
        except _BUSINESS_ERRORS as exc:
            execution_time = time.perf_counter() - start_time
            _save_history(
                user_id=user_id,
                command=command,
                input_text=argument_text,
                output_text="",
                execution_time=execution_time,
                duration_ms=int(execution_time * 1000),
                success=False,
                error_message=str(exc),
            )
            raise

        execution_time = time.perf_counter() - start_time
        _save_history(
            user_id=user_id,
            command=command,
            input_text=argument_text,
            output_text=result,
            execution_time=execution_time,
            duration_ms=int(execution_time * 1000),
            success=True,
            error_message=None,
        )

        say(blocks=spec.build_blocks(argument_text, result), text=f"SQL {command.capitalize()}")

    except _BUSINESS_ERRORS as exc:
        logger.warning("Command processing failed: %s", exc)
        say(text=f":warning: {exc}")
    except Exception:  # noqa: BLE001 - final safety net, never crash the bot
        logger.exception("Unexpected error handling app_mention event.")
        say(text=":warning: Something went wrong while processing your request.")


@app.event("message")
def handle_dataset_file_upload(event: dict, say, client) -> None:
    """
    Handle Slack messages that share an uploaded CSV file and run the
    full Dataset Intelligence workflow: download, profile the original,
    clean, profile the cleaned result, report, and re-upload.

    This handler is completely independent from handle_app_mention:
    SQL commands are triggered by @mentions, dataset intelligence is
    triggered by file_share message events, and neither affects the
    other's parsing or dispatch logic.

    Behavior:
        - Ignores any message that is not a file share, or that shares
          no .csv files.
        - Downloads the first CSV file via SlackFileHandler.
        - Profiles the original dataset via DatasetProfiler.
        - Cleans it via DatasetCleaner.
        - Profiles the cleaned dataset via DatasetProfiler again (a
          second, intentional profiling pass over the cleaned output,
          not a redundant re-read of the original file).
        - Replies with a Block Kit Dataset Intelligence Report and
          uploads the cleaned CSV to the same thread.
        - Persists the outcome to history under the "dataset" command,
          reusing the existing _save_history() helper exactly as SQL
          commands do.
        - Any failure (download, invalid CSV, AI, or cleaning) is
          reported via a friendly Slack message and is still recorded
          in history as a failure. The bot never crashes.

    Args:
        event: The Slack event payload for the message event.
        say: Slack Bolt's helper for posting a message back to the channel.
        client: Slack Bolt's WebClient, used to upload the cleaned file.
    """
    logger.info("DATASET HANDLER INVOKED. subtype=%s", event.get("subtype"))  # ADD THIS LINE
    if event.get("subtype") != "file_share":
        return

    csv_files = [
        f for f in event.get("files", []) if f.get("name", "").lower().endswith(".csv")
    ]
    if not csv_files:
        return

    slack_file = csv_files[0]
    original_filename = slack_file.get("name", "upload.csv")
    download_url = slack_file.get("url_private_download", "")
    user_id = event.get("user", "unknown")
    channel_id = event.get("channel")
    thread_ts = event.get("ts")

    start_time = time.perf_counter()

    try:
        local_path = file_handler.download_csv(download_url, original_filename)
        original_profile = dataset_profiler.profile_dataset(local_path)
        cleaning_result = dataset_cleaner.clean_dataset(local_path)
        cleaned_profile = dataset_profiler.profile_dataset(cleaning_result.output_file)

        execution_time = time.perf_counter() - start_time
        _save_history(
            user_id=user_id,
            command="dataset",
            input_text=original_filename,
            output_text=cleaned_profile.ai_report,
            execution_time=execution_time,
            duration_ms=int(execution_time * 1000),
            success=True,
            error_message=None,
        )

        say(
            blocks=_build_dataset_intelligence_blocks(
                original_filename=original_filename,
                original_profile=original_profile,
                cleaning_result=cleaning_result,
                cleaned_profile=cleaned_profile,
                execution_time=execution_time,
            ),
            text="Dataset Intelligence Report",
            thread_ts=thread_ts,
        )

        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=cleaning_result.output_file,
            filename=Path(cleaning_result.output_file).name,
            initial_comment="📂 Here's your cleaned dataset:",
        )

    except _DATASET_ERRORS as exc:
        execution_time = time.perf_counter() - start_time
        _save_history(
            user_id=user_id,
            command="dataset",
            input_text=original_filename,
            output_text="",
            execution_time=execution_time,
            duration_ms=int(execution_time * 1000),
            success=False,
            error_message=str(exc),
        )
        logger.warning("Dataset intelligence workflow failed: %s", exc)
        say(text=f":warning: {exc}", thread_ts=thread_ts)
    except Exception:  # noqa: BLE001 - final safety net, never crash the bot
        execution_time = time.perf_counter() - start_time
        _save_history(
            user_id=user_id,
            command="dataset",
            input_text=original_filename,
            output_text="",
            execution_time=execution_time,
            duration_ms=int(execution_time * 1000),
            success=False,
            error_message="Unexpected error during dataset processing.",
        )
        logger.exception("Unexpected error handling dataset file upload event.")
        say(
            text=":warning: Something went wrong while processing your dataset.",
            thread_ts=thread_ts,
        )


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    logger.info("⚡️ DataPilot AI is starting in Socket Mode...")
    handler.start()