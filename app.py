"""
app.py

Entry point for DataPilot AI's Slack bot (Socket Mode).

Responsibilities:
- Bootstrap the Slack Bolt app and environment configuration.
- Listen for @mentions and parse the "explain" command syntax.
- Delegate all SQL-explanation logic to services.analyzer.SQLAnalyzer,
  which is constructed with an AI provider obtained from the provider
  factory.
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

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from services.ai import get_provider
from services.analyzer import SQLAnalyzer, SQLExplanationError

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))

provider = get_provider()
analyzer = SQLAnalyzer(provider)

logger.info(
    "DataPilot AI configured with AI provider: name=%s, vendor=%s, model=%s",
    getattr(provider, "name", "unknown"),
    getattr(provider, "vendor", "unknown"),
    getattr(provider, "model", "unknown"),
)

_MENTION_PATTERN = re.compile(r"<@[^>]+>")
_EXPLAIN_PATTERN = re.compile(r"^\s*explain\s*(.*)$", re.IGNORECASE | re.DOTALL)


def _parse_explain_command(raw_text: str) -> str | None:
    """
    Parse a raw Slack mention message and extract the SQL query for an
    "explain" command.
    """
    without_mention = _MENTION_PATTERN.sub("", raw_text).strip()
    match = _EXPLAIN_PATTERN.match(without_mention)
    if not match:
        return None
    return match.group(1).strip()


def _build_explanation_blocks(sql_query: str, explanation: str) -> list[dict]:
    """
    Build a Block Kit message layout presenting a SQL explanation.
    """
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "🧠 SQL Explanation"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"```{sql_query}```"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": explanation}},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "🤖 Powered by DataPilot AI"}],
        },
    ]


@app.event("app_mention")
def handle_app_mention(event: dict, say) -> None:
    """
    Handle @DataPilot mentions and process the "explain" command.
    """
    try:
        raw_text = event.get("text", "")
        sql_query = _parse_explain_command(raw_text)

        if sql_query is None:
            return

        if not sql_query:
            say(text="Please paste a SQL query after the explain command.")
            return

        explanation = analyzer.explain_sql(sql_query)
        say(blocks=_build_explanation_blocks(sql_query, explanation), text="SQL Explanation")

    except SQLExplanationError as exc:
        logger.warning("SQL explanation failed: %s", exc)
        say(text=f":warning: {exc}")
    except Exception:
        logger.exception("Unexpected error handling app_mention event.")
        say(text=":warning: Something went wrong while processing your request.")


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    logger.info("⚡️ DataPilot AI is starting in Socket Mode...")
    handler.start()