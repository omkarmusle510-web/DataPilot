import os
from dotenv import load_dotenv

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

print("Bot Token Loaded:", BOT_TOKEN is not None)
print("App Token Loaded:", APP_TOKEN is not None)

app = App(token=BOT_TOKEN)


@app.event("app_mention")
def handle_mention(event, say):
    say("👋 Hello! I'm DataPilot AI and I'm connected successfully.")


if __name__ == "__main__":
    print("🚀 Starting DataPilot...")
    SocketModeHandler(app, APP_TOKEN).start()