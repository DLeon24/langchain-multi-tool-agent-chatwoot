"""
AI Agent Bot Integration with Chatwoot (Bot Identity Mode)

The bot responds to all conversations by default (stops with "ai-off" label).
Uses a separate CHATWOOT_BOT_TOKEN so replies appear with the bot's identity
(name + avatar) instead of as a human agent.
"""

from chatwoot_base import CHATWOOT_AI_OFF_LABEL, create_app, run

app = create_app({
    "mode": "bot",
    "title": "ChattyBot - AI Agent Bot with Chatwoot",
    "description": "Responds as Agent Bot with its own identity",
})

if __name__ == "__main__":
    run(app, mode="bot", label=CHATWOOT_AI_OFF_LABEL)
