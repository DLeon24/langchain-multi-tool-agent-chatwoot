"""
AI Agent Integration with Chatwoot (Opt-In Mode)

The bot only responds when the conversation has a specific activation label
(default: "ai-attends"). Conversations without the label are ignored.
Replies appear as a human agent.
"""

from chatwoot_base import CHATWOOT_BOT_LABEL, create_app, run

app = create_app({
    "mode": "opt_in",
    "title": "ChattyBot - AI Agent with Chatwoot (Opt-In)",
    "description": "Only responds when conversation has activation label",
})

if __name__ == "__main__":
    run(app, mode="opt_in", label=CHATWOOT_BOT_LABEL)
