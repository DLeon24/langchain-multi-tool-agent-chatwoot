"""
AI Agent Integration with Chatwoot (Opt-Out Mode)

The bot responds to all conversations by default. It only stops responding
when the conversation has the "ai-off" label. Replies appear as a human agent.
"""

from chatwoot_base import CHATWOOT_AI_OFF_LABEL, create_app, run

app = create_app({
    "mode": "opt_out",
    "title": "ChattyBot - AI Agent with Chatwoot (Opt-Out)",
    "description": "Responds to all conversations, stops with ai-off label",
})

if __name__ == "__main__":
    run(app, mode="opt_out", label=CHATWOOT_AI_OFF_LABEL)
