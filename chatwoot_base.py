"""
Chatwoot Integration Base Module
Shared logic for all Chatwoot entry points (opt-in, opt-out, bot).
"""

import os
import sys
import uuid
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import requests
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Request

load_dotenv(find_dotenv())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================
# 1. AGENT LOADING
# ============================================
def load_agent():
    """Load Agent 04 module."""
    base_dir = Path(__file__).parent
    path = base_dir / "04-agent-chat-memory-rag-web" / "agent_chat_memory_rag_web.py"
    spec = spec_from_file_location("agent_04", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("🤖 Loading Agent 04...")
agent = load_agent()
chat_with_agent = agent.chat_with_agent
print("✅ Agent 04 loaded successfully")

# ============================================
# 2. CHATWOOT CONFIGURATION
# ============================================
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_ACCESS_TOKEN")
CHATWOOT_BOT_TOKEN = os.getenv("CHATWOOT_BOT_TOKEN", CHATWOOT_API_TOKEN)
CHATWOOT_BOT_LABEL = os.getenv("CHATWOOT_BOT_LABEL", "ai-attends")
CHATWOOT_AI_OFF_LABEL = os.getenv("CHATWOOT_AI_OFF_LABEL", "ai-off")

HUMAN_KEYWORDS = [
    "human",
    "person",
    "advisor",
    "agent",
    "representative",
    "talk to someone",
]

HANDOFF_MESSAGE = (
    "Understood. A human advisor will reach out to you shortly. "
    "Thank you for your patience!"
)

ERROR_MESSAGE = (
    "Sorry, I had a problem processing your request. "
    "An advisor will assist you soon."
)

if not all([CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN]):
    print("⚠️  WARNING: Missing Chatwoot variables in .env")
    print(
        "   Required: CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_ACCESS_TOKEN"
    )
else:
    print(f"✅ Chatwoot configured: {CHATWOOT_BASE_URL}")


# ============================================
# 3. CHATWOOT API FUNCTIONS
# ============================================
def send_message(conversation_id: int, message: str, token: str = None) -> bool:
    """
    Send a reply message to a Chatwoot conversation.

    Args:
        conversation_id: Conversation ID
        message: Message to send
        token: API token to use (defaults to CHATWOOT_API_TOKEN)

    Returns:
        True if sent successfully, False on error
    """
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {
        "api_access_token": token or CHATWOOT_API_TOKEN,
        "Content-Type": "application/json",
    }
    payload = {"content": message, "message_type": "outgoing"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"   ✅ Message sent to conversation {conversation_id}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error sending message: {e}")
        return False


def update_labels(conversation_id: int, labels: list) -> bool:
    """
    Update labels for a Chatwoot conversation.
    Always uses the admin token (CHATWOOT_API_TOKEN).

    Args:
        conversation_id: Conversation ID
        labels: List of labels

    Returns:
        True if updated successfully
    """
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/labels"
    headers = {
        "api_access_token": CHATWOOT_API_TOKEN,
        "Content-Type": "application/json",
    }
    payload = {"labels": labels}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"   ✅ Labels updated: {labels}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error updating labels: {e}")
        return False


def conversation_id_to_uuid(conversation_id: int) -> str:
    """
    Convert a Chatwoot conversation_id to a valid UUID.
    This allows reusing the same session_id for the same conversation.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chatwoot-{conversation_id}"))


# ============================================
# 4. MODE-SPECIFIC STRATEGIES
# ============================================
def _opt_in_strategy(label: str):
    """Opt-in: only respond if the activation label is present."""

    def should_respond(labels):
        if label and label not in labels:
            print(f"   ⏭️  Ignored: missing label '{label}'")
            return False
        return True

    def on_handoff(labels):
        new_labels = [l for l in labels if l != label]
        new_labels.append("human-attends")
        return new_labels

    return should_respond, on_handoff, CHATWOOT_API_TOKEN, label


def _opt_out_strategy(label: str):
    """Opt-out: respond to all, stop only if the disable label is present."""

    def should_respond(labels):
        if label in labels:
            print(f"   ⏭️  Ignored: conversation has '{label}' label")
            return False
        return True

    def on_handoff(labels):
        new_labels = list(labels)
        if label not in new_labels:
            new_labels.append(label)
        return new_labels

    return should_respond, on_handoff, CHATWOOT_API_TOKEN, label


def _bot_strategy(label: str):
    """Bot: same as opt-out, but replies use the bot token."""
    should_respond, on_handoff, _, resolved_label = _opt_out_strategy(label)
    token = CHATWOOT_BOT_TOKEN

    if token != CHATWOOT_API_TOKEN:
        print("✅ Agent Bot token configured")
    else:
        print("⚠️  No CHATWOOT_BOT_TOKEN set, replies will appear as agent (not bot)")

    return should_respond, on_handoff, token, resolved_label


_STRATEGIES = {
    "opt_in": _opt_in_strategy,
    "opt_out": _opt_out_strategy,
    "bot": _bot_strategy,
}


# ============================================
# 5. APP FACTORY
# ============================================
def create_app(config: dict) -> FastAPI:
    """
    Create a FastAPI app for Chatwoot integration.

    Args:
        config: Dict with keys:
            - mode: "opt_in" | "opt_out" | "bot"
            - title: FastAPI app title
            - description: FastAPI app description
            - label: Label name used for filtering (optional, has defaults per mode)
    """
    mode = config["mode"]
    title = config.get("title", "ChattyBot - AI Agent with Chatwoot")
    description = config.get(
        "description", "Webhook to integrate Agent 04 with Chatwoot"
    )

    default_labels = {
        "opt_in": CHATWOOT_BOT_LABEL,
        "opt_out": CHATWOOT_AI_OFF_LABEL,
        "bot": CHATWOOT_AI_OFF_LABEL,
    }
    label = config.get("label", default_labels.get(mode, CHATWOOT_AI_OFF_LABEL))

    strategy_fn = _STRATEGIES[mode]
    should_respond, on_handoff, message_token, resolved_label = strategy_fn(label)

    app = FastAPI(title=title, description=description, version="1.0.0")

    @app.post("/webhook")
    async def chatwoot_webhook(request: Request):
        """Endpoint that receives Chatwoot webhooks."""
        data = await request.json()

        event = data.get("event")
        message_type = data.get("message_type")
        conversation = data.get("conversation", {})
        labels = conversation.get("labels", [])
        message_content = data.get("content")
        conversation_id = conversation.get("id")

        print(f"\n{'='*60}")
        print(f"📩 Webhook received: {event}")
        print(f"   Conversation: {conversation_id}")
        print(f"   Type: {message_type}")
        print(f"   Labels: {labels}")

        if event != "message_created":
            return {"status": "ignored", "reason": "Not a message_created event"}

        if message_type != "incoming":
            return {"status": "ignored", "reason": "Not an incoming message"}

        if not should_respond(labels):
            return {
                "status": "ignored",
                "reason": f"Filtered by label policy (mode={mode})",
            }

        if not message_content or not conversation_id:
            return {"status": "ignored", "reason": "Missing content or conversation_id"}

        print(f"   📝 Message: {message_content[:100]}...")

        if any(kw in message_content.lower() for kw in HUMAN_KEYWORDS):
            print("   🗣️ Human handoff detected")
            new_labels = on_handoff(labels)
            update_labels(conversation_id, new_labels)
            send_message(conversation_id, HANDOFF_MESSAGE, message_token)
            return {"status": "success", "action": "human_handoff"}

        try:
            print("   🤖 Processing with Agent 04...")
            session_id = conversation_id_to_uuid(conversation_id)
            print(f"   📝 Session ID: {session_id[:8]}...")

            response_text = chat_with_agent(message_content, session_id)
            print(f"   ✅ Response generated ({len(response_text)} chars)")

            send_message(conversation_id, response_text, message_token)
            return {"status": "success", "action": "agent_response"}

        except Exception as e:
            print(f"   ❌ Error processing: {e}")
            send_message(conversation_id, ERROR_MESSAGE, message_token)
            return {"status": "error", "message": str(e)}

    @app.get("/")
    def read_root():
        """Root endpoint with service information."""
        info = {
            "service": title,
            "version": "1.0.0",
            "agent": "Agent 04 (RAG + Internet + Memory)",
            "model": "GPT-4o",
            "tools": ["search_ai_perupe", "search_internet"],
            "mode": mode,
            "label": resolved_label,
            "chatwoot_configured": all(
                [CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN]
            ),
            "status": "ready",
        }
        if mode == "bot":
            info["bot_token_configured"] = CHATWOOT_BOT_TOKEN != CHATWOOT_API_TOKEN
        return info

    @app.get("/health")
    def health_check():
        """Service health check endpoint."""
        return {
            "status": "healthy",
            "agent": "Agent 04",
            "chatwoot": (
                "connected"
                if all([CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN])
                else "not configured"
            ),
        }

    @app.post("/test")
    async def test_agent(request: Request):
        """
        Test endpoint to try the agent without Chatwoot.

        Body: {"message": "your question", "session_id": "optional"}
        """
        data = await request.json()
        message = data.get("message", "")
        session_id = data.get("session_id", str(uuid.uuid4()))

        if not message:
            return {"error": "You must provide a 'message' in the body"}

        print(f"\n🧪 TEST - Message: {message}")
        print(f"   Session: {session_id[:8]}...")

        try:
            response_text = chat_with_agent(message, session_id)
            print(f"   ✅ Response: {response_text[:100]}...")
            return {
                "message": message,
                "session_id": session_id,
                "response": response_text,
                "status": "success",
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"message": message, "error": str(e), "status": "error"}

    return app


# ============================================
# 6. RUN HELPER
# ============================================
def run(app: FastAPI, mode: str = "", label: str = ""):
    """Start the uvicorn server with a startup banner."""
    print()
    print("=" * 60)
    print(f"🚀 STARTING CHATBOT WITH CHATWOOT ({mode.upper().replace('_', ' ')} MODE)")
    print("=" * 60)
    print("🤖 Agent: 04 (RAG + Internet + Memory)")
    print("🧠 Model: GPT-4o")
    print("🔧 Tools: search_ai_perupe, search_internet")
    print("💾 History: PostgreSQL")
    if label:
        print(f"🏷️  Label: {label}")
    if mode == "bot":
        status = (
            "configured"
            if CHATWOOT_BOT_TOKEN != CHATWOOT_API_TOKEN
            else "not set (using agent token)"
        )
        print(f"🤖 Bot token: {status}")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
