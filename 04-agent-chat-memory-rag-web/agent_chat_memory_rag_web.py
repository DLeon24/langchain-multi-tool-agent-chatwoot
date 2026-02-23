"""
Complete AI Agent: Knowledge Base + Internet + Conversation History

- Tool 1: Knowledge Base (RAG with Supabase)
- Tool 2: Internet Search (Tavily)
- History: Stores conversations in PostgreSQL
- Exposes chat_with_agent(message, session_id) for Chatwoot integration
"""

import os
import sys
import uuid
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Add root directory to path for importing tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_postgres import PostgresChatMessageHistory

from tools.datetime import get_current_datetime
from tools.internet_search import search_internet
from tools.knowledge_base import search_ai_perupe

# ============================================
# 1. DATABASE CONFIGURATION (History)
# ============================================
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

if not all([DB_USER, DB_PASSWORD, DB_HOST]):
    raise ValueError(
        "❌ Missing database variables in .env\n"
        "Required: DB_USER, DB_PASSWORD, DB_HOST\n"
        "Optional: DB_PORT (default: 5432), DB_NAME (default: postgres)"
    )

DATABASE_URL = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ============================================
# 2. AVAILABLE TOOLS LIST
# ============================================
TOOLS = [
    search_ai_perupe,  # AIPerupe Academy Knowledge Base
    search_internet,  # Internet Search (Tavily)
    get_current_datetime,  # Current date and time in a timezone
]

# ============================================
# 3. AGENT PROMPT
# ============================================
SYSTEM_PROMPT = """You are ChattyBot, a AIPerupe Academy AI assistant with internet access.

Your goal is to help users by answering their questions using the available tools.

At the start of each shift, you'll see the CURRENT DATE AND TIME; use this whenever the answer depends on "today," "now," "this week," schedules, or deadlines. For other time zones, use the get_current_datetime tool.

AVAILABLE TOOLS:
1. search_ai_perupe: For information about AIPerupe Academy (programs, courses, prices, instructors)
2. search_internet: For up-to-date information from the internet (news, events, current data)
3. get_current_datetime: For the current date and time in a timezone

INSTRUCTIONS:
- For questions about AIPerupe Academy → USE search_ai_perupe
- For questions about current events, news, or general information → USE search_internet
- For "what time is it", "what day is it", "current date" in your time zone → You can use the CURRENT DATE AND TIME from the context; for another time zone → USE get_current_datetime
- For greetings, thanks, or general conversation → Respond directly WITHOUT tools
- You can use MULTIPLE tools if the question requires it
- You remember the entire conversation thanks to your persistent memory
- Always respond in a clear and friendly manner

EXAMPLES:
- "Hello" → Respond directly
- "What courses do you offer?" → Use search_ai_perupe
- "What happened in the news today?" → Use search_internet
- "What time is it?" or "What day is it today?" → Use get_current_datetime
- "How does your AI course compare to current trends?" → Use BOTH tools (search_ai_perupe and search_internet)"""


# ============================================
# CONVERSATION LOOP (CLI)
# ============================================
def main():
    print(f"🔌 Connecting as: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    _create_history_table()

    print("=" * 60)
    print("🤖 ChattyBot - COMPLETE Agent (KB + Internet + Memory)")
    print("=" * 60)
    print("🔧 Available tools:")
    for t in TOOLS:
        print(f"   - {t.name}")
    print("💾 History: PostgreSQL")

    print("\nSession options:")
    print("  1. New conversation")
    print("  2. Continue existing session (paste UUID)")

    session_id = _get_valid_session_id()
    chat_with_tools = _get_chat_with_tools()
    sync_connection = psycopg.connect(DATABASE_URL)

    print(f"\n📝 Session ID: {session_id}")
    print("   (Save this ID to continue later)")
    print("✅ The agent can search AIPerupe and the INTERNET")
    print("Type 'exit' to return to the menu.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["salir", "exit", "quit"]:
                print(f"\n💾 Your session has been saved.")
                print(f"   UUID: {session_id}")
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            try:
                answer = _run_turn(
                    chat_with_tools, sync_connection, user_input, session_id
                )
                print(f"\n🤖 ChattyBot: {answer}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    finally:
        sync_connection.close()


# ============================================
# CHAT WITH AGENT (public API for Chatwoot)
# ============================================
def chat_with_agent(user_message: str, session_id: str) -> str:
    """
    Runs the agent with tools and memory. Used by Chatwoot webhook.
    Opens and closes its own DB connection.
    """
    chat_with_tools = _get_chat_with_tools()
    with psycopg.connect(DATABASE_URL) as sync_connection:
        history = _get_session_history(sync_connection, session_id)
        final_answer = _invoke_agent(chat_with_tools, history, user_message)
        history.add_user_message(user_message)
        history.add_ai_message(final_answer)
    return final_answer


# ============================================
# CREATE HISTORY TABLE (if it doesn't exist)
# ============================================
def _create_history_table():
    """Creates the history table if it doesn't exist."""
    try:
        with psycopg.connect(DATABASE_URL) as sync_connection:
            PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
    except Exception as e:
        print(f"⚠️ Error creating history table: {e}")


# ============================================
# GET VALID SESSION ID
# ============================================
def _get_valid_session_id() -> str:
    option = input("\nChoose (1/2): ").strip()
    if option == "2":
        session_id = input("Paste the session UUID: ").strip()
        try:
            uuid.UUID(session_id)
            return session_id
        except ValueError:
            print("⚠️ Invalid UUID. Creating new session...")
            return str(uuid.uuid4())
    return str(uuid.uuid4())


# ============================================
# GET CHAT MODEL WITH TOOLS
# ============================================
def _get_chat_with_tools():
    """Initializes the chat model and binds the available tools."""
    chat = init_chat_model("gpt-4o", temperature=0.7)
    return chat.bind_tools(TOOLS)


# ============================================
# RUN ONE TURN (CLI – reuses connection and model)
# ============================================
def _run_turn(
    chat_with_tools, sync_connection, user_message: str, session_id: str
) -> str:
    """Runs one agent turn with existing connection and model; persists and returns the answer."""
    history = _get_session_history(sync_connection, session_id)
    final_answer = _invoke_agent(chat_with_tools, history, user_message)
    history.add_user_message(user_message)
    history.add_ai_message(final_answer)
    return final_answer


# ============================================
# GET SESSION HISTORY FROM POSTGRESQL
# ============================================
def _get_session_history(
    sync_connection, session_id: str
) -> PostgresChatMessageHistory:
    """Retrieves or creates the history for a session from PostgreSQL."""
    return PostgresChatMessageHistory(
        "chat_history", session_id, sync_connection=sync_connection
    )


# ============================================
# INVOKE AGENT (core logic, no persistence)
# ============================================
def _invoke_agent(chat_with_tools, history, user_message: str) -> str:
    """
    Builds messages from history, invokes the agent (with optional tool calls),
    and returns the final answer. Does not persist to history.
    """
    previous_messages = history.messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in previous_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    messages.append({"role": "user", "content": user_message})
    response = chat_with_tools.invoke(messages)

    if response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            for t in TOOLS:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results.append(
                        {"tool_call_id": tool_call["id"], "result": result}
                    )
                    break

        messages.append(response)
        for tr in tool_results:
            messages.append(
                ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"])
            )
        final_response = chat_with_tools.invoke(messages)
        return final_response.content

    return response.content
