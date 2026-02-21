"""
AI Agent with Tools + Conversation History in PostgreSQL

This agent:
- RAG as a Tool: The LLM decides when to search the knowledge base
- History: Stores conversations in PostgreSQL
- Extensible: Easy to add more Tools
"""

import os
import sys
import uuid
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv

# Searches for .env in the current folder or parent folders
load_dotenv(find_dotenv())

# Add root directory to path for importing tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_postgres import PostgresChatMessageHistory

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
    search_ai_perupe,
]

# ============================================
# 3. AGENT PROMPT
# ============================================
SYSTEM_PROMPT = """You are ChattyBot, a AIPerupe Academy assistant.

Your goal is to help users by answering their questions.

INSTRUCTIONS:
- For questions about AIPerupe Academy (programs, courses, prices, instructors), USE the search_ai_perupe tool.
- For greetings, thanks, or general conversation, respond directly WITHOUT using tools.
- You remember the entire conversation thanks to your persistent memory.
- Always respond in English in a clear and friendly manner.

EXAMPLES of when NOT to use tools:
- "Hello" → Respond with a greeting
- "Thanks" → Respond kindly
- "How are you?" → Respond conversationally

EXAMPLES of when to USE search_ai_perupe:
- "What courses do you offer?" → Use the "search_ai_perupe" tool
- "How much does the AI program cost?" → Use the "search_ai_perupe" tool
- "Who are the instructors?" → Use the "search_ai_perupe" tool"""


# ============================================
# CONVERSATION LOOP
# ============================================
def main():
    print(f"🔌 Connecting as: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    _create_history_table()

    print("=" * 60)
    print("🤖 ChattyBot - Agent with TOOLS + PERSISTENT MEMORY")
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
    print("✅ The agent DECIDES when to search the knowledge base")
    print("Type 'exit' to return to the menu.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["salir", "exit", "quit"]:
                print(f"\n💾 Your session has been saved.")
                print(f"   UUID: {session_id}")
                print("👋 See you later!")
                break

            if not user_input:
                continue

            try:
                answer = _chat_with_agent(
                    chat_with_tools, sync_connection, user_input, session_id
                )
                print(f"\n🤖 ChattyBot: {answer}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    finally:
        sync_connection.close()


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
    else:
        return str(uuid.uuid4())


# ============================================
# GET CHAT MODEL WITH TOOLS
# ============================================
def _get_chat_with_tools():
    """Initializes the chat model and binds the available tools."""
    chat = init_chat_model("gpt-4o", temperature=0.7)
    return chat.bind_tools(TOOLS)


# ============================================
# CHAT FUNCTION WITH AGENT + TOOLS
# ============================================
def _chat_with_agent(
    chat_with_tools, sync_connection, user_message: str, session_id: str
) -> str:
    """
    Runs the agent with tools and memory.
    The agent decides whether to use tools or respond directly.
    """
    history = _get_session_history(sync_connection, session_id)
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
        final_answer = final_response.content
    else:
        final_answer = response.content

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
