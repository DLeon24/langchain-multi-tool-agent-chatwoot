"""
Complete AI Agent: Knowledge Base + Internet + Conversation History
- Tool 1: Knowledge Base (RAG with Supabase)
- Tool 2: Internet Search (Tavily)
- History: Stores conversations in PostgreSQL
"""

import os
import sys
import uuid
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Add root directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_postgres import PostgresChatMessageHistory

# Import tools from the tools/ folder
from tools.knowledge_base import search_ai_perupe
from tools.internet_search import search_internet

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
        "Required: DB_USER, DB_PASSWORD, DB_HOST"
    )

DATABASE_URL = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print(f"🔌 Connecting as: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ============================================
# 2. AVAILABLE TOOLS LIST
# ============================================
tools = [
    search_ai_perupe,  # AIPerupe Academy Knowledge Base
    search_internet,  # Internet Search (Tavily)
]

# ============================================
# 3. MODEL CONFIGURATION WITH TOOLS
# ============================================
chat = init_chat_model("gpt-4o", temperature=0.7)
chat_with_tools = chat.bind_tools(tools)

# ============================================
# 4. AGENT PROMPT
# ============================================
system_prompt = """You are ChattyBot, a AIPerupe Academy AI assistant with internet access.

Your goal is to help users by answering their questions using the available tools.

AVAILABLE TOOLS:
1. search_ai_perupe: For information about AIPerupe Academy (programs, courses, prices, instructors)
2. search_internet: For up-to-date information from the internet (news, events, current data)

INSTRUCTIONS:
- For questions about AIPerupe Academy → USE search_ai_perupe
- For questions about current events, news, or general information → USE search_internet
- For greetings, thanks, or general conversation → Respond directly WITHOUT tools
- You can use BOTH tools if the question requires it
- You remember the entire conversation thanks to your persistent memory
- Always respond in a clear and friendly manner

EXAMPLES:
- "Hello" → Respond directly
- "What courses do you offer?" → Use search_ai_perupe
- "What happened in the news today?" → Use search_internet
- "How does your AI course compare to current trends?" → Use BOTH tools"""


# ============================================
# CREATE HISTORY TABLE
# ============================================
def create_history_table():
    try:
        sync_connection = psycopg.connect(DATABASE_URL)
        PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
        sync_connection.close()
    except Exception as e:
        print(f"⚠️ Note about table: {e}")


create_history_table()


# ============================================
# CONVERSATION HISTORY
# ============================================
def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    sync_connection = psycopg.connect(DATABASE_URL)
    return PostgresChatMessageHistory(
        "chat_history", session_id, sync_connection=sync_connection
    )


# ============================================
# CHAT FUNCTION WITH AGENT + TOOLS
# ============================================
def chat_with_agent(user_message: str, session_id: str) -> str:
    """
    Runs the agent with tools and memory.
    The agent decides whether to use tools or respond directly.
    """
    history = get_session_history(session_id)
    previous_messages = history.messages

    messages = [{"role": "system", "content": system_prompt}]

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

            for t in tools:
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
# 8. CONVERSATION LOOP
# ============================================
def main():
    print("=" * 60)
    print("🤖 ChattyBot - COMPLETE Agent (KB + Internet + Memory)")
    print("=" * 60)
    print("🔧 Available tools:")
    for t in tools:
        print(f"   - {t.name}")
    print("💾 History: PostgreSQL")

    print("\nSession options:")
    print("  1. New conversation")
    print("  2. Continue existing session (paste UUID)")

    option = input("\nChoose (1/2): ").strip()

    if option == "2":
        session_id = input("Paste the session UUID: ").strip()
        try:
            uuid.UUID(session_id)
        except ValueError:
            print("⚠️ Invalid UUID. Creating new session...")
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())

    print(f"\n📝 Session ID: {session_id}")
    print("   (Save this ID to continue later)")
    print("✅ The agent can search AIPerupe and the INTERNET")
    print("Type 'exit' to return to the menu.\n")

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
            answer = chat_with_agent(user_input, session_id)
            print(f"\n🤖 ChattyBot: {answer}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
