"""
AI Agent with Conversation History in PostgreSQL
Maintains persistent memory across conversations

This agent:
- Stores each message in PostgreSQL
- Remembers previous conversations
- Can resume conversations by session_id
"""

import os
import uuid
from urllib.parse import quote_plus

import psycopg
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory

# Searches for .env in the current folder or parent folders
load_dotenv(find_dotenv())

# ============================================
# 1. DATABASE CONFIGURATION
# ============================================
# Expected environment variables in .env (at project root):
# DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")  # Default: 5432
DB_NAME = os.getenv("DB_NAME", "postgres")  # Default: postgres

if not all([DB_USER, DB_PASSWORD, DB_HOST]):
    raise ValueError(
        "❌ Missing database variables in .env\n"
        "Required: DB_USER, DB_PASSWORD, DB_HOST\n"
        "Optional: DB_PORT (default: 5432), DB_NAME (default: postgres)"
    )

# Build the connection URL (quote_plus handles special characters like @)
DATABASE_URL = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================
# 9. CONVERSATION LOOP
# ============================================
def main():
    print(f"🔌 Connecting as: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    _create_history_table()
    print("=" * 60)
    print("🤖 ChattyBot - Agent WITH PERSISTENT MEMORY (PostgreSQL)")
    print("=" * 60)

    print("\nSession options:")
    print("  1. New conversation")
    print("  2. Resume existing session (paste UUID)")

    session_id = _get_valid_session_id()
    full_chain = _get_chain_with_history()

    print(f"\n📝 Session ID: {session_id}")
    print("   (Save this ID to resume the conversation later)")
    print("✅ This agent REMEMBERS your previous messages")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print(f"\n💾 Your session has been saved.")
            print(f"   UUID: {session_id}")
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = _chat_with_agent(full_chain, user_input, session_id)
            print(f"\n🤖 ChattyBot: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


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
            print("⚠️ Invalid UUID. Creating a new session...")
            return str(uuid.uuid4())
    else:
        return str(uuid.uuid4())


# ============================================
# GET CHAIN WITH HISTORY
# ============================================
def _get_chain_with_history() -> RunnableWithMessageHistory:
    chain = _get_chat_chain()
    session_history = _get_session_history
    return RunnableWithMessageHistory(
        chain,
        session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


# ============================================
# GET CHAT CHAIN
# ============================================
def _get_chat_chain():
    chat = init_chat_model("gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful and friendly AI assistant called ChattyBot.
Answer the user's questions clearly and concisely.
You can remember previous conversations thanks to your persistent memory.
Always respond in English.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    return prompt | chat


# ============================================
# FUNCTION TO GET HISTORY FROM POSTGRESQL
# ============================================
def _get_session_history(session_id: str) -> PostgresChatMessageHistory:
    """
    Retrieves or creates the history for a session from PostgreSQL.
    """
    sync_connection = psycopg.connect(DATABASE_URL)
    return PostgresChatMessageHistory(
        "chat_history",
        session_id,
        sync_connection=sync_connection,
    )


# ============================================
# CHAT FUNCTION
# ============================================
def _chat_with_agent(full_chain, user_message: str, session_id: str) -> str:
    """
    Sends a message to the agent with persistent history.
    """
    try:
        # Invoke the full chain with the user message and session id
        response = full_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content
    except Exception as e:
        return f"Error sending message to the agent with persistent history: {e}"
