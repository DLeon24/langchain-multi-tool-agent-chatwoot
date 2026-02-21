"""
Main - AI Agent Orchestrator
=============================
Centralized entry point for running any agent.
"""

import importlib.util
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).parent


def load_module(folder_name: str, file_name: str):
    """Load a Python module from a folder that may contain hyphens."""
    path = BASE_DIR / folder_name / file_name
    spec = importlib.util.spec_from_file_location(
        file_name.replace(".py", ""), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show_menu():
    """Display the available agents menu."""
    print("\n" + "=" * 60)
    print("🤖 AI AGENT ORCHESTRATOR - AIPerupe Academy")
    print("=" * 60)
    print("\nAvailable agents:\n")
    print("  1. Basic Agent (no memory)")
    print("  2. Agent with Conversation History (PostgreSQL)")
    print("  3. Agent with History + Knowledge Base (RAG + Tool)")
    print("  4. Complete Agent (History + RAG + Internet)")
    print("\n  0. Exit")
    print("-" * 60)


def main():
    """Main orchestrator function."""
    while True:
        show_menu()

        try:
            option = input("\nSelect an agent (1/2/3/4 or 0): ").strip().upper()

            if option == "0":
                print("\nGoodbye! 👋\n")
                sys.exit(0)

            elif option == "1":
                module = load_module("01-agent-chat", "agent_chat.py")
                module.main()

            elif option == "2":
                module = load_module("02-agent-chat-memory", "agent_chat_memory.py")
                module.main()

            elif option == "3":
                module = load_module(
                    "03-agent-chat-memory-rag", "agent_chat_memory_rag.py"
                )
                module.main()

            elif option == "4":
                module = load_module(
                    "04-agent-chat-memory-rag-web", "agent_chat_memory_rag_web.py"
                )
                module.main()

            else:
                print("\n❌ Invalid option. Please try again.\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
