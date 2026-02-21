"""
Basic AI Agent WITHOUT MEMORY
Model + prompt only - Does not remember previous conversations

This is an example of what happens when an agent has NO memory:
- Each message is independent
- It doesn't remember what you said before
- It cannot maintain conversation context
"""

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())


# ============================================
#  CONVERSATION LOOP
# ============================================
def main():
    print("=" * 50)
    print("🤖 ChattyBot - Agent WITHOUT MEMORY")
    print("=" * 50)
    print("⚠️  This agent does NOT remember previous messages")
    print("Type 'exit' to quit.\n")

    chain = _get_chat_chain()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        response = _chat_with_agent(chain, user_input)
        print(f"\n🤖 ChattyBot: {response}\n")


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
Always respond in English.""",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt | chat


# ============================================
# SIMPLE FUNCTION WITHOUT MEMORY
# ============================================
def _chat_with_agent(chain, user_message: str) -> str:
    """
    Sends a message to the agent.
    ⚠️ Does NOT maintain history - each message is independent.
    """
    try:
        response = chain.invoke({"input": user_message})
        return response.content
    except Exception as e:
        return f"Error connecting to the model: {e}"
