"""
Tool: Knowledge Base (RAG with Pinecone)
Retrieves relevant information from the AIPerupe Academy knowledge base stored in Pinecone.

Index details (default):
- name: langchain-pinecone-sales-assistant
- vector type: dense
- dimension: 1536
- metric: cosine
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(find_dotenv())


# ============================================
# PINECONE CONFIGURATION
# ============================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "langchain-pinecone-sales-assistant"
)

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError(
        "❌ Missing Pinecone variables in .env\n"
        "Required: PINECONE_API_KEY, PINECONE_INDEX_NAME"
    )

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Connect to existing Pinecone index
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)


# ============================================
# EXPORTABLE TOOL
# ============================================
@tool
def search_details_ai_perupe(query: str) -> str:
    """
    Searches the AIPerupe Academy knowledge base stored in Pinecone (RAG) for relevant information.

    Use this tool when the user asks about:
    - Details of AIPerupe Academy programs (Live and On-Demand)
    - Strategic alliances
    - Benefits and support

    Args:
        query: The question or topic to search for.
    """
    print(f"   🔍 Searching (Pinecone): '{query}'")
    return _retrieve_from_knowledge_base(query)


# ============================================
# INTERNAL RETRIEVAL FUNCTION
# ============================================
def _retrieve_from_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    RAG retrieval using Pinecone: run similarity search and return the top_k chunks.

    Args:
        query: User question or topic to retrieve context for.
        top_k: Maximum number of chunks to return (default is 5).

    Returns:
        A formatted string with the retrieved chunks and relevance scores,
        or an error message if retrieval fails.
    """
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

        if not docs_with_scores:
            return "No relevant information found in the Pinecone knowledge base."

        context = "Information found:\n\n"
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            try:
                relevance_str = f"{float(score):.0%}"
            except (TypeError, ValueError):
                relevance_str = "N/A"
            context += f"[{i}] (Relevance: {relevance_str})\n{doc.page_content}\n\n"

        return context

    except Exception as e:
        return f"Retrieval error (Pinecone): {str(e)}"
