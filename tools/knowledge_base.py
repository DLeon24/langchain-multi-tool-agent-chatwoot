"""
Tool: Knowledge Base (RAG with Supabase)
Allows searching for information in the AIPerupe Academy knowledge base.
"""

import json
import os

import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

load_dotenv(find_dotenv())

# ============================================
# SUPABASE CONFIGURATION
# ============================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError(
        "❌ Missing Supabase variables in .env\n"
        "Required: SUPABASE_URL, SUPABASE_SERVICE_KEY"
    )

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

DOCUMENTS_TABLE = "documents_langchain_sales_assistant"


# ============================================
# EXPORTABLE TOOL
# ============================================
@tool
def search_ai_perupe(query: str) -> str:
    """
    Searches the AIPerupe Academy knowledge base (RAG) for relevant information.
    Use this tool when the user asks about:
    - AIPerupe Academy programs
    - Courses and content
    - Teachers and instructors
    - Prices and modalities
    - Any information related to AIPerupe Academy

    Args:
        query: The question or topic to search for
    """
    print(f"   🔍 Searching: '{query}'")
    return _retrieve_from_knowledge_base(query)


# ============================================
# INTERNAL FUNCTIONS
# ============================================
MIN_SIMILARITY = 0.3


def _retrieve_from_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    RAG retrieval: embed the query, score by cosine similarity, and return the top_k chunks.

    Args:
        query: User question or topic to retrieve context for.
        top_k: Maximum number of chunks to return (default is 5).

    Returns:
        A formatted string containing the retrieved chunks and their relevance scores,
        or an error message if retrieval fails.
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)

        # Fetch documents from Supabase
        result = (
            supabase_client.table(DOCUMENTS_TABLE)
            .select("content, embedding")
            .execute()
        )

        if not result.data:
            return "No documents found in the knowledge base."

        # Score each chunk by cosine distance (lower = more similar)
        chunks_with_distance = []
        for row in result.data:
            if row.get("embedding"):
                doc_embedding = row["embedding"]
                if isinstance(doc_embedding, str):
                    doc_embedding = json.loads(doc_embedding)

                doc_embedding = [float(x) for x in doc_embedding]
                distance = _cosine_distance(query_embedding, doc_embedding)
                similarity = 1 - distance

                if similarity >= MIN_SIMILARITY:
                    chunks_with_distance.append(
                        {"content": row.get("content", ""), "distance": distance}
                    )

        # Sort by distance ascending (closest first)
        chunks_with_distance.sort(key=lambda x: x["distance"])
        top_chunks = chunks_with_distance[:top_k]

        if not top_chunks:
            return "No relevant information found for your query."

        # Format retrieved context
        context = "Information found:\n\n"
        for i, chunk in enumerate(top_chunks, 1):
            similarity = 1 - chunk["distance"]
            context += f"[{i}] (Relevance: {similarity:.0%})\n{chunk['content']}\n\n"

        return context

    except Exception as e:
        return f"Retrieval error: {str(e)}"


def _cosine_distance(vec1: list | np.ndarray, vec2: list | np.ndarray) -> float:
    """Computes the cosine distance between two vectors (1 - cosine_similarity)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
