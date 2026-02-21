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
    Searches for information about AIPerupe Academy in the knowledge base.
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
    return _search_knowledge_base(query)


# ============================================
# INTERNAL FUNCTIONS
# ============================================
MIN_SIMILARITY = 0.3


def _search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Internal RAG search function.

    Args:
        query: Search query
        top_k: Number of documents to return

    Returns:
        str: Formatted information found
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)

        # Fetch documents from Supabase
        result = supabase_client.table(DOCUMENTS_TABLE).select("*").execute()

        if not result.data:
            return "No documents found in the knowledge base."

        # Compute similarity for each document
        scored_documents = []
        for doc in result.data:
            if doc.get("embedding"):
                doc_embedding = doc["embedding"]
                if isinstance(doc_embedding, str):
                    doc_embedding = json.loads(doc_embedding)

                doc_embedding = [float(x) for x in doc_embedding]
                score = _cosine_distance(query_embedding, doc_embedding)
                similarity = 1 - score

                if similarity >= MIN_SIMILARITY:
                    scored_documents.append(
                        {"content": doc.get("content", ""), "score": score}
                    )

        # Sort by similarity
        scored_documents.sort(key=lambda x: x["score"])
        top_docs = scored_documents[:top_k]

        if not top_docs:
            return "No relevant information found for your query."

        # Format results
        context = "Information found:\n\n"
        for i, doc in enumerate(top_docs, 1):
            similarity = 1 - doc["score"]
            context += f"[{i}] (Relevance: {similarity:.0%})\n{doc['content']}\n\n"

        return context

    except Exception as e:
        return f"Search error: {str(e)}"


def _cosine_distance(vec1, vec2):
    """Computes the cosine distance between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
