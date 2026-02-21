"""
Tool: Internet Search (Tavily)
Allows searching for up-to-date information on the internet.
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import tool

load_dotenv(find_dotenv())

# ============================================
# TAVILY CONFIGURATION
# ============================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "❌ Missing TAVILY_API_KEY in .env\n"
        "Get your free API key at: https://tavily.com"
    )

try:
    from langchain_tavily import TavilySearch

    tavily_search = TavilySearch(max_results=5)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

    tavily_search = TavilySearchResults(max_results=5)


# ============================================
# EXPORTABLE TOOL
# ============================================
@tool
def search_internet(query: str) -> str:
    """
    Searches for up-to-date information on the internet using Tavily.
    Use this tool when the user asks about:
    - Current events or recent news
    - Information that changes frequently
    - Data not available in the AIPerupe Academy knowledge base
    - Any topic that requires up-to-date internet information

    Do NOT use this tool for:
    - Questions about AIPerupe Academy (use search_ai_perupe)
    - Greetings or general conversation

    Args:
        query: The question or topic to search on the internet
    """
    print(f"   🌐 Searching the internet: '{query}'")
    return _search_internet(query)


# ============================================
# INTERNAL FUNCTIONS
# ============================================
MAX_CONTENT_LENGTH = 500


def _search_internet(query: str) -> str:
    """
    Internal internet search function.

    Args:
        query: Search query

    Returns:
        str: Formatted information found
    """
    try:
        results = tavily_search.invoke(query)

        if not results:
            return "No relevant information found on the internet."

        if not isinstance(results, list):
            return f"Information found on the internet:\n\n{results}"

        context = "Information found on the internet:\n\n"
        for i, result in enumerate(results, 1):
            context += _format_result_entry(i, result)

        return context

    except Exception as e:
        return f"Search error: {str(e)}"


def _format_result_entry(index: int, result) -> str:
    """Formats a single search result entry."""
    if isinstance(result, dict):
        title = result.get("title", "No title")
        content = result.get("content", "")
        url = result.get("url", "")
    else:
        title = f"Result {index}"
        content = str(result)
        url = ""

    entry = f"[{index}] {title}\n"
    if len(content) > MAX_CONTENT_LENGTH:
        entry += f"{content[:MAX_CONTENT_LENGTH]}...\n"
    else:
        entry += f"{content}\n"
    if url:
        entry += f"Source: {url}\n"
    entry += "\n"

    return entry
