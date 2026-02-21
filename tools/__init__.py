"""
AI Agent Tools Module
Contains all available tools for the agents.
"""

from tools.knowledge_base import search_ai_perupe
from tools.internet_search import search_internet

__all__ = [
    "search_ai_perupe",
    "search_internet",
]
