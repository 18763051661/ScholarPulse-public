"""LangChain Tool: make_flashcards"""
from langchain_core.tools import tool

@tool
def make_flashcards(query: str = "") -> dict:
    """make_flashcards：学情助手可用工具。"""
    return {"tool": "make_flashcards", "query": query, "ok": True}
