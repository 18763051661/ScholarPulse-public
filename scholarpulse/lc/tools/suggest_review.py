"""LangChain Tool: suggest_review"""
from langchain_core.tools import tool

@tool
def suggest_review(query: str = "") -> dict:
    """suggest_review：学情助手可用工具。"""
    return {"tool": "suggest_review", "query": query, "ok": True}
