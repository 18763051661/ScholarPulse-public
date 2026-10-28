"""LangChain Tool: get_review_due"""
from langchain_core.tools import tool

@tool
def get_review_due(query: str = "") -> dict:
    """get_review_due：学情助手可用工具。"""
    return {"tool": "get_review_due", "query": query, "ok": True}
