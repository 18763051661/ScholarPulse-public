"""LangChain Tool: get_streak"""
from langchain_core.tools import tool

@tool
def get_streak(query: str = "") -> dict:
    """get_streak：学情助手可用工具。"""
    return {"tool": "get_streak", "query": query, "ok": True}
