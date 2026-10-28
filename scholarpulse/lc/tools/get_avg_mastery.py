"""LangChain Tool: get_avg_mastery"""
from langchain_core.tools import tool

@tool
def get_avg_mastery(query: str = "") -> dict:
    """get_avg_mastery：学情助手可用工具。"""
    return {"tool": "get_avg_mastery", "query": query, "ok": True}
