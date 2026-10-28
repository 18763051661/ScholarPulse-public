"""LangChain Tool: get_mastery"""
from langchain_core.tools import tool

@tool
def get_mastery(query: str = "") -> dict:
    """get_mastery：学情助手可用工具。"""
    return {"tool": "get_mastery", "query": query, "ok": True}
