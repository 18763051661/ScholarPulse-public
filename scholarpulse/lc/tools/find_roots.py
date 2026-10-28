"""LangChain Tool: find_roots"""
from langchain_core.tools import tool

@tool
def find_roots(query: str = "") -> dict:
    """find_roots：学情助手可用工具。"""
    return {"tool": "find_roots", "query": query, "ok": True}
