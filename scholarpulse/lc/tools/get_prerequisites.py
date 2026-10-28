"""LangChain Tool: get_prerequisites"""
from langchain_core.tools import tool

@tool
def get_prerequisites(query: str = "") -> dict:
    """get_prerequisites：学情助手可用工具。"""
    return {"tool": "get_prerequisites", "query": query, "ok": True}
