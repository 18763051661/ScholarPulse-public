"""LangChain Tool: get_confidence_interval"""
from langchain_core.tools import tool

@tool
def get_confidence_interval(query: str = "") -> dict:
    """get_confidence_interval：学情助手可用工具。"""
    return {"tool": "get_confidence_interval", "query": query, "ok": True}
