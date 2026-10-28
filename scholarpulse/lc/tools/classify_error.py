"""LangChain Tool: classify_error"""
from langchain_core.tools import tool

@tool
def classify_error(query: str = "") -> dict:
    """classify_error：学情助手可用工具。"""
    return {"tool": "classify_error", "query": query, "ok": True}
