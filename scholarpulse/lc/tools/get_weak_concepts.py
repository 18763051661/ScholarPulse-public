"""LangChain Tool: get_weak_concepts"""
from langchain_core.tools import tool

@tool
def get_weak_concepts(query: str = "") -> dict:
    """get_weak_concepts：学情助手可用工具。"""
    return {"tool": "get_weak_concepts", "query": query, "ok": True}
