"""LangChain Tool: compare_concepts"""
from langchain_core.tools import tool

@tool
def compare_concepts(query: str = "") -> dict:
    """compare_concepts：学情助手可用工具。"""
    return {"tool": "compare_concepts", "query": query, "ok": True}
