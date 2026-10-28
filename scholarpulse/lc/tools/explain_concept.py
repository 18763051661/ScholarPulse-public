"""LangChain Tool: explain_concept"""
from langchain_core.tools import tool

@tool
def explain_concept(query: str = "") -> dict:
    """explain_concept：学情助手可用工具。"""
    return {"tool": "explain_concept", "query": query, "ok": True}
