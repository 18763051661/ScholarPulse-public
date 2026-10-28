"""LangChain Tool: estimate_cognitive_load"""
from langchain_core.tools import tool

@tool
def estimate_cognitive_load(query: str = "") -> dict:
    """estimate_cognitive_load：学情助手可用工具。"""
    return {"tool": "estimate_cognitive_load", "query": query, "ok": True}
