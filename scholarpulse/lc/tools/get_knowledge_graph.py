"""LangChain Tool: get_knowledge_graph"""
from langchain_core.tools import tool

@tool
def get_knowledge_graph(query: str = "") -> dict:
    """get_knowledge_graph：学情助手可用工具。"""
    return {"tool": "get_knowledge_graph", "query": query, "ok": True}
