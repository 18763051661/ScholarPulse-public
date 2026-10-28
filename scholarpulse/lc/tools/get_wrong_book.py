"""LangChain Tool: get_wrong_book"""
from langchain_core.tools import tool

@tool
def get_wrong_book(query: str = "") -> dict:
    """get_wrong_book：学情助手可用工具。"""
    return {"tool": "get_wrong_book", "query": query, "ok": True}
