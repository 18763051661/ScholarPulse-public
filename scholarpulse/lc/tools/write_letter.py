"""LangChain Tool: write_letter"""
from langchain_core.tools import tool

@tool
def write_letter(query: str = "") -> dict:
    """write_letter：学情助手可用工具。"""
    return {"tool": "write_letter", "query": query, "ok": True}
