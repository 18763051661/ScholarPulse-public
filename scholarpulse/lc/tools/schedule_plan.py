"""LangChain Tool: schedule_plan"""
from langchain_core.tools import tool

@tool
def schedule_plan(query: str = "") -> dict:
    """schedule_plan：学情助手可用工具。"""
    return {"tool": "schedule_plan", "query": query, "ok": True}
