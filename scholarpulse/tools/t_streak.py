"""[017] t_streak LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_streak(arg: str = "") -> dict:
    """t_streak 工具。"""
    return {"tool": "t_streak", "arg": arg}
