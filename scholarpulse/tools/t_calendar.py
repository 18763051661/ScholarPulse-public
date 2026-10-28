"""[016] t_calendar LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_calendar(arg: str = "") -> dict:
    """t_calendar 工具。"""
    return {"tool": "t_calendar", "arg": arg}
