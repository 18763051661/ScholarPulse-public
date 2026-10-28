"""[006] t_report LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_report(arg: str = "") -> dict:
    """t_report 工具。"""
    return {"tool": "t_report", "arg": arg}
