"""[004] t_quiz LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_quiz(arg: str = "") -> dict:
    """t_quiz 工具。"""
    return {"tool": "t_quiz", "arg": arg}
