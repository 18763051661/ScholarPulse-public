"""[014] t_teacher LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_teacher(arg: str = "") -> dict:
    """t_teacher 工具。"""
    return {"tool": "t_teacher", "arg": arg}
