"""[005] t_answer LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_answer(arg: str = "") -> dict:
    """t_answer 工具。"""
    return {"tool": "t_answer", "arg": arg}
