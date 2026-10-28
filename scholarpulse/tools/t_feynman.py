"""[011] t_feynman LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_feynman(arg: str = "") -> dict:
    """t_feynman 工具。"""
    return {"tool": "t_feynman", "arg": arg}
