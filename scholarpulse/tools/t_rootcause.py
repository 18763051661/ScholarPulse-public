"""[012] t_rootcause LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_rootcause(arg: str = "") -> dict:
    """t_rootcause 工具。"""
    return {"tool": "t_rootcause", "arg": arg}
