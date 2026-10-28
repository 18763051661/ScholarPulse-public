"""[009] t_flashcard LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_flashcard(arg: str = "") -> dict:
    """t_flashcard 工具。"""
    return {"tool": "t_flashcard", "arg": arg}
