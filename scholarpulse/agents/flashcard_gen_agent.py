"""[028] flashcard_gen 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 flashcard_gen 分析，返回结构化结果。"""
    return {"agent": "flashcard_gen", "ok": True, "task": task}
