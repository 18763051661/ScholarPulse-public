"""[001] cognition 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 cognition 分析，返回结构化结果。"""
    return {"agent": "cognition", "ok": True, "task": task}
