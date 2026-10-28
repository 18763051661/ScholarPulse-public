"""[034] outline_gen 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 outline_gen 分析，返回结构化结果。"""
    return {"agent": "outline_gen", "ok": True, "task": task}
