"""[080] outlier_detector 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 outlier_detector 分析，返回结构化结果。"""
    return {"agent": "outlier_detector", "ok": True, "task": task}
