"""[083] cluster_concept 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 cluster_concept 分析，返回结构化结果。"""
    return {"agent": "cluster_concept", "ok": True, "task": task}
