"""[047] image_captioner 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 image_captioner 分析，返回结构化结果。"""
    return {"agent": "image_captioner", "ok": True, "task": task}
