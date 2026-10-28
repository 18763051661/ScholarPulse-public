"""错因分类智能体：把失分归为概念/计算/审题/记忆模糊。"""
from ..llm import get_llm

SYS = "你是教学诊断专家。输出JSON:{\"type\":\"概念错误|计算错误|审题错误|记忆模糊\",\"advice\":\"...\"}"

def diagnose(question: str, reference: str, answer: str) -> dict:
    try:
        return get_llm().chat_json(SYS, f"题:{question}\n参考:{reference}\n作答:{answer}")
    except Exception:
        return {"type": "概念错误", "advice": "重读定义并重做2题"}
