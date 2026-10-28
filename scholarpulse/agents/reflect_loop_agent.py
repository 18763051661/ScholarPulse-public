"""
【进阶创新】反思式多智能体循环（Reflective Multi-Agent Loop）。
这是比单轮 tool-calling 高一个层次的 LangChain 模式：
  出题者(generator) -> 评审者(critic) -> 修订者(revise) -> 再评审 ... 直到通过或达上限。
对应论文思想：Reflexion / Multi-agent Debate。
"""
from __future__ import annotations
from ..llm import get_llm

SYS_GEN = "你是命题老师。基于考点出一道简答题。输出JSON:{\"question\":\"...\",\"answer\":\"...\"}"
SYS_CRITIC = "你是严格的审题人。判断题目是否有超纲、歧义、答案不唯一、与讲义脱节等问题。输出JSON:{\"pass\":true或false,\"issues\":[...],\"verdict\":\"...\"}"
SYS_REVISE = "根据评审意见修改题目。输出JSON:{\"question\":\"...\",\"answer\":\"...\"}"


def reflective_quiz(concept_brief: str, material_excerpt: str, max_rounds: int = 3) -> dict:
    log = []
    q = get_llm().chat_json(SYS_GEN, f"考点:{concept_brief}")
    for rnd in range(1, max_rounds + 1):
        critic = get_llm().chat_json(SYS_CRITIC, f"题目:{q.get('question')}\n讲义节选:{material_excerpt[:1500]}")
        log.append({"round": rnd, "pass": critic.get("pass"), "issues": critic.get("issues", [])})
        if critic.get("pass"):
            return {"final": q, "rounds": log, "status": "approved"}
        q = get_llm().chat_json(SYS_REVISE, f"原题目:{q}\n评审意见:{critic.get('issues')}")
    return {"final": q, "rounds": log, "status": "max_rounds"}
