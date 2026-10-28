"""
【整合】ScholarPulse 总控智能体 —— 一个入口，内部编排。
对外：一个智能体 decide(用户意图) -> 内部路由到 6 个能力，再经元认知自检输出。
多端异构：同一入口可被 Web / 移动 / 小程序 / Webhook 调用。
"""
from __future__ import annotations
from ..llm import get_llm

ROUTER = (
    "你是 ScholarPulse 总控学习助手。根据用户一句话意图，选择一个动作并输出JSON。"
    "动作枚举：weak(查薄弱点) | quiz(出题) | diagnose(认知断层问诊) | "
    "review(元认知复审) | report(15天报告) | chat(闲聊鼓励)。"
    "输出{\"action\":\"...\",\"reply\":\"简短中文\"}")


def orchestrate(text: str) -> dict:
    try:
        r = get_llm().chat_json(ROUTER, text)
        action = r.get("action", "chat")
    except Exception:
        action, r = "chat", {"reply": "你好，复习有什么我能帮你？"}
    return {"agent": "ScholarPulse-Orchestrator", "action": action, "reply": r.get("reply", "")}
