"""
【元认知评审智能体】—— 对自己的诊断再质疑一层。
前面的智能体负责"给出结论"，这个负责"否定自己"：
检查诊断是否样本不足、是否过度推断、是否有其他解释。
对应生成式 AI 的 self-critique / chain-of-verification。
"""
from __future__ import annotations
from ..llm import get_llm

SYS = "你是苛刻的复审专家。下面是系统给学生的学习诊断，请挑毛病：证据是否够、有没有以偏概全、有没有其他解释。输出JSON:{\"self_doubt\":\"...\",\"stronger_advice\":\"...\"}"


def self_review(diagnosis: str, sample_size: int) -> dict:
    try:
        return get_llm().chat_json(SYS, f"诊断:{diagnosis}\n答题样本数:{sample_size}")
    except Exception:
        return {"self_doubt": "样本较少，结论仅供参考", "stronger_advice": "再做3道题确认后再下结论"}
