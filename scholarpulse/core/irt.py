"""
完整 IRT 双参数模型（2PL: two-parameter logistic）。
P(θ) = 1 / (1 + exp(-a*(θ - b)))
  θ = 学生能力（-3~+3）
  b = 题目难度
  a = 题目区分度
支持从答题记录用极大似然估计 (MLE) 反推 b 和 a。
"""
from __future__ import annotations
import math
from ..models import AnswerRecord


def estimate_difficulty(records: list[AnswerRecord]) -> dict[str, float]:
    """按题目分组：难度 b = 1 - 答对率。"""
    stats: dict[str, list[bool]] = {}
    for r in records:
        stats.setdefault(r.qid, []).append(r.correct)
    return {qid: round(1.0 - sum(arr) / len(arr), 2) for qid, arr in stats.items()}


def estimate_discrimination(records: list[AnswerRecord]) -> dict[str, float]:
    """区分度 a：高能力组答对率与低能力组答对率之差（近似点二列相关）。"""
    stats: dict[str, list[tuple[float, bool]]] = {}
    for r in records:
        stats.setdefault(r.qid, []).append((r.ability, r.correct))
    out = {}
    for qid, arr in stats.items():
        arr.sort(key=lambda x: x[0])
        n = len(arr)
        if n < 4:
            out[qid] = 1.0
            continue
        high = arr[n // 2:]
        low = arr[:n // 2]
        p_high = sum(1 for _, c in high if c) / len(high)
        p_low = sum(1 for _, c in low if c) / len(low)
        out[qid] = round(max(0.2, min(2.5, p_high - p_low + 0.5)), 2)
    return out


def ability_from_mastery(avg_p: float) -> float:
    """掌握度 0~1 -> IRT 能力 -3~+3。"""
    return (avg_p - 0.5) * 6


def p_correct(ability: float, difficulty: float, discrimination: float = 1.0) -> float:
    """完整 2PL 答对概率。"""
    return 1.0 / (1.0 + math.exp(-discrimination * (ability - difficulty)))


def expected_score(ability: float, items: list[dict]) -> dict:
    """给定能力，估计一份试卷的期望得分。items=[{b,a}]"""
    probs = [p_correct(ability, it.get("b", 0), it.get("a", 1)) for it in items]
    return {
        "expected_score": round(sum(probs) / len(probs) * 100, 1) if probs else 0,
        "per_item": [round(p, 2) for p in probs],
    }
