"""
学习游戏化：连续打卡、积分、成就徽章。
创新点56：用即时反馈提升坚持率，是陪伴助手的粘性核心。
"""
from __future__ import annotations
from ..models import AnswerRecord


def streak_days(records: list[AnswerRecord]) -> int:
    """按答题日期去重，算连续学习天数（从最近一次往前数）。"""
    days = sorted({r.ts[:10] for r in records})
    if not days:
        return 0
    from datetime import date, timedelta
    cur = date.fromisoformat(days[-1])
    streak = 0
    for d in reversed(days):
        if cur - date.fromisoformat(d) <= timedelta(days=streak):
            streak += 1
            cur = date.fromisoformat(d)
    return streak


def achievements(records: list[AnswerRecord], mastery_avg: float) -> list[str]:
    badges = []
    n = len(records)
    correct = sum(1 for r in records if r.correct)
    if n >= 5:
        badges.append("入门学徒（累计5题）")
    if correct >= 10:
        badges.append("十题连胜者")
    if mastery_avg >= 0.8:
        badges.append("考点达人（平均掌握≥80%）")
    if n >= 30:
        badges.append("刷题狂人")
    return badges or ["新手上路"]
