"""
间隔重复 & 遗忘曲线引擎（SM-2 简化版）。
创新点14/15/16：根据上次答题时间自动安排复习节点，长期不碰的概念掌握度回落。
纯规则计算，不依赖 LLM。
"""
from __future__ import annotations
from datetime import datetime, timedelta
from ..models import Mastery

# SM-2 间隔（天）：第1次复习后 1天，再 3天，7天，15天
INTERVALS = [1, 3, 7, 15, 30]


def due_review(mastery: dict[str, Mastery], now: datetime | None = None) -> list[str]:
    """返回今天需要复习的概念 cid（距上次更新超过其对应间隔）。"""
    now = now or datetime.now()
    due = []
    for cid, m in mastery.items():
        try:
            last = datetime.strptime(m.updated_at, "%Y-%m-%d %H:%M:%S")
        except Exception:
            due.append(cid); continue
        stage = min(m.n_correct, len(INTERVALS) - 1)
        gap = INTERVALS[stage]
        if now - last >= timedelta(days=gap):
            due.append(cid)
    return due


def forgetting_decay(mastery: dict[str, Mastery], now: datetime | None = None,
                     days: int = 15) -> dict[str, Mastery]:
    """模拟经过 days 天未复习：掌握度按指数曲线回落（但不低于先验）。"""
    now = now or datetime.now()
    for cid, m in mastery.items():
        m.p = max(0.3, m.p * (0.92 ** days))
    return mastery
