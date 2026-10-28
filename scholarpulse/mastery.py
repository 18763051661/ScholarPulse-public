"""
贝叶斯掌握度引擎（纯计算，不调用 LLM）。
这是系统的"记忆"核心：每个概念的掌握度是一个概率 p∈[0,1]，
答对向上收敛，答错向下掉；并据此推断学生层级与薄弱点。
"""
from __future__ import annotations
from .config import settings
from .models import Mastery, Concept
from . import storage

PRIOR = 0.3        # 新概念先验
GAIN = 0.25        # 答对：p += (1-p)*GAIN
LOSS = 0.35        # 答错：p *= (1-LOSS)


def init_mastery(concepts: list[Concept]) -> dict[str, Mastery]:
    """为每个核心/可选概念建立先验掌握度（被过滤掉的垃圾知识点不建）。"""
    m = {}
    now = storage.now_iso()
    for c in concepts:
        if c.noise == "filtered":
            continue
        m[c.cid] = Mastery(cid=c.cid, p=PRIOR, updated_at=now)
    return m


def apply_answer(m: Mastery, correct: bool) -> Mastery:
    if correct:
        m.p = min(1.0, m.p + (1 - m.p) * GAIN)
        m.n_correct += 1
    else:
        m.p = max(0.05, m.p * (1 - LOSS))
        m.n_wrong += 1
    m.updated_at = storage.now_iso()
    return m


def student_tier(avg_p: float) -> str:
    if avg_p < 0.45:
        return "foundation"
    if avg_p < 0.7:
        return "consolidation"
    return "extension"


def weak_concepts(mastery: dict[str, Mastery], concepts: list[Concept]) -> list[Concept]:
    """返回未达阈值、且不是被过滤的概念。"""
    cmap = {c.cid: c for c in concepts}
    out = []
    for cid, m in mastery.items():
        if m.p < settings.weak_threshold and cid in cmap:
            out.append(cmap[cid])
    # 按"薄弱程度 × 考试权重"排序：越差、越可能考的越优先
    out.sort(key=lambda c: (mastery[c.cid].p * (1 - c.exam_weight)))
    return out


def avg_mastery(mastery: dict[str, Mastery]) -> float:
    if not mastery:
        return 0.0
    return sum(m.p for m in mastery.values()) / len(mastery)
