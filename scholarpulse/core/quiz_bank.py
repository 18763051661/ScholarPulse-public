"""
题库 / 错题本 / 题型管理。
创新点20/22/23/24：错题自动回收、同类题再练、题型多样、易混淆对比题。
"""
from __future__ import annotations
from collections import defaultdict
from ..models import AnswerRecord, QuizItem


def wrong_book(records: list[AnswerRecord]) -> list[dict]:
    """从答题记录里挑出答错的题，按考点聚合。"""
    wrong = defaultdict(list)
    for r in records:
        if not r.correct:
            wrong[r.cid].append(r.qid)
    return [{"cid": cid, "wrong_count": len(qids), "qids": qids}
            for cid, qids in sorted(wrong.items(), key=lambda x: -len(x[1]))]


def needs_repeat(records: list[AnswerRecord], cid: str) -> bool:
    """某考点答错>=1次 且 之后没连续答对2次 -> 需要再练。"""
    tail = [r.correct for r in records if r.cid == cid]
    if not tail:
        return False
    wrongs = sum(1 for x in tail if not x)
    return wrongs > 0 and not (len(tail) >= 3 and all(tail[-2:]))


def build_bank(concepts, llm, per_tier: int = 3) -> list["QuizItem"]:
    """LLM 批量生成自适应题库：每概念 × 3 难度层（基础/进阶/迁移），一次生成永久落盘。"""
    from ..models import QuizItem
    items: list[QuizItem] = []
    tiers = ["foundation", "consolidation", "extension"]
    tier_desc = {"foundation": "基础记忆题", "consolidation": "理解应用题", "extension": "分析迁移题"}
    for c in concepts:
        for t in tiers:
            try:
                d = llm.chat_json(
                    f'针对考点出 {per_tier} 道简答题（{tier_desc[t]}）。'
                    '只输出 JSON：{"items":[{"q":"题干","a":"参考答案(≤60字)"}]}',
                    f"考点:{c.name} 简介:{c.summary}", max_tokens=700)
            except Exception:
                continue
            arr = d.get("items") if isinstance(d, dict) else None
            for x in (arr or []):
                if isinstance(x, dict) and x.get("q") and x.get("a"):
                    items.append(QuizItem(
                        qid=f"bank_{c.cid}_{t}_{len(items)}", cid=c.cid, tier=t,
                        question=str(x["q"]).strip(), reference_answer=str(x["a"]).strip(),
                        source_quote=c.summary[:40], grounded=True))
    return items


if __name__ == "__main__":
    # 一次性构建题库：python -m scholarpulse.core.quiz_bank
    from .. import storage
    from ..llm import get_llm
    cs = storage.load_concepts()
    print(f"概念 {len(cs)} 个，开始生成题库（每概念 3 层 × 3 题）…")
    bank = build_bank(cs, get_llm())
    storage.save_quiz_bank(bank)
    print(f"完成：{len(bank)} 道题 -> data/quiz_bank.json")
