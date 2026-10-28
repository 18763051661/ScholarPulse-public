"""
高级原创智能模块（High-Level Original Features）：
- 错因分类：概念错/计算错/审题错/记忆错
- 概念混淆矩阵：发现学生把哪两个考点搞混了
- 知识缺口根因追溯：在前置图上回溯到最底层根节点
- 费曼模式：让学生用自己的话讲一遍，AI 找漏洞
- 掌握度置信区间：答题样本少时掌握度不可信
- 考试题型预测：判断该考点适合出简答/计算/论述
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept, Mastery, AnswerRecord


def error_taxonomy(question: str, reference: str, answer: str, score: float) -> dict:
    """把一次失分归类，给可操作的改进方向。"""
    if score >= 0.6:
        return {"type": "无失分", "advice": "保持"}
    sys = """你是教学诊断专家。判断学生失分类型，输出JSON:{"type":"概念错误|计算错误|审题错误|记忆模糊","advice":"一句话改进"}"""
    try:
        return get_llm().chat_json(sys, f"题:{question}\n参考:{reference}\n学生作答:{answer}")
    except Exception:
        return {"type": "概念错误", "advice": "重读该考点定义并重做2题"}


def confusion_matrix(records: list[AnswerRecord], concepts: list[Concept]) -> list[dict]:
    """同一批题里相邻考点接连答错 -> 疑似概念混淆对。"""
    weak_pairs: dict[tuple, int] = {}
    cmap = {c.cid: c.name for c in concepts}
    wrong_seq = [r.cid for r in records if not r.correct]
    for i in range(len(wrong_seq) - 1):
        a, b = wrong_seq[i], wrong_seq[i + 1]
        if a != b:
            key = tuple(sorted([a, b]))
            weak_pairs[key] = weak_pairs.get(key, 0) + 1
    out = [{"pair": f"{cmap.get(a,a)} ≈ {cmap.get(b,b)}", "co_wrong": n}
           for (a, b), n in weak_pairs.items() if n >= 2]
    return sorted(out, key=lambda x: -x["co_wrong"])


def root_cause(cid: str, mastery: dict[str, Mastery], concepts: list[Concept]) -> list[str]:
    """在前置依赖图上沿 prerequisites 回溯，找到最初的根薄弱点。"""
    cmap = {c.cid: c for c in concepts}
    chain = []
    seen = set()
    cur = cid
    while cur in cmap and cur not in seen:
        seen.add(cur)
        chain.append(cmap[cur].name)
        pres = cmap[cur].prerequisites
        weak_pre = [p for p in pres if p in mastery and mastery[p].p < 0.6]
        if not weak_pre:
            break
        cur = weak_pre[0]
    return chain


def confidence_interval(p: float, n: int) -> tuple[float, float]:
    """Beta 后验近似：样本越少区间越宽，提醒'掌握度还不够信'。"""
    alpha = 1 + n * p
    beta = 1 + n * (1 - p)
    mean = alpha / (alpha + beta)
    width = 1.96 * (mean * (1 - mean) / (alpha + beta)) ** 0.5 if (alpha + beta) > 0 else 0.5
    return round(max(0, mean - width), 2), round(min(1, mean + width), 2)


def feynman_check(concept_name: str, student_explain: str) -> dict:
    """费曼学习法：学生用自己的话讲概念，AI 找讲错/讲漏的地方。"""
    sys = """你是费曼教练。判断学生能否用大白话把概念讲清楚。
输出JSON:{"clear":true或false,"gaps":["讲漏/讲错的点"],"fix":"一句话补讲"}"""
    try:
        return get_llm().chat_json(sys, f"概念:{concept_name}\n学生讲解:{student_explain or '(空)'}")
    except Exception:
        return {"clear": False, "gaps": ["未提供讲解"], "fix": "试着用一句话向同学解释这个概念"}
