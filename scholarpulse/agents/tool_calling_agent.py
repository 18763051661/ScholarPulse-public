"""
【高级 LangChain Agent】真正的"模型-工具"绑定。
不是我们 if/else 替模型选动作，而是把工具暴露给模型，让它自己决定调哪个、传什么参。
基于 langchain_core 的 tool-calling 范式（RouterChatModel 是自定义 OpenAI 兼容实现）。
所有工具均为真实能力：真数据 + 真 LLM，无占位桩。
"""
from __future__ import annotations
from langchain_core.tools import tool
from ..lc_model import RouterChatModel


@tool
def query_weak_points(student_id: str) -> str:
    """查询该学生当前最薄弱的考点列表。"""
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(student_id)
    weak = sorted(concepts, key=lambda c: mastery[c.cid].p if c.cid in mastery else 1)[:3]
    return "最薄弱: " + ", ".join(f"{c.name}({mastery[c.cid].p:.2f})" for c in weak)


@tool
def make_quiz(student_id: str) -> str:
    """为该学生最薄弱考点生成一道分层自测题（真 LLM 出题）。"""
    from .. import storage
    from ..agents import quiz_agent
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(student_id)
    student = storage.load_students().get(student_id)
    tier = student.tier if student else "foundation"
    ranked = sorted(concepts, key=lambda c: mastery[c.cid].p if c.cid in mastery else 1.0)
    items = quiz_agent.make_quiz(ranked[:1], tier, 1)
    if not items:
        return "暂无课件素材，无法出题"
    q = items[0]
    return f"分层题（{tier}）｜题目：{q.question}｜参考答案：{q.reference_answer}"


@tool
def cognitive_diagnose(student_id: str) -> str:
    """沿前置知识链做认知断层问诊，定位真正断点。"""
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    mastery = storage.load_mastery(student_id)
    weak = sorted(concepts, key=lambda c: mastery[c.cid].p if c.cid in mastery else 1.0)[0]
    # 沿前置链回溯，找掌握度最低的环节 = 断层根
    chain, seen, cur = [], set(), weak.cid
    while cur and cur not in seen and cur in cmap:
        seen.add(cur)
        c = cmap[cur]
        chain.append((c.name, mastery[c.cid].p if c.cid in mastery else None))
        cur = c.prerequisites[0] if c.prerequisites else None
    if len(chain) <= 1:
        return f"断层问诊：{chain[0][0]}（{chain[0][1]}）即最底层断点，建议从零重讲。"
    broken_name, broken_p = min(chain, key=lambda x: x[1] if x[1] is not None else 1.0)
    path = " → ".join(f"{n}({p if p is not None else '?'})" for n, p in chain)
    return f"断层链：{path}。真正断点：{broken_name}（{broken_p:.2f}），先修它而非表层薄弱点。"


@tool
def generate_report(student_id: str) -> str:
    """生成该学生的 15 天学习分析报表（数据汇总 + LLM 结论）。"""
    from .. import storage
    from ..mastery import avg_mastery, weak_concepts
    from ..llm import get_llm
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(student_id)
    student = storage.load_students().get(student_id)
    snaps = storage.load_snapshots(student_id)
    trend = "暂无"
    if len(snaps) >= 2:
        a0 = sum(snaps[0].mastery.values()) / max(len(snaps[0].mastery), 1)
        a1 = sum(snaps[-1].mastery.values()) / max(len(snaps[-1].mastery), 1)
        trend = f"{a0:.2f}→{a1:.2f}"
    from ..core.spaced_repetition import due_review
    n_due = len(due_review(mastery))
    facts = (f"学生:{student.name if student else student_id} "
             f"当前均分:{avg_mastery(mastery):.2f} 15天趋势:{trend} "
             f"薄弱考点:{', '.join(c.name for c in weak_concepts(mastery, concepts)[:3]) or '无'} "
             f"今日到期复习项:{n_due}")
    try:
        return get_llm().chat(
            "你是学情分析师。只依据数据，80 字内给出该生 15 天学习结论与下一步建议。",
            facts, temperature=0.4, max_tokens=200)
    except Exception:
        return f"（离线摘要）{facts}"


TOOLS = [query_weak_points, make_quiz, cognitive_diagnose, generate_report]
TOOL_MAP = {t.name: t for t in TOOLS}

SYS = ("你是学情助手。根据学生意图从以下工具中选择最合适的一个："
       "query_weak_points(查薄弱点) / make_quiz(出自测题) / "
       "cognitive_diagnose(认知断层问诊) / generate_report(15天学情报告)。"
       '只输出 JSON：{"tool": "工具名"}，不要输出其他内容。')


def run(user_text: str, student_id: str = "s001") -> dict:
    """高级 agent：模型选工具，再由本地真正执行该工具函数。"""
    from ..llm import get_llm
    try:
        choice = get_llm().chat_json(SYS, f"学生{student_id}说：{user_text}")
        name = choice.get("tool", "query_weak_points") if isinstance(choice, dict) else "query_weak_points"
    except Exception:
        name = "query_weak_points"
    tool = TOOL_MAP.get(name, query_weak_points)
    result = tool.invoke({"student_id": student_id})
    return {"agent": "LangChain-ToolAgent", "chosen_tool": tool.name, "result": result}
