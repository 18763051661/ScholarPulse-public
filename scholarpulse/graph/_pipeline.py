"""
LangGraph 编排：把六个智能体串成"知识图谱 -> 过滤 -> 出题 -> 反幻觉校验 -> 答题 -> 掌握度更新 -> 再规划"的闭环。
关键创新：闭环会根据更新后的掌握度自动决定是否进入下一轮，而不是一次性出完题。
"""
from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from ..agents import ingest_agent, filter_agent, quiz_agent, verify_agent, planner_agent
from .. import mastery as M
from ..config import settings
from ..models import Concept, Mastery, AnswerRecord, ReportSnapshot
from .. import storage


class ScholarState(TypedDict, total=False):
    sid: str
    material_text: str
    material_name: str
    concepts: list[Concept]
    # 学习循环
    mastery: dict[str, Mastery]
    round: int
    weak: list[Concept]
    quiz: list
    student_tier: str
    advice: dict
    # 结果
    grounded_count: int
    dropped_count: int


def _node_ingest(state: ScholarState) -> ScholarState:
    concepts = ingest_agent.build_knowledge_graph(state["material_text"], state["material_name"])
    concepts = filter_agent.filter_noise(concepts)
    # 合并入库而非整体覆盖：同 cid 更新、新 cid 追加，全局概念库只增不减——
    # 防止上传单门课件把全学科概念库清成一门课（曾导致思维导图只剩操作系统）
    bank = {c.cid: c for c in storage.load_concepts()}
    for c in concepts:
        bank[c.cid] = c
    storage.save_concepts(list(bank.values()))
    state["concepts"] = concepts
    return state


def _node_init_mastery(state: ScholarState) -> ScholarState:
    state["mastery"] = M.init_mastery(state["concepts"])
    state["round"] = 0
    return state


def _node_plan(state: ScholarState) -> ScholarState:
    m = state["mastery"]
    state["student_tier"] = M.student_tier(M.avg_mastery(m))
    state["weak"] = M.weak_concepts(m, state["concepts"])
    state["advice"] = planner_agent.plan_advice(state["weak"], m)
    return state


def _node_quiz(state: ScholarState) -> ScholarState:
    limit = settings.max_quiz_per_round
    items = quiz_agent.make_quiz(state["weak"], state["student_tier"], limit)
    items = verify_agent.verify_quiz(items, state["concepts"], state["material_text"])
    state["quiz"] = items
    state["grounded_count"] = sum(1 for i in items if i.grounded)
    state["dropped_count"] = sum(1 for i in items if not i.grounded)
    return state


def _node_update_mastery(state: ScholarState, answers: list[AnswerRecord]) -> ScholarState:
    """答题后更新掌握度。answers 由 CLI/上层注入。"""
    m = state["mastery"]
    for a in answers:
        if a.cid in m:
            M.apply_answer(m[a.cid], a.correct)
    storage.save_mastery(state["sid"], m)
    storage.append_answers(state["sid"], answers)
    state["round"] = state.get("round", 0) + 1
    return state


def _should_continue(state: ScholarState) -> str:
    if state["round"] >= settings.max_loop_rounds:
        return "end"
    still_weak = [c for c in state["weak"] if state["mastery"][c.cid].p < settings.weak_threshold]
    return "loop" if still_weak else "end"


def build_pipeline():
    g = StateGraph(ScholarState)
    g.add_node("ingest", _node_ingest)
    g.add_node("init_mastery", _node_init_mastery)
    g.add_node("plan", _node_plan)
    g.add_node("quiz", _node_quiz)
    # 注意：update_mastery 需要外部答题结果，由 CLI 在 quiz 后手动注入，
    # 这里用 plan 作为循环回边点。
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "init_mastery")
    g.add_edge("init_mastery", "plan")
    g.add_edge("plan", "quiz")
    # quiz 之后暂停，等 CLI 收集答案 -> 调用 apply_answers -> 再回到 plan
    g.add_conditional_edges("quiz", lambda s: "hold", {"hold": END})
    return g.compile()


def build_loop_graph():
    """纯闭环图（无 ingest）：plan -> quiz -> END，由外部驱动回边。"""
    g = StateGraph(ScholarState)
    g.add_node("plan", _node_plan)
    g.add_node("quiz", _node_quiz)
    g.add_edge(START, "plan")
    g.add_edge("plan", "quiz")
    g.add_conditional_edges("quiz", _should_continue, {"loop": "plan", "end": END})
    return g.compile()
