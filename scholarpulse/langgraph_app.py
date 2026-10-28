"""
LangGraph 多智能体状态图（课程项目核心演示）。
用标准 StateGraph 编排：plan -> quiz -> answer(update mastery) -> 条件回边。
这是真正的 LangGraph 应用，而非手写循环。
"""
from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from . import mastery as M, storage
from .models import AnswerRecord
from .agents import quiz_agent, verify_agent, planner_agent
from .config import settings
import random

random.seed(7)


class LearnState(TypedDict, total=False):
    sid: str
    round: int
    weak_count: int


def _plan(state: LearnState) -> LearnState:
    sid = state["sid"]
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    state["weak_count"] = len(weak)
    return state


def _quiz(state: LearnState) -> LearnState:
    sid = state["sid"]
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    tier = M.student_tier(M.avg_mastery(mastery))
    items = quiz_agent.make_quiz(weak, tier, settings.max_quiz_per_round)
    text = open("data/sample_lecture.txt", encoding="utf-8").read()
    items = verify_agent.verify_quiz(items, concepts, text)
    state["_items"] = [i for i in items if i.grounded]
    return state


def _answer(state: LearnState) -> LearnState:
    sid = state["sid"]
    mastery = storage.load_mastery(sid)
    recs = []
    for q in state.get("_items", []):
        correct = random.random() < 0.6
        if q.cid in mastery:
            M.apply_answer(mastery[q.cid], correct)
        recs.append(AnswerRecord(qid=q.qid, cid=q.cid, correct=correct, ts=storage.now_iso()))
    storage.save_mastery(sid, mastery)
    storage.append_answers(sid, recs)
    state["round"] = state.get("round", 0) + 1
    return state


def _route(state: LearnState) -> str:
    if state.get("round", 0) >= settings.max_loop_rounds or state.get("weak_count", 0) == 0:
        return "end"
    return "loop"


def build_app():
    g = StateGraph(LearnState)
    g.add_node("plan", _plan)
    g.add_node("quiz", _quiz)
    g.add_node("answer", _answer)
    g.add_edge(START, "plan")
    g.add_edge("plan", "quiz")
    g.add_edge("quiz", "answer")
    g.add_conditional_edges("answer", _route, {"loop": "plan", "end": END})
    return g.compile()


def run(sid: str):
    app = build_app()
    final = app.invoke({"sid": sid, "round": 0})
    return final
