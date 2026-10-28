"""
【L2 层级治理多智能体】—— LangGraph Send API 的 Map-Reduce 治理
一个学区总监如何"管住" 80 个班？不是逐个看，而是组织结构本身 智能：
  Map   ：80 个教师智能体 并行（LangGraph Send 扇出 + 线程池），各出一份纯统计班级画像
  筛选  ：两级注意力——统计筛查全部 80 班，只有 Top-K 风险班级才值得花 LLM 深析（成本守恒）
  深析  ：K 个教研督导智能体并行，LLM 做根因归因与干预建议
  Reduce：学区总监智能体把 80 份画像 + K 份深析归约为全局态势报告与资源调配令
关键工程点：Send 扇出后汇聚节点（screen/director）会被触发 N 次，必须用 len() 幂等守卫；
并行节点只经返回值写 state（operator.add reducer 合并），禁止改共享可变对象。
"""
from __future__ import annotations
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .. import storage
from .. import mastery as M
from ..llm import get_llm
from ..models import ClassBrief, GovernanceReport

SYS_DEEP = """你是资深教研督导。给你一个班级的统计画像（平均掌握度、高危学生、共性薄弱考点）。
请做根因归因与干预设计：为什么这个班是这个状况（结合薄弱考点的先修关系推断卡点），下一步教师该最先做什么。
输出JSON: {"analysis":"3-4句根因分析与具体干预建议","action":"给该教师的一条最高优先级行动(一句话)"}
只依据给定数据推断。只输出JSON。"""

SYS_DIRECTOR = """你是学区总监。期末将至，你手里有全部班级的统计画像与重点班级的督导深析。
请输出全局态势判断：(1)学区分层态势与最需警觉的结构性问题（如两极分化/普遍性薄弱考点）；
(2)资源调配建议（教研支援该流向哪几类班、是否组织跨班教研）。
输出JSON: {"global_report":"5-8句全局报告，语气务实、有数字意识"}
只依据给定数据推断。只输出JSON。"""

SYS_CRITIC = """你是治理报告评审官（Reflexion 反思者）。给你学区治理报告与底层数据摘要。
审查报告是否：数字引用与数据摘要一致、资源调配建议具体可执行、覆盖了两极分化与共性薄弱考点。
若有实质缺陷，输出具体可操作的修订意见；若已足够好，明确通过。
输出JSON: {"ok": true或false, "critique": "1-3句修订意见（ok为true时输出空串）"}。只输出JSON。"""

TOP_K_DEFAULT = 5
RISK_LINE = 0.4          # 学生高危线：平均掌握度低于此
AT_RISK_FRACTION_W = 0.4
MAX_ROUNDS = 2           # Reflexion 最多 2 轮（初稿 + 一次修订）


# ---------- 教师智能体：纯统计画像（零 LLM，线程安全，只读存储） ----------
def class_profile(tid: str) -> dict:
    """一份班级画像：班均掌握、高危名单、共性薄弱考点、综合风险分。"""
    students = storage.load_students()
    roster = sorted(sid for sid, s in students.items() if s.teacher_id == tid)
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    per_concept: dict[str, list[float]] = {}
    avgs: list[float] = []
    at_risk: list[str] = []
    for sid in roster:
        m = storage.load_mastery(sid)
        if not m:
            continue
        avgs.append(M.avg_mastery(m))
        if M.avg_mastery(m) < RISK_LINE:
            at_risk.append(sid)
        for cid, cell in m.items():
            per_concept.setdefault(cid, []).append(cell.p)
    n = len(avgs)
    avg = round(sum(avgs) / n, 4) if n else 0.0
    weak = sorted(((cid, sum(ps) / len(ps)) for cid, ps in per_concept.items()),
                  key=lambda x: x[1])[:3]
    weak_names = [cmap[cid].name if cid in cmap else cid for cid, mp in weak if mp < 0.5]
    frac_risk = (len(at_risk) / n) if n else 1.0
    risk = round(0.6 * (1 - avg) + AT_RISK_FRACTION_W * frac_risk, 4)
    brief = ClassBrief(tid=tid, n_students=len(roster), avg_mastery=avg,
                       at_risk=at_risk, weak_concepts=weak_names, risk_score=risk)
    return brief.model_dump()


def rank_by_risk(briefs: list[dict]) -> list[dict]:
    return sorted(briefs, key=lambda b: -b["risk_score"])


# ---------- 教研督导智能体：LLM 深析（仅 Top-K） ----------
def deep_analyze(brief: dict) -> dict:
    user = (f"班级:{brief['tid']}(教师 {brief['tid']}) 人数:{brief['n_students']}\n"
            f"班均掌握度:{brief['avg_mastery']:.3f}  高危学生数:{len(brief['at_risk'])}\n"
            f"共性薄弱考点:{'、'.join(brief['weak_concepts']) or '（无低于0.5的考点）'}")
    analysis, action = "", ""
    try:
        data = get_llm().chat_json(SYS_DEEP, user)
        analysis, action = str(data.get("analysis", "")), str(data.get("action", ""))
    except Exception:
        analysis = (f"（LLM不可用，规则降级）该班平均掌握度 {brief['avg_mastery']:.2f}，"
                    f"高危率 {len(brief['at_risk'])}/{brief['n_students']}。"
                    f"建议优先针对 {'、'.join(brief['weak_concepts']) or '基础考点'} 组织小班重讲。")
        action = "按薄弱考点先修顺序重排本周复习计划"
    return {"brief": {**brief, "llm_analysis": f"{analysis} ▶ {action}"}}


# ---------- 学区总监智能体：Reduce 归约 ----------
def _data_digest(briefs: list[dict]) -> str:
    """底层数据摘要：给评审官对账用的客观事实。"""
    ranked = rank_by_risk(briefs)
    avgs = [b["avg_mastery"] for b in briefs if b["n_students"]]
    overall = sum(avgs) / len(avgs) if avgs else 0.0
    top = "; ".join(f"{b['tid']}(均{b['avg_mastery']:.2f},高危{len(b['at_risk'])}人)"
                    for b in ranked[:5])
    return f"共{len(briefs)}个班，全学区平均掌握度{overall:.3f}；风险Top5: {top}"


def merge_report(briefs: list[dict], deep: list[dict], feedback: str = "") -> str:
    ranked = rank_by_risk(briefs)
    all_avg = [b["avg_mastery"] for b in briefs if b["n_students"]]
    overall = sum(all_avg) / len(all_avg) if all_avg else 0.0
    worst = ranked[:5]
    deep_map = {d["brief"]["tid"]: d["brief"]["llm_analysis"] for d in deep}
    lines = [f"学区班级总数:{len(briefs)}  全学区平均掌握度:{overall:.3f}",
             "风险最高5个班: " + "; ".join(
                 f"{b['tid']}(均{b['avg_mastery']:.2f},高危{len(b['at_risk'])}人)" for b in worst)]
    if deep:
        lines.append("重点班督导深析: " + " | ".join(
            f"{d['brief']['tid']}:{deep_map.get(d['brief']['tid'], '')[:80]}" for d in deep[:3]))
    user = "班级画像汇总:\n" + "\n".join(lines)
    if feedback:
        user = f"上一版报告收到评审官如下修订意见，请据此重写：\n{feedback}\n\n{user}"
    try:
        data = get_llm().chat_json(SYS_DIRECTOR, user)
        return str(data.get("global_report", ""))
    except Exception:
        return (f"（LLM不可用，规则降级）全学区 {len(briefs)} 个班平均掌握度 {overall:.3f}；"
                f"风险最高的班级为 {', '.join(b['tid'] for b in worst)}。"
                f"建议教研支援优先流向风险分前列班级，并针对共性薄弱考点组织跨班专题教研。")


# ---------- LangGraph 图：Send API Map-Reduce + Reflexion 回环 ----------
class GovState(TypedDict, total=False):
    tids: list[str]
    top_k: int
    briefs: Annotated[list[dict], operator.add]   # 并行节点返回值由此 reducer 合并
    risk_tids: list[dict]
    deep: Annotated[list[dict], operator.add]
    global_report: str
    critique: str        # 评审官意见（director 消费后清空，防条件边重复触发）
    rounds: int          # director 已执行轮数


def _fanout_all(state: GovState) -> list[Send]:
    return [Send("teacher_stat", {"tid": t}) for t in state["tids"]]


def _node_teacher_stat(arg: dict) -> dict:
    return {"briefs": [class_profile(arg["tid"])]}


def _node_screen(state: GovState) -> dict:
    """汇聚屏障：被上游 80 条出边各触发一次，未收齐直接放行空更新。"""
    if len(state["briefs"]) < len(state["tids"]):
        return {}
    return {"risk_tids": rank_by_risk(state["briefs"])[:state.get("top_k", TOP_K_DEFAULT)]}


def _fanout_deep(state: GovState) -> list[Send]:
    return [Send("teacher_llm", b) for b in state.get("risk_tids", [])]


def _node_teacher_llm(brief: dict) -> dict:
    return {"deep": [deep_analyze(brief)]}


def _node_director(state: GovState) -> dict:
    if len(state.get("deep", [])) < len(state.get("risk_tids", [])):
        return {}
    # 第二轮起携带评审官的修订意见重写报告；critique 用后即清
    fb = state.get("critique", "")
    return {"global_report": merge_report(state["briefs"], state["deep"], feedback=fb),
            "rounds": state.get("rounds", 0) + 1, "critique": ""}


# ---------- 评审官智能体：Reflexion 反思节点 ----------
def _node_critic(state: GovState) -> dict:
    report = state.get("global_report", "")
    digest = _data_digest(state.get("briefs", []))
    try:
        data = get_llm().chat_json(SYS_CRITIC,
                                   f"【数据摘要】{digest}\n\n【待审报告】{report}")
        if data.get("ok"):
            return {"critique": ""}
        return {"critique": str(data.get("critique", ""))}
    except Exception:
        # LLM 不可用 → 视为通过（critique 留空），既不阻塞也绝不死循环
        return {"critique": ""}


def _route_after_critic(state: GovState) -> str:
    """评审不通过且还有修订预算 → 回环 director 重写；否则收口。"""
    if state.get("critique") and state.get("rounds", 1) < MAX_ROUNDS:
        return "revise"
    return "end"


def build_governance_graph():
    g = StateGraph(GovState)
    g.add_node("teacher_stat", _node_teacher_stat)
    g.add_node("screen", _node_screen)
    g.add_node("teacher_llm", _node_teacher_llm)
    g.add_node("director", _node_director)
    g.add_node("critic", _node_critic)
    g.add_conditional_edges(START, _fanout_all)
    g.add_edge("teacher_stat", "screen")
    g.add_conditional_edges("screen", _fanout_deep)
    g.add_edge("teacher_llm", "director")
    g.add_edge("director", "critic")                       # 每稿必审
    g.add_conditional_edges("critic", _route_after_critic,
                            {"revise": "director", "end": END})   # Reflexion 回环
    return g.compile()


def run_governance(top_k: int = TOP_K_DEFAULT) -> dict:
    """跑完整 Map-Reduce + Reflexion 治理并落盘 governance.json。
    llm_calls 口径 = len(deep) + rounds（只计报告生成类调用，评审调用不计）。"""
    teachers = storage.load_teachers()
    tids = sorted(teachers) or [f"T{i:03d}" for i in range(1, 81)]
    graph = build_governance_graph()
    final = graph.invoke({"tids": tids, "top_k": max(1, top_k)},
                         config={"max_concurrency": 8})
    briefs = sorted((ClassBrief(**b) for b in final["briefs"]), key=lambda b: b.tid)
    # 把 K 份 LLM 深析回写到对应班级画像（深析结果在 deep 分支，不在 briefs 里）
    deep_map = {d["brief"]["tid"]: d["brief"]["llm_analysis"] for d in final.get("deep", [])}
    for b in briefs:
        if b.tid in deep_map:
            b.llm_analysis = deep_map[b.tid]
    rounds = final.get("rounds", 1)
    rep = GovernanceReport(ts=storage.now_iso(), briefs=briefs,
                           global_report=final["global_report"],
                           critique=final.get("critique", ""),
                           rounds=rounds,
                           llm_calls=len(final.get("deep", [])) + rounds, top_k=top_k)
    storage.save_governance(rep)
    return rep.model_dump()
