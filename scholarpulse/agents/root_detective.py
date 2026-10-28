"""🌳 根因追溯·侦探（LangChain ReAct 智能体）。
把"根因追溯"从静态列表升级为查案：侦探对最薄弱考点立案，
自主调取掌握度图谱、沿先修链扫描、核对错误时间先后（下游先错→上游缺口拖累），
最后 file_case 出具查案报告（根因链+修复顺序）落盘才生效。
复用 diagnosis_agents 的真实数据工具。"""
from __future__ import annotations
from ..lc_model import RouterChatModel
from .diagnosis_agents import pull_mastery_map, trace_chain, error_timeline, _run_react
from langchain_core.tools import tool


@tool
def check_error_order(sid: str, cid: str) -> dict:
    """核对时间线：该考点(下游)与其全部先修概念(上游)各自的最近错误时间——
    判断是『下游先错(独立薄弱)』还是『上游也错/上游更晚才错(根因在上游)』。"""
    from .. import storage
    cmap = {c.cid: c for c in storage.load_concepts()}
    cur = cmap.get(cid)
    if not cur:
        return {"error": "考点不存在"}
    answers = storage.load_answers(sid)
    def last_wrong_ts(c):
        ts = [a.ts for a in answers if a.cid == c and not a.correct]
        return ts[-1] if ts else None
    ups = []
    for p in cur.prerequisites:
        ups.append({"cid": p, "name": cmap[p].name if p in cmap else p,
                    "last_wrong": last_wrong_ts(p),
                    "n_wrong": sum(1 for a in answers if a.cid == p and not a.correct)})
    return {"downstream_last_wrong": last_wrong_ts(cid),
            "upstream": ups,
            "hint": "上游从无错误记录=可能从未学过（根因在上游未接触）" if all(not u["last_wrong"] for u in ups) else ""}


@tool
def file_case(sid: str, case_name: str, root_cause: str, chain_desc: str, fix_order: str) -> dict:
    """结案归档。case_name=立案考点名，root_cause=根因结论(一句话)，
    chain_desc=根因链描述(如 A←B←C，标注每环掌握度)，fix_order=建议修复顺序。"""
    import json, os
    from .. import storage
    rec = {"case": case_name, "root_cause": root_cause, "chain": chain_desc,
           "fix_order": fix_order, "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"rootcase_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "case": rec}


TOOLS = [pull_mastery_map, trace_chain, error_timeline, check_error_order, file_case]

SYS = ("你是学区根因追溯科的侦探，专门查『这个考点为什么学不会』的真凶。可调用工具："
       "pull_mastery_map（掌握度图谱）、trace_chain（沿先修链扫描每层掌握度）、"
       "error_timeline（答题时序）、check_error_order（核对上下游错误时间先后）、"
       "file_case（结案归档，唯一生效入口）。"
       "规则：对最薄弱考点立案；必须 trace_chain 看整条先修链；"
       "若上游掌握度更低或上游也有错误 → 真凶在上游（下游只是受害者）；"
       "若上游都健康 → 真凶是该考点本身。最后必须 file_case 归档，"
       "再用≤80字宣布查案结论（谁是真凶、证据是什么、按什么顺序补）。")


def run_detective(sid: str) -> dict:
    """ReAct 查案：返回 {steps, answer}。"""
    model = RouterChatModel(temperature=0.4, max_tokens=500)
    msg = f"学生{sid}的最薄弱考点已立案，请查清真凶并归档。"
    try:
        steps, answer = _run_react(model, TOOLS, SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"侦探失联：{e}"}
    return {"steps": steps, "answer": answer}
