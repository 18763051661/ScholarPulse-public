"""记忆保卫战·防御指挥官（LangChain ReAct 智能体）。
把间隔复习从"列表打卡"升级为对抗遗忘的阵地战：
每个到期考点是一个正在失守的阵地，学生答题即收复。
指挥官是一个真正的 LangChain ReAct 智能体（create_react_agent），
持有三个标准 @tool（扫描阵地/选定目标/判定收复），复用自研
RouterChatModel 的 bind_tools JSON 协议——不依赖厂商 tool-calling API。
"""
from __future__ import annotations
from datetime import datetime, timedelta
from langchain_core.tools import tool
from ..lc_model import RouterChatModel

INTERVALS = [1, 2, 4, 7, 15]


@tool
def scan_front(sid: str) -> dict:
    """扫描记忆阵地：返回今天到期（艾宾浩斯间隔超期）的考点及紧急度（超期天数）。"""
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(sid)
    fronts = []
    for cid, m in mastery.items():
        stage = min(m.n_correct, len(INTERVALS) - 1)
        gap = INTERVALS[stage]
        try:
            last = datetime.strptime(m.updated_at, "%Y-%m-%d %H:%M:%S")
            overdue = max(0, (datetime.now() - last - timedelta(days=gap)).days)
        except Exception:
            overdue = 9
        if overdue > 0:
            fronts.append({"cid": cid, "name": name_of.get(cid, cid),
                           "mastery": round(m.p, 2), "overdue_days": overdue})
    fronts.sort(key=lambda x: (-x["overdue_days"], x["mastery"]))
    return {"fronts": fronts[:12], "n_lost": len(fronts)}


@tool
def pick_target(cid: str) -> dict:
    """为指定考点阵地从题库抽一道题（优先薄弱考点的高权重题）。"""
    from .. import storage
    for q in storage.load_quiz_bank():
        if q.cid == cid:
            return {"qid": q.qid, "cid": cid, "question": q.question,
                    "tier": q.tier, "reference": q.reference_answer}
    return {"cid": cid, "question": "", "msg": "题库中暂无该考点题目"}


@tool
def judge_answer(sid: str, qid: str, answer: str) -> dict:
    """批改学生对阵地的作答：LLM 判卷（要点覆盖≥60% 算收复成功），
    更新掌握度/Elo 并写入答题记录。返回收复结果。"""
    from .. import storage, mastery as M
    from ..llm import get_llm
    from ..models import AnswerRecord
    q = next((x for x in storage.load_quiz_bank() if x.qid == qid), None)
    if not q:
        return {"error": "题目不存在"}
    try:
        g = get_llm().chat_json(
            '你是阅卷老师，比对作答与参考要点给 0 到 1 的分（覆盖≥60% 给 1）。只输出 JSON：{"score":<数值>,"comment":"≤15字"}',
            f"题目:{q.question}\n参考:{q.reference_answer}\n作答:{answer}", temperature=0.2, max_tokens=120)
        score = float(g.get("score", 0))
        comment = str(g.get("comment", ""))
    except Exception:
        score, comment = 0.0, "批改降级"
    correct = score >= 0.6
    mastery = storage.load_mastery(sid)
    if q.cid in mastery:
        M.apply_answer(mastery[q.cid], correct)
        storage.save_mastery(sid, mastery)
    base = {"foundation": 1000, "consolidation": 1400, "extension": 1800}.get(q.tier, 1200)
    elo_d = storage.load_quiz_elo(sid)
    expect = 1.0 / (1.0 + 10 ** ((base - elo_d["elo"]) / 400.0))
    elo_d["elo"] = max(800, elo_d["elo"] + round(32 * ((1.0 if correct else 0.0) - expect)))
    elo_d["n"] += 1
    storage.save_quiz_elo(sid, elo_d)
    storage.append_answers(sid, [AnswerRecord(qid=q.qid, cid=q.cid, correct=correct,
                                              ts=storage.now_iso(), question=q.question,
                                              reference=q.reference_answer)])
    return {"correct": correct, "score": score, "comment": comment,
            "reclaimed": correct, "mastery": round(mastery[q.cid].p, 2)}


TOOLS = [scan_front, pick_target, judge_answer]

SYS = ("你是学区记忆防御指挥官，指挥学生对抗遗忘的阵地战。可调用工具："
       "scan_front（扫描失守阵地）、pick_target（为阵地抽题）、judge_answer（判定收复）。"
       "用户要战术简报时：先调用 scan_front，再按紧急度输出简报（≤120字："
       "失守阵地数、最危急的 1-2 个阵地及原因、今天建议先收复哪个）。军事口吻但亲切。")


def run_defender(sid: str, message: str = "指挥官，给我今天的战术简报") -> dict:
    """ReAct 主循环：返回 {steps, answer}（工具调用轨迹 + 最终简报）。"""
    from langgraph.prebuilt import create_react_agent
    model = RouterChatModel(temperature=0.5, max_tokens=400)
    try:
        agent = create_react_agent(model, TOOLS, prompt=SYS)
    except TypeError:
        agent = create_react_agent(model, TOOLS, state_modifier=SYS)
    out = agent.invoke({"messages": [("user", f"学生{sid}：{message}")]})
    steps, answer = [], ""
    for m in out["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                steps.append({"tool": tc["name"], "args": tc["args"]})
        elif type(m).__name__ == "ToolMessage":
            steps[-1]["result"] = str(m.content)[:200] if steps else str(m.content)[:200]
        elif type(m).__name__ == "AIMessage" and m.content:
            answer = str(m.content)
    return {"steps": steps, "answer": answer}
