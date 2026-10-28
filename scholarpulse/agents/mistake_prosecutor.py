"""错题检察院·检察官（LangChain ReAct 智能体）。
错题本从"记录本"升级为司法系统：每个顽固错题簇是一桩未决案件，
检察官自主取证（调卷宗/查先修/查康复）后当庭宣判，四类判决——
真性薄弱 / 假性康复（背答案嫌疑）/ 已自愈结案 / 上游传染。
与病理科（固定流水线）不同：检察官是真正的 create_react_agent，
自己决定查什么证据、下什么判决，判决必须落到 submit_verdict 才生效。
"""
from __future__ import annotations
from langchain_core.tools import tool
from ..lc_model import RouterChatModel

VERDICTS = {"guilty_persistent": "真性薄弱·判处重修",
            "fake_recovery": "假性康复·撤销并留观",
            "healed": "已自愈·当庭结案",
            "upstream_infection": "上游传染·移送先修概念"}


@tool
def pull_case_file(sid: str, cid: str) -> dict:
    """调阅案件卷宗：该考点全部答题记录时序（对错序列与时间跨度）、当前掌握度、最近错题原文。"""
    from datetime import date
    from .. import storage
    answers = [a for a in storage.load_answers(sid) if a.cid == cid]
    if not answers:
        return {"cid": cid, "msg": "无答题记录"}
    marks = "".join("✔" if a.correct else "✘" for a in answers)
    mastery = storage.load_mastery(sid).get(cid)
    last_wrong = next((a for a in reversed(answers) if not a.correct), None)
    span = 0
    if len(answers) > 1:
        try:
            span = max(0, (date.fromisoformat(answers[-1].ts[:10]) -
                           date.fromisoformat(answers[0].ts[:10])).days)
        except Exception:
            span = 0
    return {"cid": cid, "n": len(answers), "timeline": marks, "span_days": span,
            "mastery": round(mastery.p, 2) if mastery else None,
            "last_wrong_q": (last_wrong.question or "")[:60] if last_wrong else "",
            "last_wrong_ref": (last_wrong.reference or "")[:60] if last_wrong else ""}


@tool
def check_upstream(sid: str, cid: str) -> dict:
    """查先修概念的掌握度：判断薄弱根因是否在上游概念（上游传染）。"""
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    cur = next((c for c in concepts if c.cid == cid), None)
    if not cur or not cur.prerequisites:
        return {"upstream": [], "msg": "该考点无先修概念，根因只能在本概念"}
    mastery = storage.load_mastery(sid)
    return {"upstream": [{"cid": p, "name": name_of.get(p, p),
                          "mastery": round(mastery[p].p, 2) if p in mastery else None}
                         for p in cur.prerequisites]}


@tool
def check_recovery(sid: str, cid: str) -> dict:
    """查康复证据：最近一次错误之后连续答对几次、距今天数。连对少且间隔短=背答案嫌疑。"""
    from .. import storage
    from datetime import datetime
    answers = [a for a in storage.load_answers(sid) if a.cid == cid]
    if not any(not a.correct for a in answers):
        return {"after_last_wrong": 0, "msg": "从未答错，无需起诉"}
    idx = max(i for i, a in enumerate(answers) if not a.correct)
    tail = answers[idx + 1:]
    streak = 0
    for a in tail:
        if a.correct:
            streak += 1
        else:
            break
    days = 0
    if tail:
        try:
            d0 = datetime.strptime(answers[idx].ts, "%Y-%m-%d %H:%M:%S")
            d1 = datetime.strptime(tail[-1].ts, "%Y-%m-%d %H:%M:%S")
            days = max(0, (d1 - d0).days)
        except Exception:
            days = 0
    return {"after_last_wrong": len(tail), "right_streak": streak,
            "span_days": days,
            "hint": ("错后立即连对且间隔极短——背答案嫌疑重大" if streak >= 2 and days <= 1
                     else "连对且间隔充分——有康复迹象" if streak >= 2
                     else "错后无充分连对——康复证据不足")}


@tool
def submit_verdict(sid: str, cid: str, charge: str, verdict: str, prescription: str) -> dict:
    """提交当庭判决并落盘。charge=罪名(概念混淆/技能缺失/审题习惯/记忆模糊/根因缺失)，
    verdict 四选一：guilty_persistent / fake_recovery / healed / upstream_infection，
    prescription=量刑处方(一句话)。判决必须经此工具才生效。"""
    from .. import storage
    if verdict not in VERDICTS:
        return {"error": "verdict 非法", "allowed": list(VERDICTS)}
    concepts = storage.load_concepts()
    name = next((c.name for c in concepts if c.cid == cid), cid)
    book = [v for v in storage.load_prosecution(sid) if v["cid"] != cid]
    book.append({"cid": cid, "concept": name, "charge": charge, "verdict": verdict,
                 "prescription": prescription, "ts": storage.now_iso(), "closed": verdict == "healed"})
    storage.save_prosecution(sid, book)
    return {"ok": True, "verdict_label": VERDICTS[verdict]}


TOOLS = [pull_case_file, check_upstream, check_recovery, submit_verdict]

SYS = ("你是学区错题检察院的检察官，对学生的顽固错题提起公诉。可调用工具："
       "pull_case_file（调卷宗看答题时序）、check_upstream（查先修概念掌握度）、"
       "check_recovery（查错后康复证据）、submit_verdict（当庭宣判，唯一生效入口）。"
       "审判规则：①先修概念掌握度也低且错题在该概念之前就错 → upstream_infection；"
       "②错后短时间内连续答对≥2 → 警惕 fake_recovery（背答案）；"
       "③错后再无作答或连对不足 → guilty_persistent；"
       "④错后间隔充分且连对 → healed。"
       "流程：必须先 pull_case_file，按需查上游/康复，最后必须 submit_verdict 宣判，"
       "再用≤60字给出结案陈词（判决理由+量刑说明），口吻庄严但关切学生。")


def run_prosecutor(sid: str, cases: list[dict]) -> list[dict]:
    """对每桩案件独立 ReAct 侦查：返回 [{cid, concept, steps, answer}]。"""
    from langgraph.prebuilt import create_react_agent
    model = RouterChatModel(temperature=0.4, max_tokens=500)
    try:
        agent = create_react_agent(model, TOOLS, prompt=SYS)
    except TypeError:
        agent = create_react_agent(model, TOOLS, state_modifier=SYS)
    out = []
    for c in cases:
        msg = (f"起诉案件：被告考点[{c['cid']}]{c['concept']}，累计答错{c['wrong']}/{c['total']}次，"
               f"最近错题：{c['last_q'][:50]}。学生{sid}。请侦查并宣判。")
        try:
            res = agent.invoke({"messages": [("user", msg)]})
        except Exception as e:
            out.append({"cid": c["cid"], "concept": c["concept"],
                        "steps": [], "answer": f"检察官失联：{e}"})
            continue
        steps, answer = [], ""
        for m in res["messages"]:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    steps.append({"tool": tc["name"], "args": {k: v for k, v in tc["args"].items() if k != "sid"}})
            elif type(m).__name__ == "ToolMessage":
                if steps:
                    steps[-1]["result"] = str(m.content)[:160]
            elif type(m).__name__ == "AIMessage" and m.content:
                answer = str(m.content)
        out.append({"cid": c["cid"], "concept": c["concept"], "steps": steps, "answer": answer})
    return out
