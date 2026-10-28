"""🔮 考点押题·先知（LangChain ReAct 智能体）。
把"考点押题"从静态预测升级为真 ReAct：先知自主采集三路证据——
班级共性薄弱（350 人合成人口的同班口径）、个人错题本、考试权重榜，
交叉研判后 lock_predictions 落盘押题清单（每题附概率与依据）才生效。
复用 diagnosis_agents 的工具与通用 ReAct 循环。"""
from __future__ import annotations
from ..lc_model import RouterChatModel
from .diagnosis_agents import pull_mastery_map, error_timeline, due_review_count, _run_react
from langchain_core.tools import tool


@tool
def scan_class_hotspots(sid: str) -> dict:
    """扫描同班共性薄弱：与该生同教师的学生中，每考点平均掌握度排名（最弱 Top5）。"""
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    students = storage.load_students()
    me = students.get(sid)
    pool = {k: v for k, v in students.items() if me and v.teacher_id == me.teacher_id}
    if not pool:
        pool = dict(students)
    acc: dict[str, list[float]] = {}
    for k in pool:
        m = storage.load_mastery(k)
        for c in concepts:
            acc.setdefault(c.cid, []).append(m[c.cid].p if c.cid in m else 1.0)
    rows = sorted(((c.cid, c.name, sum(acc[c.cid]) / len(acc[c.cid])) for c in concepts),
                  key=lambda x: x[2])
    return {"n_classmates": len(pool),
            "hotspots": [{"cid": r[0], "name": r[1], "class_avg": round(r[2], 2)} for r in rows[:5]]}


@tool
def exam_weight_rank() -> dict:
    """查考试权重榜：全部考点按 exam_weight 从高到低排名（越高越可能考大题）。"""
    from .. import storage
    cs = sorted(storage.load_concepts(), key=lambda c: -c.exam_weight)
    return {"top": [{"cid": c.cid, "name": c.name, "weight": round(c.exam_weight, 2)} for c in cs[:10]]}


@tool
def wrong_book_brief(sid: str) -> dict:
    """查个人错题本简报：各考点错误次数排名（Top5）。"""
    from .. import storage
    name_of = {c.cid: c.name for c in storage.load_concepts()}
    acc: dict[str, int] = {}
    for a in storage.load_answers(sid):
        if not a.correct:
            acc[a.cid] = acc.get(a.cid, 0) + 1
    rows = sorted(acc.items(), key=lambda x: -x[1])[:5]
    return {"wrong_rank": [{"cid": k, "name": name_of.get(k, k), "n_wrong": v} for k, v in rows]}


@tool
def lock_predictions(sid: str, predictions: str) -> dict:
    """落盘押题清单。predictions 为 JSON 数组字符串，每项：
    {"cid":"考点id","name":"考点名","prob":0-100概率,"reason":"押题依据(≤25字)"}。
    必须 5 条，按概率从高到低。"""
    import json, os
    from .. import storage
    try:
        arr = json.loads(predictions)
        if not isinstance(arr, list) or len(arr) < 3:
            raise ValueError
    except Exception:
        return {"error": "predictions 必须是至少 3 条的 JSON 数组"}
    rec = {"predictions": arr[:5], "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"oracle_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "n": len(rec["predictions"])}


TOOLS = [scan_class_hotspots, exam_weight_rank, wrong_book_brief, pull_mastery_map, lock_predictions]

SYS = ("你是学区考试中心的押题先知。可调用工具：scan_class_hotspots（同班共性薄弱）、"
       "exam_weight_rank（考试权重榜）、wrong_book_brief（个人错题本）、"
       "pull_mastery_map（个人掌握度）、lock_predictions（落盘押题清单，唯一生效入口）。"
       "研判规则：押题概率 = 考试权重 × 班级共性弱 × 个人弱 三路证据交叉；"
       "权重高但全班都强 → 降低概率；个人错 3 次以上且权重高 → 概率拉满。"
       "必须先采集至少 3 路证据，然后必须 lock_predictions 落盘 5 条押题，"
       "最后用≤90字宣布押题结论（最稳的 1-2 押和理由，先知口吻）。")


def run_oracle(sid: str) -> dict:
    """ReAct 押题：返回 {steps, answer}。"""
    model = RouterChatModel(temperature=0.4, max_tokens=600)
    msg = f"请为学生{sid}研判下次考试最可能考的 5 个考点，落盘押题清单。"
    try:
        steps, answer = _run_react(model, TOOLS, SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"先知闭关中：{e}"}
    return {"steps": steps, "answer": answer}
