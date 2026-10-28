"""学生个性化学习规划师（LangChain ReAct）。

博士级创新：不止画『画像』，而是真正给学生一份可执行的学习路径——
读画像 → 规划路径 → 生成练习 → 排复习日历，形成『诊断-规划-干预-巩固』闭环。

解决现实教育痛点：
  - 每个学生薄弱点不同，统一作业浪费时间
  - 学生不知道『先学什么』，盲目刷题
  - 没有复习日历，学了就忘（艾宾浩斯）
  - 路径与练习脱节，练的非学的

4 个 @tool：
  1. read_profile  — 读学生深度画像（掌握度分层/薄弱考点/学习风格）
  2. plan_path     — 基于先修依赖+掌握度规划学习路径（拓扑序+薄弱优先）
  3. gen_practice  — 按路径为最薄弱3考点生成针对性练习（调用大模型）
  4. schedule_review — 按艾宾浩斯曲线排14天复习日历

采用过程式编排，4 工具顺序执行，LLM 负责生成规划说明。
"""
from __future__ import annotations
from langchain_core.tools import tool
from .diagnosis_agents import _strip_markdown
from ..lc_model import RouterChatModel


_PLAN_CACHE: dict = {}


@tool
def read_profile(sid: str) -> dict:
    """读学生深度画像：掌握度分层（薄弱/巩固/精通）、薄弱考点清单、答题习惯、学习风格推断。

    返回 {n_concepts, n_weak(<0.4), n_mid, n_strong, weakest5:[{cid,name,mastery}],
         accuracy, avg_dur, style}。style 为学习风格推断（精读型/刷题型/突击型/均衡型）。"""
    from .. import storage
    concepts = {c.cid: c for c in storage.load_concepts()}
    m = storage.load_mastery(sid)
    answers = storage.load_answers(sid)
    if not m:
        return {"n_concepts": len(concepts), "n_weak": 0, "msg": "尚无学习数据"}
    rows = sorted(({"cid": c, "name": concepts[c].name if c in concepts else c,
                    "mastery": round(x.p, 2)} for c, x in m.items()),
                  key=lambda r: r["mastery"])
    n_w = sum(1 for r in rows if r["mastery"] < 0.4)
    n_m = sum(1 for r in rows if 0.4 <= r["mastery"] < 0.7)
    n_s = sum(1 for r in rows if r["mastery"] >= 0.7)
    # 答题习惯
    acc = sum(1 for a in answers if a.correct) / len(answers) if answers else 0
    durs = [a.ts for a in answers]  # 时序粗略
    # 风格推断：正确率高+题量大=刷题型；正确率低=精读型；近期集中=突击型
    style = "均衡型"
    if acc > 0.85 and len(answers) > 20:
        style = "刷题型(高正确率大题量)"
    elif acc < 0.5:
        style = "精读型(正确率待提升)"
    elif len(answers) > 0:
        # 近3天答题占比>60%视为突击
        recent = [a for a in answers if a.ts and a.ts[:10] >= answers[-1].ts[:10]]
        if len(recent) > len(answers) * 0.6:
            style = "突击型(近期集中刷题)"
    return {"n_concepts": len(rows), "n_weak": n_w, "n_mid": n_m, "n_strong": n_s,
            "weakest5": rows[:5], "accuracy": round(acc, 2),
            "n_answers": len(answers), "style": style}


@tool
def plan_path(sid: str) -> dict:
    """基于先修依赖+掌握度规划学习路径：拓扑序中薄弱(<0.4)优先、含前置断点。

    返回 {path:[{step,cid,name,mastery,is_breakpoint,chapter}], n_steps, est_hours}。"""
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    m = storage.load_mastery(sid)
    # 拓扑序（Kahn）
    in_deg = {c.cid: len(c.prerequisites) for c in concepts}
    rev = {c.cid: [] for c in concepts}
    for c in concepts:
        for p in c.prerequisites:
            if p in rev:
                rev[p].append(c.cid)
    queue = [cid for cid, d in in_deg.items() if d == 0]
    order = []
    while queue:
        cid = queue.pop(0)
        order.append(cid)
        for nxt in rev.get(cid, []):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    # 薄弱优先排序：掌握度<0.4 的排前面，但保持拓扑序约束
    ord_idx = {cid: i for i, cid in enumerate(order)}
    def sort_key(cid):
        p = m[cid].p if cid in m else 2.0
        return (0 if p < 0.4 else 1 if p < 0.7 else 2, ord_idx.get(cid, len(order)))
    order.sort(key=sort_key)
    path = []
    for i, cid in enumerate(order[:12]):
        if cid not in cmap:
            continue
        p = round(m[cid].p, 2) if cid in m else None
        is_bp = p is not None and p < 0.4
        # 检查前置是否也薄弱（断点）
        prereqs = [pr for pr in cmap[cid].prerequisites if pr in m and m[pr].p < 0.4]
        path.append({"step": i + 1, "cid": cid, "name": cmap[cid].name,
                      "mastery": p, "is_breakpoint": is_bp,
                      "has_prereq_gap": bool(prereqs),
                      "chapter": cmap[cid].chapter})
    # 估算学时：每考点30分钟，断点+15分钟
    est = sum(0.5 + (0.25 if s["is_breakpoint"] else 0) for s in path)
    return {"path": path, "n_steps": len(path), "est_hours": round(est, 1)}


@tool
def gen_practice(sid: str, path_json: str) -> dict:
    """按学习路径为最薄弱3考点生成针对性练习（调用大模型）。

    path_json 为 plan_path 的输出 JSON。返回 {items:[{cid,question,options,answer,hint}]}。"""
    import json
    try:
        path = json.loads(path_json)
    except Exception:
        return {"error": "path_json 非合法 JSON"}
    from .. import storage
    concepts = {c.cid: c for c in storage.load_concepts()}
    # 取最薄弱3个
    targets = [s for s in path.get("path", []) if s.get("is_breakpoint")][:3]
    if not targets:
        targets = path.get("path", [])[:3]
    items = []
    from ..llm import get_llm
    for t in targets:
        cid = t["cid"]
        c = concepts.get(cid)
        if not c:
            continue
        sys = ("你是出题专家，针对该考点出1道选择题，4选项，附答案和提示。"
               "只输出JSON：{\"question\":\"题干\",\"options\":[\"A\",\"B\",\"C\",\"D\"],"
               "\"answer\":\"正确选项字母\",\"hint\":\"解题提示(≤20字)\"}，不要多余文字。")
        usr = f"考点：{c.name}（{c.summary}），难度：{'薄弱需基础题' if t.get('is_breakpoint') else '巩固题'}。"
        try:
            raw = get_llm().chat(sys, usr, temperature=0.5, max_tokens=300)
            q = json.loads(raw) if isinstance(raw, str) else raw
            q["cid"] = cid
            q["concept"] = c.name
            items.append(q)
        except Exception as e:
            items.append({"cid": cid, "concept": c.name,
                          "question": f"{c.name}基础题（生成失败:{e}）",
                          "options": ["A", "B", "C", "D"], "answer": "A", "hint": c.summary[:20]})
    return {"items": items, "n": len(items)}


@tool
def schedule_review(sid: str, path_json: str) -> dict:
    """按艾宾浩斯遗忘曲线排14天复习日历。

    复习节点：第1/2/4/7/14天。返回 {calendar:[{day,items:[考点名],goal}]}。"""
    import json
    try:
        path = json.loads(path_json)
    except Exception:
        return {"error": "path_json 非合法 JSON"}
    # 艾宾浩斯节点
    nodes = [1, 2, 4, 7, 14]
    cids = [s["cid"] for s in path.get("path", [])[:8]]
    if not cids:
        return {"calendar": [], "msg": "路径为空"}
    from .. import storage
    concepts = {c.cid: c for c in storage.load_concepts()}
    calendar = []
    # 每个复习日轮转分配考点
    for d in nodes:
        idx = (d - 1) % len(cids)
        batch = cids[idx:idx + 3]
        names = [concepts[c].name if c in concepts else c for c in batch]
        calendar.append({"day": d, "items": names,
                         "goal": "首次学习+即时复习" if d == 1
                         else f"第{d}天巩固复习(防遗忘)"})
    return {"calendar": calendar, "n_nodes": len(nodes)}


PLANNER_SYS = (
    "你是学生个性化学习规划师，为每位学生生成可执行的学习路径+练习+复习日历。"
    "可调用工具：read_profile（读深度画像）、plan_path（规划学习路径）、"
    "gen_practice（生成针对性练习）、schedule_review（排复习日历）。"
    "流程：必须 read_profile 读画像 → plan_path 规划路径 → "
    "gen_practice 生成练习 → schedule_review 排复习日历。"
    "原则：薄弱(<0.4)优先、含前置断点检测、艾宾浩斯防遗忘。"
    "4 工具调用后，用≤150字向学生说明：你的薄弱在哪、先学什么、"
    "为什么这个顺序、复习日历怎么用。语气鼓励。"
)

PLANNER_TOOLS = [read_profile, plan_path, gen_practice, schedule_review]


def run_learning_planner(sid: str) -> dict:
    """个性化学习规划师（过程式编排，绕过 _run_react 截断）。

    返回 {steps, answer, profile, path, practice, schedule}。"""
    import json
    model = RouterChatModel(temperature=0.5, max_tokens=500)
    steps: list[dict] = []

    # 1. 读画像
    profile = read_profile.invoke({"sid": sid})
    steps.append({"tool": "read_profile", "args": {"sid": sid},
                  "result": json.dumps(profile, ensure_ascii=False)[:400]})

    # 2. 规划路径
    path = plan_path.invoke({"sid": sid})
    steps.append({"tool": "plan_path", "args": {"sid": sid},
                  "result": json.dumps(path, ensure_ascii=False)[:400]})

    # 3. 生成练习
    path_str = json.dumps(path, ensure_ascii=False)
    practice = gen_practice.invoke({"sid": sid, "path_json": path_str})
    steps.append({"tool": "gen_practice", "args": {"sid": sid, "path_json": "..."},
                  "result": json.dumps(practice, ensure_ascii=False)[:400]})

    # 4. 排复习日历
    schedule = schedule_review.invoke({"sid": sid, "path_json": path_str})
    steps.append({"tool": "schedule_review", "args": {"sid": sid, "path_json": "..."},
                  "result": json.dumps(schedule, ensure_ascii=False)[:400]})

    # 5. LLM 生成规划说明
    facts = (f"画像：{profile['n_concepts']}考点，薄弱{profile['n_weak']}个，"
             f"风格『{profile.get('style','未知')}』；"
             f"路径：{path['n_steps']}步，预估{path['est_hours']}小时，"
             f"断点{sum(1 for s in path['path'] if s.get('is_breakpoint'))}个；"
             f"练习：{practice.get('n',0)}题；复习日历{schedule.get('n_nodes',0)}节点。")
    sys_msg = (f"{PLANNER_SYS}\n证据采集已结束，禁止再调工具，"
               "直接输出最终学习规划说明（纯文本短段落，语气鼓励）。")
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        final = model.invoke([SystemMessage(content=sys_msg),
                              HumanMessage(content=facts)])
        answer = _strip_markdown(str(final.content))
    except Exception as e:
        answer = f"规划生成失败，使用确定性结论：{facts}"

    _PLAN_CACHE.update({"profile": profile, "path": path,
                        "practice": practice, "schedule": schedule})
    return {"steps": steps, "answer": answer,
            "profile": profile, "path": path,
            "practice": practice, "schedule": schedule}
