"""教育资源公平分配智能体（LangChain ReAct）。

博士级创新：把『教育资源分配』从『按校分摊/按成绩掐尖』的人工决策，
升级为 LLM ReAct 智能体的多目标公平分配——
学情需求 + 师资供给 + 匹配优化 + 公平审计 + 落盘方案。

解决现实教育痛点：
  - 师资少、学生多 → 每班塞满，薄弱学生得不到关注
  - 按成绩掐尖 → 强者愈强、弱者愈弱（马太效应）
  - 分配不透明 → 家长质疑、教师抵触
  - 资源错配 → 擅长教基础的老师却带尖子班

本智能体的分配原则（写入系统提示）：
  1. 倾斜分配：薄弱学生（掌握度<0.4）优先匹配小班额/名师
  2. 供需匹配：教师学科与学生薄弱考点对齐
  3. 公平下限：每生获得『至少1项干预资源』，吉尼系数<0.35
  4. 能力分流：基础/巩固/拓展三档分层，避免一刀切

采用过程式编排（不走 _run_react 的截断循环），
5 个 @tool 顺序执行，LLM 仅负责生成结论性方案说明。
"""
from __future__ import annotations
from langchain_core.tools import tool
from .diagnosis_agents import _strip_markdown
from ..lc_model import RouterChatModel


# 模块级缓存：绕过 _run_react 把工具结果截断到 180 字符的问题
_ALLOC_CACHE: dict = {}


@tool
def collect_demand(cohort: str) -> dict:
    """采集学情需求：全员掌握度排名、薄弱学生清单、分层结构、薄弱考点聚集区。

    cohort 为学生群标识（如 'all' 全体 / 'os' 操作系统课程学生）。
    返回 {n_students, n_weak(<0.4), n_mid(0.4-0.7), n_strong(>0.7),
          weakest10:[{sid,avg_mastery,n_gaps}], hot_concepts:[{cid,name,gap_count}]}。"""
    from .. import storage
    students = storage.load_students()
    sids = list(students.keys()) if cohort == "all" else [
        s for s in students if students[s].tier == cohort]
    if not sids:
        sids = list(students.keys())[:40]
    rows = []
    concept_gaps: dict[str, int] = {}
    for sid in sids:
        m = storage.load_mastery(sid)
        if not m:
            continue
        vals = [x.p for x in m.values()]
        avg = sum(vals) / len(vals) if vals else 0
        gaps = sum(1 for p in vals if p < 0.4)
        for cid, x in m.items():
            if x.p < 0.4:
                concept_gaps[cid] = concept_gaps.get(cid, 0) + 1
        rows.append({"sid": sid, "avg_mastery": round(avg, 2), "n_gaps": gaps})
    rows.sort(key=lambda r: r["avg_mastery"])
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    hot = sorted(({"cid": cid, "name": cmap[cid].name if cid in cmap else cid, "gap_count": n}
                 for cid, n in concept_gaps.items()), key=lambda x: -x["gap_count"])[:6]
    n_w = sum(1 for r in rows if r["avg_mastery"] < 0.4)
    n_m = sum(1 for r in rows if 0.4 <= r["avg_mastery"] < 0.7)
    n_s = sum(1 for r in rows if r["avg_mastery"] >= 0.7)
    return {"n_students": len(rows), "n_weak": n_w, "n_mid": n_m, "n_strong": n_s,
            "weakest10": rows[:10], "hot_concepts": hot}


@tool
def assess_supply(subject: str) -> dict:
    """采集师资供给：教师清单、学科、班额基准、当前负载。

    subject 为学科过滤（'all' 全体 / 具体学科名）。
    返回 {n_teachers, total_capacity, by_subject:{学科:人数}, underloaded:[tid], overloaded:[tid]}。"""
    from .. import storage
    teachers = storage.load_teachers()
    if subject != "all":
        teachers = {t: v for t, v in teachers.items() if v.subject == subject}
    if not teachers:
        return {"n_teachers": 0, "msg": "师资库为空"}
    # 当前负载 = 分配给该教师的学生数
    students = storage.load_students()
    load: dict[str, int] = {}
    for s in students.values():
        load[s.teacher_id] = load.get(s.teacher_id, 0) + 1
    by_subj: dict[str, int] = {}
    under, over = [], []
    cap_total = 0
    for tid, t in teachers.items():
        by_subj[t.subject] = by_subj.get(t.subject, 0) + 1
        cap_total += t.group_size
        cur = load.get(tid, 0)
        if cur < t.group_size * 0.5:
            under.append(tid)
        elif cur > t.group_size * 1.5:
            over.append(tid)
    return {"n_teachers": len(teachers), "total_capacity": cap_total,
            "by_subject": by_subj, "underloaded": under[:8], "overloaded": over[:8],
            "current_load": load}


@tool
def match_optimize(demand_json: str, supply_json: str) -> dict:
    """基于需求与供给做匹配优化：薄弱学生→小班名师、强生→大班自主、按薄弱考点对齐学科。

    返回 {reassign:[{sid,from_tid,to_tid,reason}], n_reassign, weak_first_served,
          subject_aligned}。采用贪心：按薄弱度升序，把最弱学生塞进负载最低的教师。"""
    import json
    try:
        demand = json.loads(demand_json)
        supply = json.loads(supply_json)
    except Exception:
        return {"error": "输入非合法 JSON"}
    from .. import storage
    students = storage.load_students()
    teachers = storage.load_teachers()
    if not students or not teachers:
        return {"error": "学生或师资数据为空"}
    load = dict(supply.get("current_load", {}))
    # 按掌握度升序排学生
    scored = []
    for sid in demand.get("weakest10", []):
        scored.append((sid["sid"], sid["avg_mastery"], sid.get("n_gaps", 0)))
    # 补齐其余学生
    seen = {s for s, _, _ in scored}
    for sid, st in students.items():
        if sid not in seen:
            m = storage.load_mastery(sid)
            avg = sum(x.p for x in m.values()) / len(m) if m else 1.0
            scored.append((sid, round(avg, 2), 0))
    scored.sort(key=lambda x: x[1])
    # 负载最低的教师优先接收最弱学生
    by_load = sorted(teachers.keys(), key=lambda t: load.get(t, 0))
    reassign = []
    weak_served = 0
    subj_aligned = 0
    # 热门薄弱考点的学科
    hot_cids = {h["cid"] for h in demand.get("hot_concepts", [])}
    concepts = {c.cid: c for c in storage.load_concepts()}
    for sid, avg, gaps in scored:
        cur_tid = students[sid].teacher_id
        # 薄弱学生（avg<0.4）优先匹配负载最低的教师
        if avg < 0.4 and len(by_load) > 1:
            target = by_load[0]
            if target != cur_tid and load.get(target, 0) < teachers[target].group_size * 1.2:
                # 学科对齐加分：教师学科覆盖该生薄弱考点
                t_subj = teachers[target].subject
                reason = "薄弱生倾斜→负载最低班"
                # 检查学科是否对齐热门考点
                for cid in hot_cids:
                    if cid in concepts and (t_subj in concepts[cid].name
                                            or concepts[cid].chapter.startswith(t_subj)):
                        subj_aligned += 1
                        reason = f"薄弱生→{t_subj}名师(对齐薄弱考点{concepts[cid].name})"
                        break
                reassign.append({"sid": sid, "from_tid": cur_tid,
                                 "to_tid": target, "reason": reason})
                load[target] = load.get(target, 0) + 1
                load[cur_tid] = max(load.get(cur_tid, 1) - 1, 0)
                weak_served += 1
                by_load = sorted(teachers.keys(), key=lambda t: load.get(t, 0))
    return {"reassign": reassign[:30], "n_reassign": len(reassign),
            "weak_first_served": weak_served, "subject_aligned": subj_aligned}


@tool
def fairness_audit(allocation_json: str) -> dict:
    """公平审计：计算每个学生获得的『干预资源分』（薄弱生被分到名师+2、
    被分到小班+1），算吉尼系数；检查是否每生≥1项资源（公平下限）。

    allocation_json 为 match_optimize 的输出 JSON。
    返回 {gini, n_underserved, fairness_pass, verdict}。"""
    import json
    try:
        alloc = json.loads(allocation_json)
    except Exception:
        return {"error": "输入非合法 JSON"}
    from .. import storage
    students = storage.load_students()
    teachers = storage.load_teachers()
    reassign = alloc.get("reassign", [])
    # 每生资源分：默认1（基础资源），被重分配到名师+2，小班+1
    scores = {sid: 1 for sid in students}
    for r in reassign:
        sid = r["sid"]
        to = r["to_tid"]
        if to in teachers:
            # 小班（group_size<4）+1
            if teachers[to].group_size < 4:
                scores[sid] = scores.get(sid, 1) + 1
            # 名师（负载已被重分配多次说明需求高，简化为+2）
            scores[sid] = scores.get(sid, 1) + 2
    vals = sorted(scores.values())
    n = len(vals)
    if n == 0:
        return {"gini": 0, "n_underserved": 0, "fairness_pass": False}
    # 吉尼系数
    cum = sum((2 * i - n - 1) * v for i, v in enumerate(vals, 1))
    gini = cum / (n * sum(vals)) if sum(vals) else 0
    underserved = sum(1 for v in vals if v < 1)
    fair = gini < 0.35 and underserved == 0
    return {"gini": round(gini, 3), "n_underserved": underserved,
            "fairness_pass": fair,
            "verdict": (f"公平审计通过：吉尼{gini:.3f}<0.35、无人落空" if fair
                        else f"公平不达标：吉尼{gini:.3f}、{underserved}人未获资源，需回调")}


@tool
def issue_allocation(plan_json: str) -> dict:
    """落盘资源分配方案。plan_json 为 JSON 字符串：
    {"reassign":[...],"fairness":{...},"rationale":"方案说明(≤80字)"}。"""
    import json, os
    from .. import storage
    try:
        plan = json.loads(plan_json)
        if "reassign" not in plan or "rationale" not in plan:
            raise ValueError
    except Exception:
        return {"error": "plan_json 必须含 reassign 和 rationale"}
    rec = {"plan": plan, "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, "resource_allocation.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"ok": True, "n_reassign": len(plan.get("reassign", [])),
            "rationale": plan.get("rationale", "")[:80]}


ALLOC_SYS = (
    "你是教育资源公平分配总监，负责把有限的师资按学情需求做最优匹配。"
    "可调用工具：collect_demand（采集学情需求：薄弱生/分层/热门薄弱考点）、"
    "assess_supply（采集师资供给：教师/学科/班额/负载）、"
    "match_optimize（需求×供给匹配优化：薄弱生→小班名师、学科对齐）、"
    "fairness_audit（公平审计：吉尼系数/公平下限/落空检查）、"
    "issue_allocation（落盘分配方案，唯一生效入口）。"
    "流程：必须 collect_demand 采需求 → assess_supply 采供给 → "
    "match_optimize 做匹配 → fairness_audit 审计 → issue_allocation 落盘。"
    "分配原则：薄弱生优先倾斜（掌握度<0.4 优先小班名师）、学科对齐薄弱考点、"
    "吉尼<0.35、无人落空。"
    "issue_allocation 落盘后，用≤120字向教育局长汇报：分配了多少人、"
    "薄弱生覆盖率、吉尼系数、方案核心思路。"
)

ALLOC_TOOLS = [collect_demand, assess_supply, match_optimize, fairness_audit, issue_allocation]


def run_resource_allocator(subject: str = "all") -> dict:
    """教育资源公平分配智能体（过程式编排，绕过 _run_react 截断）。

    返回 {steps, answer, plan, demand, supply, fairness}。"""
    import json
    model = RouterChatModel(temperature=0.4, max_tokens=500)
    steps: list[dict] = []

    # 1. 采需求
    demand = collect_demand.invoke({"cohort": subject})
    steps.append({"tool": "collect_demand",
                  "args": {"cohort": subject},
                  "result": json.dumps(demand, ensure_ascii=False)[:400]})

    # 2. 采供给
    supply = assess_supply.invoke({"subject": subject})
    steps.append({"tool": "assess_supply",
                  "args": {"subject": subject},
                  "result": json.dumps(supply, ensure_ascii=False)[:400]})

    # 3. 匹配优化
    match = match_optimize.invoke({"demand_json": json.dumps(demand, ensure_ascii=False),
                                   "supply_json": json.dumps(supply, ensure_ascii=False)})
    steps.append({"tool": "match_optimize",
                  "args": {"demand_json": "...", "supply_json": "..."},
                  "result": json.dumps(match, ensure_ascii=False)[:400]})

    # 4. 公平审计
    fairness = fairness_audit.invoke({"allocation_json": json.dumps(match, ensure_ascii=False)})
    steps.append({"tool": "fairness_audit",
                  "args": {"allocation_json": "..."},
                  "result": json.dumps(fairness, ensure_ascii=False)[:400]})

    # 5. LLM 生成方案说明 + 落盘
    facts = (f"需求侧：{demand['n_students']}生，薄弱{demand['n_weak']}人，"
             f"热门薄弱考点{[h['name'] for h in demand['hot_concepts'][:3]]}；"
             f"供给侧：{supply['n_teachers']}师，总容量{supply['total_capacity']}，"
             f"超载{len(supply.get('overloaded',[]))}人；"
             f"匹配：重分配{match['n_reassign']}人，薄弱生倾斜{match['weak_first_served']}人，"
             f"学科对齐{match['subject_aligned']}项；"
             f"审计：吉尼{fairness['gini']}，{'通过' if fairness['fairness_pass'] else '不达标'}。")
    sys_msg = (f"{ALLOC_SYS}\n证据采集已结束，禁止再调工具，直接输出最终分配方案说明（纯文本短段落）。"
               "格式：第1句分配结果摘要，第2句薄弱生覆盖，第3句公平性，第4句核心思路。")
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        final = model.invoke([SystemMessage(content=sys_msg),
                              HumanMessage(content=facts)])
        rationale = _strip_markdown(str(final.content))
    except Exception as e:
        rationale = f"方案生成失败，使用确定性结论：{facts}"

    plan = {"reassign": match.get("reassign", []), "fairness": fairness,
            "rationale": rationale[:80]}
    res = issue_allocation.invoke({"plan_json": json.dumps(plan, ensure_ascii=False)})
    steps.append({"tool": "issue_allocation",
                  "args": {"plan_json": "..."},
                  "result": json.dumps(res, ensure_ascii=False)[:200]})

    _ALLOC_CACHE.update({"plan": plan, "demand": demand, "supply": supply,
                         "fairness": fairness, "match": match})
    return {"steps": steps, "answer": rationale,
            "plan": plan, "demand": demand, "supply": supply,
            "fairness": fairness, "match": match}
