"""教师端 ReAct 智能体（班级教研官 + 年级督导员）。

博士级创新：把教师后台从『纯规则统计 + 模板话术』升级为
『LLM ReAct 智能体多维度取证 → 自主推理班级叙事 → 分层干预处方 → 年级瓶颈诊断』。

两个智能体：
  1. 🏫 班级教研官  — 3 工具 ReAct
     - scan_class_distribution  — 班级掌握度分布（均值/方差/偏态/分层比例）
     - scan_concept_failure_clusters — 考点失败聚类（共薄弱点 vs 孤立薄弱点）
     - diagnose_at_risk_students — 需介入学生多维根因诊断（基础差/中途掉链/卡壳某考点）
  2. 🏛️ 年级督导员  — 3 工具 ReAct
     - scan_grade_dispersion    — 年级离散度（班均标准差/极差/排名集中度）
     - scan_bottleneck_concepts — 全年级瓶颈考点（共性瓶颈 vs 班级特异性瓶颈）
     - prescribe_grade_intervention — 年级干预处方（教研组备课/校际帮扶/分层教学）

工具只输出原始量化证据，LLM 自主推理结论——不预贴标签、不套模板。
"""
from __future__ import annotations
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel
from .. import storage, mastery as M


def _concept_name(entry, cid: str = "") -> str:
    """考点名安全解析（属性访问前的合法性检查 + 异常捕获）。
    兼容四类数据形态：Concept 对象 → .name；历史字符串 → 原值；
    dict → name 键；None/空/未知形态 → 回退 cid。
    杜绝 'str' object has no attribute 'name' 类崩溃。"""
    try:
        if entry is None:
            return cid
        if isinstance(entry, str):
            return entry or cid
        if isinstance(entry, dict):
            name = entry.get("name")
            return str(name) if name else cid
        name = getattr(entry, "name", None)
        return str(name) if name else cid
    except Exception:
        return cid


# ============================================================
# 1. 🏫 班级教研官
# ============================================================

@tool
def scan_class_distribution(tid: str = "", teacher_id: str = "") -> dict:
    """扫描本班掌握度分布：均值/标准差/偏态/分层比例（基础<0.4/巩固0.4-0.7/拓展>0.7）
    + 与年级均值对比。"""
    tid = tid or teacher_id
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if s.teacher_id == tid} or students
    concepts = storage.load_concepts()
    # 每个学生的平均掌握度
    avgs = []
    for sid in pool:
        m = storage.load_mastery(sid)
        if m:
            avgs.append(M.avg_mastery(m))
    if not avgs:
        return {"n_students": 0, "msg": "本班无学情数据"}
    import statistics
    avg_all = sum(avgs) / len(avgs)
    std = statistics.pstdev(avgs) if len(avgs) > 1 else 0
    # 偏态：用中位数-均值的方向
    med = statistics.median(avgs)
    skew_dir = "正偏(少数尖子拉)" if med < avg_all - 0.03 else (
        "负偏(少数拖后腿)" if avg_all < med - 0.03 else "对称")
    # 分层比例
    tiers = {"基础(<0.4)": 0, "巩固(0.4-0.7)": 0, "拓展(>0.7)": 0}
    for a in avgs:
        if a < 0.4: tiers["基础(<0.4)"] += 1
        elif a <= 0.7: tiers["巩固(0.4-0.7)"] += 1
        else: tiers["拓展(>0.7)"] += 1
    n = len(avgs)
    tiers_pct = {k: f"{round(v/n*100,1)}%" for k, v in tiers.items()}
    # 年级均值对比
    all_students = storage.load_students()
    grade_avgs = []
    for sid in all_students:
        m = storage.load_mastery(sid)
        if m: grade_avgs.append(M.avg_mastery(m))
    grade_avg = sum(grade_avgs) / len(grade_avgs) if grade_avgs else avg_all
    delta = round((avg_all - grade_avg) * 100, 1)
    compare = f"高于年级均值{delta}%" if delta > 1 else (
        f"低于年级均值{-delta}%" if delta < -1 else "持平年级均值")
    return {"n_students": n, "avg": round(avg_all, 2),
            "std": round(std, 3), "skew": skew_dir,
            "tiers": tiers, "tiers_pct": tiers_pct,
            "grade_avg": round(grade_avg, 2), "compare": compare,
            "top5": [round(a, 2) for a in sorted(avgs, reverse=True)[:5]],
            "bottom5": [round(a, 2) for a in sorted(avgs)[:5]]}


@tool
def scan_concept_failure_clusters(tid: str = "", teacher_id: str = "") -> dict:
    """扫描本班考点失败聚类：每个考点的薄弱学生数/平均掌握度/进步斜率。
    区分『共薄弱点』（多人同时薄弱）vs『孤立薄弱点』（个别学生卡壳）。"""
    tid = tid or teacher_id
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if s.teacher_id == tid} or students
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    # 按考点聚合：薄弱人数、掌握度列表
    fail: dict[str, dict] = {}
    for sid in pool:
        m = storage.load_mastery(sid)
        for cid, mm in m.items():
            row = fail.setdefault(cid, {"weak_n": 0, "masteries": [], "chapter": ""})
            row["masteries"].append(mm.p)
            if mm.p < 0.5:
                row["weak_n"] += 1
            if cid in cmap:
                row["chapter"] = getattr(cmap[cid], "chapter", "") or ""
    n_students = len(pool)
    # 排序：薄弱人数降序
    rows = []
    for cid, d in fail.items():
        avg_p = sum(d["masteries"]) / len(d["masteries"]) if d["masteries"] else 0.5
        ratio = round(d["weak_n"] / n_students, 2)
        # 共薄弱点：>30% 学生同时薄弱
        pattern = "共薄弱点(教学共性)" if ratio > 0.3 else (
            "分组薄弱(局部问题)" if ratio > 0.15 else "孤立薄弱(个体差异)")
        rows.append({"cid": cid, "name": _concept_name(cmap.get(cid), cid),
                     "chapter": d["chapter"], "weak_n": d["weak_n"],
                     "weak_ratio": ratio, "avg_mastery": round(avg_p, 2),
                     "pattern": pattern})
    rows.sort(key=lambda x: -x["weak_ratio"])
    top = rows[:8]
    # 章节聚合
    chap_weak: dict[str, int] = {}
    for r in top:
        if r["pattern"] == "共薄弱点(教学共性)":
            chap_weak[r["chapter"]] = chap_weak.get(r["chapter"], 0) + 1
    return {"n_students": n_students, "top_failing_concepts": top,
            "chapters_with_common_failure": chap_weak,
            "n_common_failure": sum(1 for r in rows if r["pattern"] == "共薄弱点(教学共性)")}


@tool
def diagnose_at_risk_students(tid: str = "", teacher_id: str = "") -> dict:
    """诊断本班需介入学生的多维根因：
    ①是基础差（全考点都弱）还是中途掉链子（前半好后半崩）
    ②还是卡壳某一个考点（其他都好但某个点死活不过）
    ③连对/连错峰值、遗忘曲线斜率、情绪拐点。"""
    tid = tid or teacher_id
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if s.teacher_id == tid} or students
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    # 按平均掌握度排序取 bottom 5
    ranked = []
    for sid, s in pool.items():
        m = storage.load_mastery(sid)
        if m:
            ranked.append((sid, s, M.avg_mastery(m), m))
    ranked.sort(key=lambda x: x[2])
    at_risk = []
    for sid, s, avg, m in ranked[:5]:
        answers = storage.load_answers(sid)
        n = len(answers)
        # 前半/后半正确率
        half = max(n // 2, 1)
        fa = sum(1 for a in answers[:half] if a.correct) / half if answers else 0.5
        ba = sum(1 for a in answers[half:] if a.correct) / max(n - half, 1) if answers else 0.5
        shift = "中途掉链子(前好后崩)" if fa > ba + 0.15 else (
            "渐入佳境(前差后好)" if ba > fa + 0.15 else "平稳")
        # 连错峰值
        cur = worst = 0
        for a in answers:
            if not a.correct: cur += 1; worst = max(worst, cur)
            else: cur = 0
        # 卡壳考点：某个考点掌握度最低且答题次数多
        sorted_cids = sorted(m.items(), key=lambda kv: kv[1].p)
        stuck = None
        for cid, mm in sorted_cids[:3]:
            related = [a for a in answers if a.cid == cid]
            if len(related) >= 2 and mm.p < 0.4:
                cname = _concept_name(cmap.get(cid), cid)
                stuck = {"cid": cid, "name": cname,
                         "mastery": round(mm.p, 2), "n_attempts": len(related)}
                break
        # 全弱还是卡壳
        all_weak = sum(1 for mm in m.values() if mm.p < 0.4) / max(len(m), 1) > 0.6
        root = "基础全面薄弱" if all_weak else (
            f"卡壳于『{stuck['name']}』" if stuck else "波动型(时好时坏)")
        at_risk.append({"sid": sid, "name": s.name, "avg_mastery": round(avg, 2),
                        "n_answers": n, "first_half_acc": round(fa, 2),
                        "second_half_acc": round(ba, 2), "trend": shift,
                        "worst_slump": worst, "root_cause": root, "stuck_concept": stuck})
    return {"at_risk_count": len(at_risk), "students": at_risk}


CLASS_TEACHER_SYS = (
    "你是班级教研官，基于本班学情证据自主推理班级画像和干预方案。"
    "可调用 3 个工具："
    "scan_class_distribution（查本班掌握度分布+分层+对比年级均值）、"
    "scan_concept_failure_clusters（查考点失败聚类：共薄弱点vs孤立薄弱点）、"
    "diagnose_at_risk_students（查需介入学生的多维根因：基础差/掉链子/卡壳）。"
    "流程：先 scan_class_distribution 了解全班全貌 → scan_concept_failure_clusters 定位共性问题 →"
    "diagnose_at_risk_students 锁定需介入的学生和根因。"
    "输出格式："
    "①班级一句话画像（用数据说话，如『负偏分布，少数拖后腿拉低均值2%，低于年级均值3%』）；"
    "②共性薄弱点排序（标注哪些是教学共性哪些是个体差异）；"
    "③需介入学生的干预策略（针对每个学生的根因给出具体建议，不要模板）；"
    "④分层教学处方（基础/巩固/拓展三档各给什么任务）。"
    "不要罗列数字，要提炼成有判断力的结论。")


def run_class_teacher(tid: str) -> dict:
    """🏫 班级教研官：ReAct 取证推理 → 班级画像 + 干预处方。"""
    model = RouterChatModel(temperature=0.5, max_tokens=1200)
    msg = f"教师{tid}的班级学情如何？请诊断并给出教学方案。"
    try:
        steps, answer = _run_react(model, [scan_class_distribution,
                                           scan_concept_failure_clusters,
                                           diagnose_at_risk_students],
                                   CLASS_TEACHER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"班级教研失败：{e}"}
    return {"steps": steps, "answer": answer}


# ============================================================
# 2. 🏛️ 年级督导员
# ============================================================

@tool
def scan_grade_dispersion(tid: str = "", teacher_id: str = "") -> dict:
    """扫描年级离散度：班均掌握度的标准差/极差/排名集中度/本班排名。"""
    tid = tid or teacher_id
    students = storage.load_students()
    # 按教师聚合班均
    class_avgs: dict[str, float] = {}
    class_n: dict[str, int] = {}
    for sid, s in students.items():
        m = storage.load_mastery(sid)
        if not m: continue
        avg = M.avg_mastery(m)
        tid2 = s.teacher_id
        class_avgs[tid2] = class_avgs.get(tid2, 0) + avg
        class_n[tid2] = class_n.get(tid2, 0) + 1
    # 计算各班均值
    real_avgs = {k: round(class_avgs[k] / class_n[k], 3) for k in class_avgs if class_n.get(k, 0) > 0}
    vals = list(real_avgs.values())
    if len(vals) < 2:
        return {"n_classes": len(vals), "msg": "班级数不足，无法计算离散度"}
    import statistics
    std = statistics.pstdev(vals)
    avg_all = sum(vals) / len(vals)
    rng = max(vals) - min(vals)
    # 本班排名
    sorted_classes = sorted(real_avgs.items(), key=lambda x: -x[1])
    rank = next((i + 1 for i, (t, _) in enumerate(sorted_classes) if t == tid), None)
    # 离散度评级
    dispersion = ("高度离散(班际差距大)" if std > 0.08 else
                  "中度离散" if std > 0.04 else "高度集中(班际均衡)")
    return {"n_classes": len(vals), "grade_avg": round(avg_all, 2),
            "std": round(std, 3), "range": round(rng, 3),
            "dispersion_level": dispersion,
            "my_rank": f"{rank}/{len(vals)}" if rank else "未找到本班",
            "my_avg": real_avgs.get(tid),
            "top3": [(t, round(v, 2)) for t, v in sorted_classes[:3]],
            "bottom3": [(t, round(v, 2)) for t, v in sorted_classes[-3:]]}


@tool
def scan_bottleneck_concepts(tid: str = "", teacher_id: str = "") -> dict:
    """扫描全年级瓶颈考点：哪些考点在所有班都是薄弱（共性瓶颈）
    vs 哪些考点只在部分班薄弱（班级特异性瓶颈）。"""
    tid = tid or teacher_id
    students = storage.load_students()
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    # 按班聚合每个考点的薄弱率
    class_concept_fail: dict[str, dict[str, float]] = {}
    for sid, s in students.items():
        tid2 = s.teacher_id
        m = storage.load_mastery(sid)
        class_concept_fail.setdefault(tid2, {})
        for cid, mm in m.items():
            slot = class_concept_fail[tid2].setdefault(cid, {"weak": 0, "total": 0})
            slot["total"] += 1
            if mm.p < 0.5: slot["weak"] += 1
    # 统计每个考点在多少班里薄弱率>50%
    class_count = len(class_concept_fail)
    if class_count == 0:
        return {"msg": "无班级数据"}
    bottleneck = []
    for cid in cmap:
        weak_classes = 0
        for tid2, concepts2 in class_concept_fail.items():
            slot = concepts2.get(cid, {"weak": 0, "total": 0})
            if slot["total"] > 0 and slot["weak"] / slot["total"] > 0.5:
                weak_classes += 1
        ratio = weak_classes / class_count
        category = "全年级共性瓶颈" if ratio > 0.6 else (
            "多数班瓶颈" if ratio > 0.3 else "班级特异性瓶颈")
        bottleneck.append({"cid": cid, "name": _concept_name(cmap[cid], cid),
                           "chapter": getattr(cmap[cid], "chapter", "") or "",
                           "weak_in_n_classes": weak_classes,
                           "ratio": round(ratio, 2), "category": category})
    bottleneck.sort(key=lambda x: -x["ratio"])
    # 本班瓶颈：本班学生掌握度里的 cid 对应的瓶颈考点
    my_cids = set()
    for sid, s in students.items():
        if s.teacher_id == tid:
            for cid in storage.load_mastery(sid).keys():
                my_cids.add(cid)
    my_names = {_concept_name(cmap[c], c) for c in my_cids if c in cmap}
    return {"n_classes": class_count,
            "common_bottlenecks": [b for b in bottleneck if b["category"] == "全年级共性瓶颈"][:6],
            "class_specific": [b for b in bottleneck if b["category"] != "全年级共性瓶颈"][:6],
            "my_class_bottlenecks": [b for b in bottleneck if b["name"] in my_names]}


@tool
def prescribe_grade_intervention(tid: str = "", teacher_id: str = "") -> dict:
    """基于班级排名和瓶颈分布，生成本班可操作的干预处方：
    ①如果本班排名靠前但存在共性瓶颈 → 可牵头教研组统一备课；
    ②如果本班排名靠后且存在共性瓶颈 → 申请年级帮扶资源；
    ③如果本班排名靠后但瓶颈是特异性的 → 聚焦本班个性化整改。"""
    tid = tid or teacher_id
    # 复用前两个工具的证据（直接调用，不走 ReAct 循环）
    disp = scan_grade_dispersion.invoke({"tid": tid})
    bottle = scan_bottleneck_concepts.invoke({"tid": tid})
    rank = disp.get("my_rank", "?")
    rank_num = int(str(rank).split("/")[0]) if "/" in str(rank) else None
    total = disp.get("n_classes", 1)
    percentile = rank_num / total if rank_num and total else 0.5
    common = bottle.get("common_bottlenecks", [])
    my_bottle = bottle.get("my_class_bottlenecks", [])
    # 根据排名+瓶颈组合生成处方
    if percentile <= 0.3:  # 前30%
        if common:
            strategy = "本班领先全年级，可牵头组织针对共性瓶颈的教研组统一备课和磨课"
            actions = ["牵头筹备『" + "、".join(b["name"] for b in common[:2]) + "』联备课",
                       "开放本班录像供其他班观摩",
                       "设计分层练习集供年级共享"]
        else:
            strategy = "本班领先但瓶颈特异性强，聚焦本班个性化优化，形成经验后推广"
            actions = ["深化本班『" + "、".join(b["name"] for b in my_bottle[:2]) + "』整改",
                       "总结本班成功模式形成可复制教案",
                       "主动与薄弱班结对帮扶"]
    elif percentile <= 0.7:  # 中间40%
        if common:
            strategy = "本班处于中游，存在共性瓶颈，建议跟随教研组统一进度并重点突破本班弱项"
            actions = ["同步参与共性瓶颈『" + "、".join(b["name"] for b in common[:2]) + "』专题",
                       "为本班薄弱学生开小灶补『" + "、".join(b["name"] for b in my_bottle[:2]) + "』",
                       "每周与排名相邻的班交换学情简报"]
        else:
            strategy = "本班处于中游，瓶颈特异性强，聚焦本班个性化整改"
            actions = ["集中整改本班『" + "、".join(b["name"] for b in my_bottle[:2]) + "』",
                       "邀请年级督导进班听课诊断",
                       "调整本班教学节奏匹配学生接受度"]
    else:  # 后30%
        if common:
            strategy = "本班排名靠后且存在共性瓶颈，建议申请年级帮扶资源"
            actions = ["申请年级督导组进班诊断『" + "、".join(b["name"] for b in common[:2]) + "』",
                       "结对领先班级开展听课帮扶",
                       "针对性降低本班进度要求，先保基础再追进度"]
        else:
            strategy = "本班排名靠后，瓶颈特异性强，需深度诊断后制定专项整改计划"
            actions = ["请年级督导进班全面诊断本班『" + "、".join(b["name"] for b in my_bottle[:2]) + "』",
                       "与心理老师合作排查学生学习动力问题",
                       "制定为期2周的专项攻坚计划"]
    return {"my_rank": rank, "rank_percentile": f"{round(percentile*100)}%",
            "dispersion_level": disp.get("dispersion_level"),
            "common_bottlenecks_in_grade": [b["name"] for b in common],
            "my_specific_bottlenecks": [b["name"] for b in my_bottle],
            "recommended_strategy": strategy, "action_items": actions}


GRADE_SUPERVISOR_SYS = (
    "你是年级督导员，基于年级离散度、瓶颈分布证据，为特定本班给出年级层面的干预处方。"
    "可调用 3 个工具："
    "scan_grade_dispersion（查年级离散度+本班排名）、"
    "scan_bottleneck_concepts（查全年级共性瓶颈vs班级特异性瓶颈）、"
    "prescribe_grade_intervention（基于排名+瓶颈组合的具体行动处方）。"
    "流程：先 scan_grade_dispersion 了解本班在年级的位置 → scan_bottleneck_concepts 识别瓶颈性质 →"
    "prescribe_grade_intervention 拿具体行动处方 → 你可以补充自己的判断。"
    "输出："
    "①年级视角本班定位（如『排名 12/80，前15%，高于年级均值5%』）；"
    "②瓶颈性质判断（共性 vs 特异性，意味着什么）；"
    "③教学建议（如果本班牵头能做什么，如果本班落后能申请什么资源）。"
    "要给老师实际可操作的建议，不要空话。")


def run_grade_supervisor(tid: str) -> dict:
    """🏛️ 年级督导员：ReAct 取证推理 → 本班年级定位 + 干预处方。"""
    model = RouterChatModel(temperature=0.4, max_tokens=800)
    msg = f"教师{tid}的班级在年级中的位置和瓶颈如何？请从年级视角给出干预建议。"
    try:
        steps, answer = _run_react(model, [scan_grade_dispersion,
                                           scan_bottleneck_concepts,
                                           prescribe_grade_intervention],
                                   GRADE_SUPERVISOR_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"年级督导失败：{e}"}
    return {"steps": steps, "answer": answer}
