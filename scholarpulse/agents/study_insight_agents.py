"""学情洞察 ReAct 智能体矩阵（8 个 create_react_agent）。
把原先 8 个纯统计/规则面板升级为真正的 LangChain ReAct 智能体：
模型自主决定查什么证据、下什么结论，前端展示工具轨迹。

  1. 🌟 星图制图师  — 知识星图拓扑分析 + 学习路径导航
  2. 😊 情绪观察员  — 答题时序推断情绪 + 干预建议
  3. 🧭 画像鉴定师  — 五维深度画像 + 鉴定书
  4. 🧠 负荷监测官  — 认知负荷校准 + 调节处方
  5. 🏫 班级情报官  — 全班分布 + 共性弱点简报
  6. 📉 遗忘预测师  — 掌握度衰减拟合 + 预警
  7. 📑 课件拆解师  — 上传课件后智能拆解 + 学习指南
  8. 📅 半月报告官  — 15 天成长报告

共用 diagnosis_agents._run_react 通用 ReAct 循环（含 markdown 清洗 + 截断兜底）。
"""
from __future__ import annotations
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel


# ============================================================
# 1. 🌟 星图制图师
# ============================================================
@tool
def scan_topology(sid: str) -> dict:
    """扫描知识星图拓扑：节点数/边数/孤立节点/桥接概念/最长依赖链深度。"""
    from .. import storage
    concepts = storage.load_concepts()
    if not concepts:
        return {"n_nodes": 0, "msg": "尚未上传课件，知识星图为空"}
    ids = {c.cid for c in concepts}
    cmap = {c.cid: c for c in concepts}
    edges = sum(len(c.prerequisites) for c in concepts)
    # 孤立：既无前置也不是别人的前置
    be_depended = {p for c in concepts for p in c.prerequisites if p in ids}
    has_prereq = {c.cid for c in concepts if c.prerequisites}
    isolated = [c.name for c in concepts
                if c.cid not in be_depended and c.cid not in has_prereq]
    # 桥接：被 ≥2 个节点依赖的中间概念
    dep_count: dict[str, int] = {}
    for c in concepts:
        for p in c.prerequisites:
            dep_count[p] = dep_count.get(p, 0) + 1
    bridges = sorted(((cmap[p].name, n) for p, n in dep_count.items() if n >= 2 and p in cmap),
                     key=lambda x: -x[1])[:5]
    # 最长链深度
    def depth(cid, seen):
        if cid not in cmap or cid in seen:
            return 0
        seen.add(cid)
        ps = [p for p in cmap[cid].prerequisites if p in cmap]
        return 1 + max((depth(p, seen) for p in ps), default=0)
    max_depth = max((depth(c.cid, set()) for c in concepts), default=0)
    return {"n_nodes": len(concepts), "n_edges": edges,
            "isolated": isolated[:6], "bridges": bridges,
            "max_chain_depth": max_depth}


@tool
def path_suggest(sid: str) -> dict:
    """基于掌握度+先修依赖生成学习路径：拓扑序中掌握度最低的前 8 个考点。"""
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
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
    scored = sorted(order, key=lambda cid: (
        mastery[cid].p if cid in mastery else 2.0,))[:8]
    return {"path": [{"cid": cid, "name": cmap[cid].name,
                      "mastery": round(mastery[cid].p, 2) if cid in mastery else None,
                      "chapter": cmap[cid].chapter}
                     for cid in scored if cid in cmap]}


@tool
def assess_topological_strategy(sid: str) -> dict:
    """拓扑战略证据：基于星图拓扑(孤立岛/桥接枢纽/链深度)+掌握度梯度，
    采集战略推理证据——哪些枢纽打通收益最大、哪些孤岛是知识盲区、
    最长依赖链的瓶颈在哪层，供 LLM 自主推理最优学习战略顺序。

    工具只给战略证据不下结论——让 LLM 从『枢纽X被5个考点依赖但掌握度0.2+
    孤岛Y掌握度0』等证据自己推理『先打通X再攻Y』的战略，而非套路径模板。"""
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
    if not concepts:
        return {"n_nodes": 0, "msg": "星图为空"}
    # 桥接枢纽被依赖次数 + 掌握度（低掌握+高被依赖=高战略价值）
    dep_count: dict[str, int] = {}
    for c in concepts:
        for p in c.prerequisites:
            dep_count[p] = dep_count.get(p, 0) + 1
    bridges = []
    for pid, n in dep_count.items():
        if pid in cmap:
            m = mastery[pid].p if pid in mastery else 2.0
            # 战略价值=被依赖次数×(1-掌握度)：枢纽被依赖多又掌握低=打通收益大
            bridges.append({"cid": pid, "name": cmap[pid].name,
                            "depended_by": n, "mastery": round(m, 2),
                            "strategy_value": round(n * (1 - min(m, 1)), 2)})
    bridges.sort(key=lambda x: -x["strategy_value"])
    # 孤岛：既无前置也不是别人的前置（知识盲区）
    ids = {c.cid for c in concepts}
    be_depended = {p for c in concepts for p in c.prerequisites if p in ids}
    has_prereq = {c.cid for c in concepts if c.prerequisites}
    isolated = [{"cid": c.cid, "name": c.name,
                 "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None}
                for c in concepts
                if c.cid not in be_depended and c.cid not in has_prereq]
    # 最长依赖链瓶颈：沿最深链找掌握度最低的环节
    def chain_depth(cid, seen):
        if cid not in cmap or cid in seen:
            return 0, []
        seen.add(cid)
        ps = [p for p in cmap[cid].prerequisites if p in cmap]
        if not ps:
            return 1, [cid]
        best_d, best_chain = 0, []
        for p in ps:
            d, ch = chain_depth(p, seen.copy())
            if d > best_d:
                best_d, best_chain = d, ch
        return best_d + 1, [cid] + best_chain
    deepest = max((chain_depth(c.cid, set()) for c in concepts), key=lambda x: x[0])
    bottleneck_chain = [{"cid": cid, "name": cmap.get(cid, type("x",(),{"name":cid})()).name,
                         "mastery": round(mastery[cid].p, 2) if cid in mastery else None}
                        for cid in deepest[1]]
    # 瓶颈环节=链上掌握度最低
    return {"high_value_bridges": bridges[:4],
            "isolated_islands": isolated[:5],
            "deepest_chain": {"depth": deepest[0], "bottleneck_chain": bottleneck_chain[:6]},
            "evidence_brief": (f"最高战略价值枢纽:{bridges[0]['name'] if bridges else '无'}"
                               f"(收益{bridges[0]['strategy_value'] if bridges else 0}),"
                               f"孤岛{len(isolated)}个,最深链{deepest[0]}层")}


STAR_CARTOGRAPHER_SYS = (
    "你是知识星图制图师，基于拓扑结构+掌握度梯度自主推理最优学习战略并导航。"
    "可调用工具：scan_topology（查星图拓扑：孤立岛/桥接枢纽/链深度）、"
    "path_suggest（基于掌握度+先修序生成学习路径）、"
    "assess_topological_strategy（查枢纽战略价值/孤岛盲区/最深链瓶颈等战略证据）。"
    "流程：先 scan_topology 看星图全貌，再 assess_topological_strategy 取战略证据，"
    "最后 path_suggest 取路径，然后综合推理。"
    "关键：不要只列路径，要基于战略证据自己推理学习顺序"
    "（如『先打通收益最大的枢纽X(被5个考点依赖且未掌握)→再攻孤岛Y盲区→最后沿最深链逐层推进』）。"
    "输出≤140字星图简报：用『地形』比喻+推理出的战略顺序（先打哪里、为什么、收益多大）。")


def run_star_cartographer(sid: str) -> dict:
    """🌟 星图制图师：LLM 自主推理拓扑战略 + 导航（返回 steps + answer）。"""
    model = RouterChatModel(temperature=0.5, max_tokens=550)
    msg = f"学生{sid}想了解自己的知识星图地形，请基于拓扑战略证据推理最优学习顺序并导航。"
    try:
        steps, answer = _run_react(model, [scan_topology, path_suggest,
                                           assess_topological_strategy],
                                   STAR_CARTOGRAPHER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"星图制图失败：{e}"}
    return {"steps": steps, "answer": answer,
            "topology": scan_topology.invoke({"sid": sid}),
            "path": path_suggest.invoke({"sid": sid})["path"],
            "strategy": assess_topological_strategy.invoke({"sid": sid})}


# ============================================================
# 2. 😊 情绪观察员
# ============================================================
@tool
def scan_mood(sid: str) -> dict:
    """扫描答题时序情绪信号：连对/连错峰值、前后半场正确率走势、答题密度、情绪标签。"""
    from .. import storage
    answers = storage.load_answers(sid)
    if not answers:
        return {"n": 0, "mood": "平静", "msg": "尚无答题数据"}
    n = len(answers)
    # 连对/连错峰值
    best_up = cur_up = best_down = cur_down = 0
    for a in answers:
        if a.correct:
            cur_up += 1; cur_down = 0; best_up = max(best_up, cur_up)
        else:
            cur_down += 1; cur_up = 0; best_down = max(best_down, cur_down)
    half = max(n // 2, 1)
    fa = sum(1 for a in answers[:half] if a.correct) / half
    ba = sum(1 for a in answers[half:] if a.correct) / max(n - half, 1)
    recent = answers[-10:]
    r_acc = sum(1 for a in recent if a.correct) / len(recent)
    # 情绪标签（规则基准，智能体可覆盖）
    if r_acc >= 0.7 and best_up >= 3:
        mood = "心流"
    elif best_down >= 4:
        mood = "受挫"
    elif r_acc < 0.4:
        mood = "焦虑"
    elif ba > fa + 0.15:
        mood = "渐入佳境"
    elif ba < fa - 0.15:
        mood = "疲软"
    else:
        mood = "平稳"
    return {"n": n, "best_streak": best_up, "worst_slump": best_down,
            "first_half_acc": round(fa, 2), "second_half_acc": round(ba, 2),
            "recent_acc": round(r_acc, 2), "trend": "↑" if ba > fa else "↓" if ba < fa else "→",
            "mood_signal": mood}


@tool
def scan_recent_errors(sid: str) -> dict:
    """查最近 5 道错题的考点名+题干，用于情绪归因（是不是卡在同一类题）。"""
    from .. import storage
    cmap = {c.cid: c.name for c in storage.load_concepts()}
    wrongs = [a for a in storage.load_answers(sid) if not a.correct][-5:]
    if not wrongs:
        return {"n": 0, "msg": "近期无错题"}
    return {"errors": [{"cid": a.cid, "name": cmap.get(a.cid, a.cid),
                        "q": (a.question or "")[:50]} for a in wrongs],
            "n": len(wrongs)}


@tool
def trace_emotion_trajectory(sid: str) -> dict:
    """情绪演变轨迹证据：把答题序列切成4段，每段输出对错序列/连对错峰值/正确率/耗时，
    作为情绪如何随时间演变的原始证据，供 LLM 自主推理情绪轨迹（而非单一标签）。

    工具只给证据不下结论——让 LLM 从『前段连对→中段断裂→尾段回升』等轨迹自己判断
    心理状态演变（如『开场顺利→中段遇挫卡壳→尾段重新找回手感』）。"""
    from .. import storage
    answers = storage.load_answers(sid)
    if not answers:
        return {"n": 0, "msg": "无答题数据"}
    n = len(answers)
    q = max(n // 4, 1)
    segs = []
    for k in range(4):
        seg = answers[k * q:(k + 1) * q] if k < 3 else answers[3 * q:]
        if not seg:
            continue
        marks = "".join("✔" if a.correct else "✘" for a in seg)
        acc = sum(1 for a in seg if a.correct) / len(seg)
        # 连对/连错峰值
        bu = cd = cu = 0
        for a in seg:
            if a.correct:
                cu += 1; cd = 0; bu = max(bu, cu)
            else:
                cd += 1; cu = 0
        segs.append({"seg": k + 1, "n": len(seg), "marks": marks,
                     "acc": round(acc, 2), "best_streak": bu, "worst_slump": cd})
    # 段间正确率差分（拐点检测）
    diffs = [round(segs[i + 1]["acc"] - segs[i]["acc"], 2)
             for i in range(len(segs) - 1) if i + 1 < len(segs)]
    return {"n": n, "segments": segs, "acc_diffs": diffs,
            "trajectory_hint": ("上升" if all(d > 0 for d in diffs)
                                else "下降" if all(d < 0 for d in diffs)
                                else "波动" if any(abs(d) > 0.2 for d in diffs)
                                else "平稳")}


MOOD_OBSERVER_SYS = (
    "你是情绪观察员，从答题行为时序的演变自主推理学生心理状态并给出关怀式干预。"
    "可调用工具：scan_mood（查连对/连错峰值、前后半场走势、情绪信号——注意mood_signal只是规则基准）、"
    "scan_recent_errors（查最近错题，判断是否卡在同一类）、"
    "trace_emotion_trajectory（把答题切4段，给对错序列/连对错/正确率/段间差分原始证据）。"
    "流程：先 trace_emotion_trajectory 读情绪演变轨迹证据（4段对错序列+拐点），"
    "再 scan_mood 看整体信号，若轨迹显示受挫/焦虑则 scan_recent_errors 归因是否卡同一考点。"
    "关键：不要直接采用工具给的mood_signal标签，要基于轨迹证据自己推理情绪如何演变"
    "（如『前段顺利→中段遇挫→尾段回升』），用行为证据说话而非贴标签。"
    "输出≤110字：描述情绪演变轨迹+当前状态+一条具体干预建议（如『先回看X讲解再战』）。")


def run_mood_observer(sid: str) -> dict:
    """😊 情绪观察员：LLM 自主推理情绪演变轨迹 + 干预（返回 steps + answer）。"""
    model = RouterChatModel(temperature=0.6, max_tokens=450)
    msg = f"学生{sid}的情绪状态如何？请基于答题轨迹证据推理情绪演变并给出干预建议。"
    try:
        steps, answer = _run_react(model, [scan_mood, scan_recent_errors,
                                           trace_emotion_trajectory],
                                   MOOD_OBSERVER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"情绪观察失败：{e}"}
    return {"steps": steps, "answer": answer,
            "signal": scan_mood.invoke({"sid": sid}),
            "trajectory": trace_emotion_trajectory.invoke({"sid": sid})}


# ============================================================
# 3. 🧭 画像鉴定师
# ============================================================
@tool
def scan_profile(sid: str) -> dict:
    """五维深度画像：掌握度/活跃度/稳定性/进步性/广度 + 错题章节分布。"""
    from .. import storage
    from .. import mastery as M
    concepts = storage.load_concepts()
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    avg = M.avg_mastery(mastery)
    n = len(answers)
    best = cur = 0
    for a in answers:
        cur = cur + 1 if a.correct else 0
        best = max(best, cur)
    half = max(n // 2, 1)

    def acc(xs):
        return sum(1 for a in xs if a.correct) / len(xs) if xs else 0

    fa, ba = acc(answers[:half]), acc(answers[half:])
    prog = 0.5 + ((ba - fa) * 1.5 if fa or ba else 0)
    breadth = sum(1 for c in concepts if c.cid in mastery and mastery[c.cid].p >= 0.5) / max(len(concepts), 1)
    dims = {"掌握度": round(avg, 2), "活跃度": round(min(n / 30, 1.0), 2),
            "稳定性": round(min(best / 10, 1.0), 2), "进步性": round(max(0, min(1, prog)), 2),
            "广度": round(breadth, 2)}
    # 错题章节分布
    chap_err: dict[str, int] = {}
    cmap = {c.cid: c for c in concepts}
    for a in answers:
        if not a.correct and a.cid in cmap:
            ch = cmap[a.cid].chapter or "未分类"
            chap_err[ch] = chap_err.get(ch, 0) + 1
    return {"dims": dims, "best_streak": best, "n": n,
            "error_chapters": sorted(chap_err.items(), key=lambda x: -x[1])[:5],
            "avg": round(avg, 2)}


@tool
def collect_archetype_evidence(sid: str) -> dict:
    """学习原型推理证据：采集偏科程度/粗心指数/进步斜率/知识盲区数等原始证据，
    供 LLM 自主推理学生属于哪种学习原型（偏科型/稳扎型/急起直追型/粗心型/摇摆型）。

    工具只给量化证据不下结论——让 LLM 从『偏科0.7+粗心0.4+盲区2』等证据组合自己判断原型与根因，
    而非套用规则标签。"""
    from .. import storage
    from .. import mastery as M
    concepts = storage.load_concepts()
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
    n = len(answers)
    if not n:
        return {"n": 0, "msg": "无答题数据"}
    # 偏科程度：错题章节集中度（基尼系数式）
    chap_err: dict[str, int] = {}
    for a in answers:
        if not a.correct and a.cid in cmap:
            ch = cmap[a.cid].chapter or "未分类"
            chap_err[ch] = chap_err.get(ch, 0) + 1
    tot_err = sum(chap_err.values()) or 1
    share = sorted(chap_err.values(), reverse=True)
    focus = sum(s * s for s in share) / (tot_err * tot_err) if share else 0  # 集中度
    # 粗心指数：基础题(tier=foundation)错率
    found_total = found_wrong = 0
    for a in answers:
        c = cmap.get(a.cid)
        if c and c.tier == "foundation":
            found_total += 1
            if not a.correct:
                found_wrong += 1
    careless = round(found_wrong / max(found_total, 1), 2)
    # 进步斜率：前1/3 vs 后1/3 正确率差
    k = max(n // 3, 1)
    fa = sum(1 for a in answers[:k] if a.correct) / max(k, 1)
    ba = sum(1 for a in answers[-k:] if a.correct) / max(k, 1)
    slope = round(ba - fa, 2)
    # 知识盲区数：掌握度<0.3 的考点数
    blind = sum(1 for cid, m in mastery.items() if m.p < 0.3)
    # 稳定性：连对最大 vs 连错最大
    bu = cd = cu = 0
    for a in answers:
        if a.correct:
            cu += 1; cd = 0; bu = max(bu, cu)
        else:
            cd += 1; cu = 0
    stability = round(bu / max(bu + cd, 1), 2)
    return {"n": n, "chapter_focus": round(focus, 2),
            "careless_index": careless, "progress_slope": slope,
            "n_blind_spots": blind, "stability": stability,
            "best_streak": bu, "worst_slump": cd,
            "top_error_chapters": sorted(chap_err.items(), key=lambda x: -x[1])[:3]}


PROFILE_APPRAISER_SYS = (
    "你是画像鉴定师，用五维雷达+错题分布+学习原型证据自主推理学生画像并出具鉴定书。"
    "可调用工具：scan_profile（查五维数据+错题章节分布）、"
    "collect_archetype_evidence（查偏科程度/粗心指数/进步斜率/盲区数等原型推理证据）。"
    "流程：先 scan_profile 取五维+错题分布，再 collect_archetype_evidence 取原型证据，"
    "然后综合推理。"
    "关键：不要套用固定标签，要基于原型证据组合自己推理学生属于哪种学习原型"
    "（如偏科0.7+盲区3=偏科型，粗心0.5+稳定性高=粗心型，slope+0.3=急起直追型）并说出根因。"
    "输出≤130字鉴定书：最突出长项+短板（数据说话）+推理出的学习原型与根因+一句话进阶建议。")


def run_profile_appraiser(sid: str) -> dict:
    """🧭 画像鉴定师：LLM 自主推理学习原型 + 鉴定书（返回 steps + answer + dims）。"""
    model = RouterChatModel(temperature=0.5, max_tokens=500)
    msg = f"学生{sid}的画像鉴定，请基于原型证据推理学习原型并出具鉴定书。"
    try:
        steps, answer = _run_react(model, [scan_profile, collect_archetype_evidence],
                                   PROFILE_APPRAISER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"画像鉴定失败：{e}"}
    prof = scan_profile.invoke({"sid": sid})
    return {"steps": steps, "answer": answer, "dims": prof["dims"],
            "best_streak": prof["best_streak"], "n": prof["n"],
            "error_chapters": prof["error_chapters"]}


# ============================================================
# 4. 🧠 负荷监测官
# ============================================================
@tool
def scan_load(sid: str) -> dict:
    """扫描认知负荷源：到期复习密度/近期答题密度/正确率走势/难度配比。"""
    from .. import storage
    from ..core.spaced_repetition import due_review
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    due = due_review(mastery) if mastery else []
    recent = answers[-10:]
    r_acc = sum(1 for a in recent if a.correct) / len(recent) if recent else None
    # 难度配比（近期答题的 tier 分布）
    cmap = {c.cid: c for c in storage.load_concepts()}
    tier_cnt = {"foundational": 0, "core": 0, "advanced": 0}
    for a in recent:
        c = cmap.get(a.cid)
        if c:
            tier_cnt[c.tier] = tier_cnt.get(c.tier, 0) + 1
    # 负荷合成：到期多+正确率低+密度高 = 高负荷
    load = 0
    load += min(len(due) / 8, 0.4)
    if r_acc is not None:
        load += (1 - r_acc) * 0.4
    load += min(len(recent) / 20, 0.2)
    return {"n_due": len(due), "recent_acc": round(r_acc, 2) if r_acc is not None else None,
            "recent_n": len(recent), "tier_mix": tier_cnt,
            "load": round(min(load, 1.0), 2)}


@tool
def scan_calibration(sid: str) -> dict:
    """查把握度校准偏差：学生自评把握 vs 实际正确率的偏差（过度自信/过度谨慎）。"""
    from .. import storage
    answers = storage.load_answers(sid)
    if not answers:
        return {"n": 0, "msg": "无答题数据"}
    # AnswerRecord 没存 confidence，用掌握度 p 近似校准
    mastery = storage.load_mastery(sid)
    over_confident = sum(1 for cid, m in mastery.items() if m.p > 0.7 and m.n_wrong > m.n_correct)
    under_confident = sum(1 for cid, m in mastery.items() if m.p < 0.3 and m.n_correct > m.n_wrong)
    return {"n": len(answers),
            "over_confident_concepts": over_confident,
            "under_confident_concepts": under_confident,
            "bias": "过度自信" if over_confident > under_confident else "过度谨慎" if under_confident > over_confident else "校准良好"}


@tool
def diagnose_load_root_cause(sid: str) -> dict:
    """负荷根因证据：深入采集负荷过载的真正根因证据——
    到期密度vs新学密度失衡、难度配比是否超出能力区、校准偏差与连错耦合、
    负荷趋势(正确率下滑还是密度上升)，供 LLM 自主推理根因而非只看负荷指数。

    工具只给根因证据不下结论——让 LLM 从『到期12项+难度配比进阶占60%+
    过度自信3概念+连错峰值4』等证据自己推理『不是题多而是盲目冲超纲题+
    忽视复习到期』的根因。"""
    from .. import storage
    from ..core.spaced_repetition import due_review
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    due = due_review(mastery) if mastery else []
    recent = answers[-15:]
    # 根因1：到期密度 vs 新学密度失衡（到期远多于近期答题=被复习积压压垮）
    due_pressure = len(due)
    learn_rate = len(recent)
    review_learn_ratio = round(due_pressure / max(learn_rate, 1), 2)
    review_overload = due_pressure > learn_rate * 2 and due_pressure >= 5
    # 根因2：难度配比是否超出能力区（进阶题占比高但掌握度低=盲目冲超纲）
    tier_cnt = {"foundational": 0, "core": 0, "advanced": 0}
    adv_low_mastery = 0
    for a in recent:
        c = cmap.get(a.cid)
        if c:
            tier_cnt[c.tier] = tier_cnt.get(c.tier, 0) + 1
            if c.tier == "advanced" and a.cid in mastery and mastery[a.cid].p < 0.4:
                adv_low_mastery += 1
    total_t = sum(tier_cnt.values()) or 1
    adv_ratio = round(tier_cnt.get("advanced", 0) / total_t, 2)
    capability_mismatch = adv_ratio > 0.4 and adv_low_mastery >= 2
    # 根因3：校准偏差×连错耦合（过度自信+连错=盲目自信冲难题）
    over_conf = sum(1 for cid, m in mastery.items()
                    if m.p > 0.7 and m.n_wrong > m.n_correct)
    bu = cd = cu = 0
    for a in recent:
        if a.correct:
            cu += 1; cd = 0; bu = max(bu, cu)
        else:
            cd += 1; cu = 0
    confidence_mismatch = over_conf >= 2 and cd >= 3
    # 根因4：负荷趋势（前7 vs 后8 正确率，下滑=过载恶化）
    if len(recent) >= 6:
        h = len(recent) // 2
        early = sum(1 for a in recent[:h] if a.correct) / max(h, 1)
        late = sum(1 for a in recent[h:] if a.correct) / max(len(recent) - h, 1)
        trend = round(late - early, 2)
        deteriorating = trend < -0.15
    else:
        trend, deteriorating = None, False
    roots = []
    if review_overload:
        roots.append(f"复习积压({due_pressure}到期vs{learn_rate}新学)")
    if capability_mismatch:
        roots.append(f"难度超纲(进阶{adv_ratio}占比但掌握不足{adv_low_mastery}个)")
    if confidence_mismatch:
        roots.append(f"盲目自信({over_conf}过度自信概念+连错{cd})")
    if deteriorating:
        roots.append(f"过载恶化(趋势{trend})")
    return {"review_learn_ratio": review_learn_ratio, "review_overload": review_overload,
            "tier_mix": tier_cnt, "advanced_ratio": adv_ratio,
            "capability_mismatch": capability_mismatch,
            "over_confident_concepts": over_conf, "recent_worst_slump": cd,
            "confidence_mismatch": confidence_mismatch,
            "acc_trend": trend, "deteriorating": deteriorating,
            "root_cause_evidence": roots or ["负荷水平在合理范围，无明显结构性根因"]}


LOAD_MONITOR_SYS = (
    "你是认知负荷监测官，基于负荷指数+根因证据自主诊断负荷过载的真正根因并开具调节处方。"
    "可调用工具：scan_load（查到期密度/近期正确率/难度配比/负荷指数——综合快照）、"
    "scan_calibration（查把握度校准偏差）、"
    "diagnose_load_root_cause（查复习积压/难度超纲/盲目自信耦合/过载恶化等根因证据）。"
    "流程：先 scan_load 读负荷全景，再 diagnose_load_root_cause 取根因证据，"
    "最后 scan_calibration 看校准偏差，然后综合推理。"
    "关键：不要只报绿/黄/红区，要基于根因证据自己推理过载的真正原因"
    "（如『不是题多而是盲目冲超纲进阶题+忽视复习积压』）并针对根因开处方。"
    "输出≤120字处方：判负荷水平+推理出的根因+针对性调节建议（针对根因而非泛泛而谈）。")


def run_load_monitor(sid: str) -> dict:
    """🧠 负荷监测官：LLM 自主推理负荷根因 + 处方（返回 steps + answer）。"""
    model = RouterChatModel(temperature=0.4, max_tokens=480)
    msg = f"学生{sid}的认知负荷监测，请基于根因证据推理过载原因并开具处方。"
    try:
        steps, answer = _run_react(model, [scan_load, scan_calibration,
                                           diagnose_load_root_cause],
                                   LOAD_MONITOR_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"负荷监测失败：{e}"}
    return {"steps": steps, "answer": answer, "load": scan_load.invoke({"sid": sid}),
            "calibration": scan_calibration.invoke({"sid": sid}),
            "root_cause": diagnose_load_root_cause.invoke({"sid": sid})}


# ============================================================
# 5. 🏫 班级情报官
# ============================================================
@tool
def scan_class(sid: str) -> dict:
    """扫描全班分布：同班各人掌握度/我的排名百分位/梯队划分/共性薄弱考点。"""
    from .. import storage
    from .. import mastery as M
    import random as _rd
    students = storage.load_students()
    me = students.get(sid)
    if not me:
        return {"error": "学生不存在"}
    same = [s.sid for s in students.values()
            if getattr(s, "teacher_id", None) == getattr(me, "teacher_id", None) and s.sid != sid]
    if not same:
        same = _rd.sample([s.sid for s in students.values() if s.sid != sid],
                          min(40, len(students) - 1))
    my_m = M.avg_mastery(storage.load_mastery(sid))
    vals, weak_map = [], {}
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    for p in same[:40]:
        m = storage.load_mastery(p)
        if m:
            vals.append(M.avg_mastery(m))
            # 统计全班薄弱考点
            for cid, mas in m.items():
                if mas.p < 0.4:
                    weak_map[cid] = weak_map.get(cid, 0) + 1
    vals.append(my_m)
    vals_sorted = sorted(vals)
    rank = sum(1 for v in vals if v > my_m) + 1
    pct = round((1 - (rank - 1) / len(vals)) * 100)
    # 梯队划分
    tiers = {"领跑(≥70%)": 0, "中坚(40-70%)": 0, "追赶(<40%)": 0}
    for v in vals:
        if v >= 0.7:
            tiers["领跑(≥70%)"] += 1
        elif v >= 0.4:
            tiers["中坚(40-70%)"] += 1
        else:
            tiers["追赶(<40%)"] += 1
    common_weak = sorted(((cmap[c].name, n) for c, n in weak_map.items() if c in cmap),
                         key=lambda x: -x[1])[:5]
    return {"my": round(my_m, 3), "class_avg": round(sum(vals) / len(vals), 3),
            "rank": rank, "n": len(vals), "pct": pct,
            "tiers": tiers, "common_weak": common_weak,
            "hist": [round(v, 2) for v in vals_sorted]}


@tool
def infer_class_dynamics(sid: str) -> dict:
    """班级学习动力学证据：分析全班掌握度分布形态(双峰/正态/左偏/右偏)、
    共性薄弱聚类(哪些考点大家集体卡住)、梯队人数配比、竞争烈度(相邻排名分差)，
    作为推理班级学习氛围/竞争态势/共性根因的原始证据。

    工具只给分布形态证据不下结论——让 LLM 从『双峰分布+3个共性薄弱+梯队断层』等证据
    自己推理班级是『两极分化型/整体偏弱型/均衡竞争型』及共性根因。"""
    import statistics as _st
    from .. import storage
    from .. import mastery as M
    import random as _rd
    students = storage.load_students()
    me = students.get(sid)
    if not me:
        return {"error": "学生不存在"}
    same = [s.sid for s in students.values()
            if getattr(s, "teacher_id", None) == getattr(me, "teacher_id", None) and s.sid != sid]
    if not same:
        same = _rd.sample([s.sid for s in students.values() if s.sid != sid],
                          min(40, len(students) - 1))
    vals, weak_map = [], {}
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    for p in same[:40]:
        m = storage.load_mastery(p)
        if m:
            vals.append(M.avg_mastery(m))
            for cid, mas in m.items():
                if mas.p < 0.4:
                    weak_map[cid] = weak_map.get(cid, 0) + 1
    if not vals:
        return {"n": 0, "msg": "无同班数据"}
    # 分布形态：均值+标准差+偏度+峰度
    avg = sum(vals) / len(vals)
    sd = _st.pstdev(vals) if len(vals) > 1 else 0
    # 偏度：>0.3右偏(整体偏弱)，<-0.3左偏(整体偏强)，|<0.3|近正态
    try:
        skew = round(_st.skew(vals), 2) if len(vals) >= 3 else 0
    except Exception:
        skew = 0
    # 双峰检测：看是否有两个聚集中心
    sorted_v = sorted(vals)
    gaps = [sorted_v[i + 1] - sorted_v[i] for i in range(len(sorted_v) - 1)]
    max_gap_idx = gaps.index(max(gaps)) if gaps else -1
    bimodal = bool(max_gap_idx >= 0 and max(gaps) > 2 * sd and max_gap_idx > 0 and max_gap_idx < len(sorted_v) - 2)
    # 共性薄弱聚类：按章节聚合
    common_clusters: dict[str, list] = {}
    for cid, cnt in weak_map.items():
        if cid in cmap and cnt >= 2:
            ch = cmap[cid].chapter or "未分类"
            common_clusters.setdefault(ch, []).append({"name": cmap[cid].name, "n": cnt})
    for ch in common_clusters:
        common_clusters[ch].sort(key=lambda x: -x["n"])
    # 竞争烈度：我周围5名的分差
    my_m = M.avg_mastery(storage.load_mastery(sid))
    sv = sorted(vals + [my_m])
    my_idx = sv.index(my_m) if my_m in sv else 0
    nearby = sv[max(0, my_idx - 2):my_idx + 3]
    competition = round(max(nearby) - min(nearby), 3) if len(nearby) > 1 else 0
    shape = ("双峰(两极分化)" if bimodal
             else "右偏(整体偏弱)" if skew > 0.3
             else "左偏(整体偏强)" if skew < -0.3
             else "近正态(均衡)")
    return {"n": len(vals), "class_avg": round(avg, 3), "sd": round(sd, 3),
            "skew": skew, "distribution_shape": shape,
            "bimodal": bimodal,
            "common_weak_clusters": dict(list(common_clusters.items())[:4]),
            "competition_around_me": competition,
            "evidence_brief": f"分布:{shape},均值{round(avg,2)},共性薄弱{sum(len(v) for v in common_clusters.values())}个考点聚类,周边分差{competition}"}


CLASS_INTEL_SYS = (
    "你是班级情报官，负责把全班数据+分布形态证据翻译成给学生看的班情简报。"
    "可调用工具：scan_class（查全班分布/我的排名百分位/梯队/共性薄弱考点）、"
    "infer_class_dynamics（查分布形态/共性薄弱聚类/竞争烈度等动力学证据）。"
    "流程：先 scan_class 取全班情报，再 infer_class_dynamics 取动力学证据，然后综合推理。"
    "关键：不要只列数据，要基于分布形态证据自己推理班级学习氛围"
    "（双峰=两极分化需分层教学，左偏=整体偏强竞争激烈，共性聚类=集体卡点根因）。"
    "输出≤130字班情简报：我处在什么梯队+离上一档差多少+推理出的班级氛围与共性根因"
    "+一条基于班情的行动建议。")


def run_class_intel(sid: str) -> dict:
    """🏫 班级情报官：LLM 自主推理班情 + 简报（返回 steps + answer）。"""
    model = RouterChatModel(temperature=0.5, max_tokens=500)
    msg = f"学生{sid}想了解班级情报，请基于分布形态证据推理班情并出具简报。"
    try:
        steps, answer = _run_react(model, [scan_class, infer_class_dynamics],
                                    CLASS_INTEL_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"班级情报失败：{e}"}
    return {"steps": steps, "answer": answer, **scan_class.invoke({"sid": sid})}


# ============================================================
# 6. 📉 遗忘预测师
# ============================================================
@tool
def scan_decay(sid: str) -> dict:
    """扫描各考点遗忘状态：当前掌握度 × 距上次作答天数，标注高危（即将遗忘）。"""
    import datetime as _dt
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(sid)
    last_by_cid: dict[str, str] = {}
    for a in storage.load_answers(sid):
        last_by_cid[a.cid] = str(a.ts)[:10]
    now = _dt.date.today()
    pts, at_risk = [], []
    for cid, d in last_by_cid.items():
        try:
            days = (now - _dt.date.fromisoformat(d)).days
        except ValueError:
            continue
        if cid in mastery:
            p = mastery[cid].p
            pts.append({"cid": cid, "name": name_of.get(cid, cid),
                        "days": days, "retention": round(p, 2)})
            # 艾宾浩斯理论保持率：R = e^(-t/S)，S≈5 天（粗略）
            theory = round(2.718 ** (-days / 5), 2)
            if p < theory and days >= 3:
                at_risk.append({"cid": cid, "name": name_of.get(cid, cid),
                                "days": days, "p": round(p, 2),
                                "theory": theory})
    at_risk.sort(key=lambda x: x["p"])
    return {"points": pts, "at_risk": at_risk[:6],
            "n_tracked": len(pts)}


@tool
def forecast_retention_curve(sid: str) -> dict:
    """遗忘曲线外推预测：基于各考点当前掌握度+距上次作答天数，用艾宾浩斯公式
    R=e^(-t/S) 外推未来7/15/30天保持率，标注将跌入危险区(<0.4)的考点与跌入时间。

    供 LLM 自主推理哪些考点即将遗忘、何时该介入复习——而非只看当前快照。"""
    import datetime as _dt
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(sid)
    last_by_cid: dict[str, str] = {}
    for a in storage.load_answers(sid):
        last_by_cid[a.cid] = str(a.ts)[:10]
    now = _dt.date.today()
    forecast = []
    for cid, m in mastery.items():
        last = last_by_cid.get(cid)
        if not last:
            continue
        try:
            days = (now - _dt.date.fromisoformat(last)).days
        except ValueError:
            continue
        # 稳定性 S 估计：掌握度越高、答题次数越多→S 越大（记得越牢）
        n_total = m.n_correct + m.n_wrong
        S = max(3, 5 + m.p * 10 + min(n_total, 10))  # 3~25 天
        # 当前保持率（已衰减）
        cur_R = round(2.718 ** (-days / S), 2)
        # 未来 7/15/30 天保持率外推（基于当前衰减速率）
        future = {}
        for d in (7, 15, 30):
            t = days + d
            future[str(d)] = round(max(2.718 ** (-t / S), 0), 2)
        # 跌入危险区(<0.4)的天数
        import math
        danger_day = round(S * math.log(1 / 0.4) - days) if cur_R > 0.4 else -1
        forecast.append({"cid": cid, "name": name_of.get(cid, cid),
                         "current_R": cur_R, "S": round(S, 1),
                         "days_since": days, "future": future,
                         "danger_in_days": danger_day if danger_day > 0 else None})
    forecast.sort(key=lambda x: (x["future"]["7"] if x["future"] else 1))
    imminent = [f for f in forecast if f["future"].get("7", 1) < 0.4][:6]
    return {"n_tracked": len(forecast), "forecast": forecast[:10],
            "imminent_danger": imminent,
            "note": f"{len(imminent)}个考点将在7天内跌入危险区(<0.4)"}


@tool
def plan_spaced_review(sid: str) -> dict:
    """艾宾浩斯复习日历生成：对高危考点(掌握度<0.5)生成5节点复习日历
    (今天/明天/3天后/7天后/15天后)，每个节点分配考点，供 LLM 推理最优复习节奏。

    工具只生成结构化日历，由 LLM 根据考点紧急度+难度自主推理复习顺序与重点。"""
    import datetime as _dt
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
    # 高危考点：掌握度<0.5
    targets = [(cid, m) for cid, m in mastery.items() if m.p < 0.5 and cid in cmap]
    targets.sort(key=lambda x: x[1].p)  # 掌握度升序
    today = _dt.date.today()
    nodes = [{"day": 0, "label": "今天", "date": str(today), "items": []},
             {"day": 1, "label": "明天", "date": str(today + _dt.timedelta(1)), "items": []},
             {"day": 3, "label": "3天后", "date": str(today + _dt.timedelta(3)), "items": []},
             {"day": 7, "label": "7天后", "date": str(today + _dt.timedelta(7)), "items": []},
             {"day": 15, "label": "15天后", "date": str(today + _dt.timedelta(15)), "items": []}]
    # 分配：最危的放今天，次危的放明天，其余循环分配到各节点
    for i, (cid, m) in enumerate(targets[:10]):
        c = cmap[cid]
        node_idx = i % 5
        nodes[node_idx]["items"].append({"cid": cid, "name": c.name,
                                          "mastery": round(m.p, 2),
                                          "tier": c.tier})
    return {"calendar": nodes, "n_targets": len(targets[:10]),
            "horizon_days": 15,
            "note": f"为{len(targets[:10])}个高危考点生成15天5节点复习日历"}


FORGETTING_FORECASTER_SYS = (
    "你是遗忘预测师，基于艾宾浩斯遗忘曲线理论对学生的记忆衰减做预测预警并规划复习节奏。"
    "可调用工具：scan_decay（查各考点当前掌握度×距上次天数、高危遗忘列表——当前快照）、"
    "forecast_retention_curve（外推未来7/15/30天保持率，标注将跌入危险区的考点与时间）、"
    "plan_spaced_review（为高危考点生成5节点艾宾浩斯复习日历）。"
    "流程：先 scan_decay 看当前遗忘快照，再 forecast_retention_curve 外推未来衰减趋势，"
    "最后 plan_spaced_review 取复习日历。"
    "关键：要基于未来衰减预测自己推理复习优先级（而非只列当前高危），"
    "如『X将在5天后跌入危险区，今天复习可延缓衰减曲线』。"
    "输出≤110字预警：将遗忘的考点(名字+几天没碰+何时跌入危险区)+按紧急度排复习顺序+一条建议。")


def run_forgetting_forecaster(sid: str) -> dict:
    """📉 遗忘预测师：LLM 自主推理衰减趋势 + 复习节奏（返回 steps + answer + points）。"""
    model = RouterChatModel(temperature=0.4, max_tokens=480)
    msg = f"学生{sid}的遗忘预测，请基于衰减外推推理复习优先级并出具预警。"
    try:
        steps, answer = _run_react(model, [scan_decay, forecast_retention_curve,
                                           plan_spaced_review],
                                   FORGETTING_FORECASTER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"遗忘预测失败：{e}"}
    decay = scan_decay.invoke({"sid": sid})
    return {"steps": steps, "answer": answer,
            "points": decay["points"], "at_risk": decay["at_risk"]}


# ============================================================
# 7. 📑 课件拆解师
# ============================================================
@tool
def scan_courseware(sid: str) -> dict:
    """读取已上传课件的结构：考点数/章节分布/难度分层/依赖图深度/字符数。"""
    import os
    from .. import storage
    concepts = storage.load_concepts()
    if not concepts:
        return {"n": 0, "msg": "尚未上传课件"}
    # 章节分布
    chap: dict[str, int] = {}
    tier: dict[str, int] = {}
    for c in concepts:
        ch = c.chapter or "未分类"
        chap[ch] = chap.get(ch, 0) + 1
        tier[c.tier] = tier.get(c.tier, 0) + 1
    # 依赖深度
    cmap = {c.cid: c for c in concepts}

    def depth(cid, seen):
        if cid not in cmap or cid in seen:
            return 0
        seen.add(cid)
        return 1 + max((depth(p, seen) for p in cmap[cid].prerequisites if p in cmap), default=0)

    max_depth = max((depth(c.cid, set()) for c in concepts), default=0)
    # 课件文本量
    txt_path = f"data/uploads/{sid}.txt"
    chars = os.path.getsize(txt_path) if os.path.exists(txt_path) else 0
    return {"n_concepts": len(concepts), "chapters": sorted(chap.items(), key=lambda x: -x[1]),
            "tier_mix": tier, "max_depth": max_depth, "chars": chars,
            "sample": [c.name for c in concepts[:8]]}


@tool
def scan_weak_after_upload(sid: str) -> dict:
    """查上传后各考点初始掌握度，定位还没开始的空白区和已学薄弱区。"""
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    untouched = [c.name for c in concepts if c.cid not in mastery or mastery[c.cid].n_correct + mastery[c.cid].n_wrong == 0]
    weak = [{"name": c.name, "p": round(mastery[c.cid].p, 2)}
            for c in concepts if c.cid in mastery and mastery[c.cid].p < 0.4]
    return {"untouched": untouched[:8], "n_untouched": len(untouched),
            "weak": weak[:5]}


# ---- 模块级缓存：聚类结果在一次拆解内复用，避免 ReAct 与过程式采集重复调 LLM ----
_cluster_cache: dict = {}


@tool
def retrieve_course_knowledge(sid: str, query: str) -> dict:
    """混合检索 RAG + GraphRAG：对学生课件原文做『密集向量+稀疏BM25』双策略检索，RRF 融合排序，
    并沿概念先修图扩展一跳邻居（上游先修/下游依赖+掌握度）。
    返回与 query 最相关的课件段落证据 + 知识图谱上下文。

    工具只给检索证据（原文片段+两路得分+图谱结构），不做任何总结——由 LLM 自主解读。
    """
    from ..core.rag_retriever import get_student_retriever, expand_with_graph
    r, meta = get_student_retriever(sid)
    if r is None:
        return meta
    hits = r.search(query, top_k=3)
    if not hits:
        return {"source": meta.get("source"), "n_chunks": meta.get("n_chunks"),
                "msg": "两路检索均无命中", "query": query}
    graph = expand_with_graph(hits, storage.load_concepts(), storage.load_mastery(sid))
    return {"source": meta.get("source"), "n_chunks": meta.get("n_chunks"),
            "query": query,
            "hits": [{"rank": h["rank"], "chunk_id": h["chunk_id"],
                      "text": h["text"], "rrf": h["rrf"],
                      "dense_score": h["dense"]["score"],
                      "bm25": h["sparse"]["bm25"]} for h in hits],
            "graph_context": graph}


@tool
def diagnose_prerequisite_gaps(sid: str) -> dict:
    """前置知识缺口诊断：对比课件考点 + 学生掌握度 + 先修依赖，找出『该学但缺先修』的危险断点。

    危险断点 = 考点本身未掌握(掌握度<0.5) 且 其某条先修掌握度<0.4。
    风险分 = 缺失先修数 × 考试权重，按风险降序排列。揭示出学生『想学X但底层B没补』的结构性断点，
    而非表层薄弱。
    """
    from .. import storage
    concepts = [c for c in storage.load_concepts() if c.noise != "filtered"]
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
    gaps = []
    for c in concepts:
        own = mastery[c.cid].p if c.cid in mastery else 0.3
        if own >= 0.5:
            continue  # 已掌握，跳过
        missing = []
        for p in c.prerequisites:
            if p in cmap:
                pp = mastery[p].p if p in mastery else 0.3
                if pp < 0.4:
                    missing.append({"cid": p, "name": cmap[p].name,
                                    "mastery": round(pp, 2)})
        if missing:
            risk = round(len(missing) * c.exam_weight, 2)
            gaps.append({"cid": c.cid, "name": c.name, "chapter": c.chapter,
                         "own_mastery": round(own, 2), "exam_weight": c.exam_weight,
                         "missing": missing, "risk": risk})
    gaps.sort(key=lambda x: -x["risk"])
    return {"n_gaps": len(gaps), "top_gaps": gaps[:5],
            "total_missing": sum(len(g["missing"]) for g in gaps)}


@tool
def mine_concept_clusters(sid: str) -> dict:
    """语义聚类：不按显性章节，按概念语义把考点聚成隐性主题群，揭示学科深层结构。

    揭示被章节割裂的同族概念（cross_chapter=True）——学生常因章节分隔误以为无关，
    实则同属一个主题群，应打通学习。结果带模块级缓存，一次拆解内复用。
    """
    from .. import storage
    from ..llm import get_llm
    # 缓存命中（同一 sid 的概念图未变则复用）
    cached = _cluster_cache.get(sid)
    if cached:
        return cached
    concepts = [c for c in storage.load_concepts() if c.noise != "filtered"]
    if len(concepts) < 4:
        res = {"n_clusters": 0, "msg": "考点不足4个，无法聚类"}
        _cluster_cache[sid] = res
        return res
    sys = ("你是学科结构挖掘专家。把下列考点按【语义主题】聚类（不按章节），"
           "揭示学科深层结构。同族概念即使分属不同章节也应聚到一起。"
           "聚类数3-6个，每个群给主题名、成员cid、群摘要、是否跨章。"
           "只输出JSON: {\"clusters\":[{\"theme\":\"主题名\",\"members\":[\"cid\"],"
           "\"gist\":\"群摘要≤30字\",\"cross_chapter\":true/false}]}")
    brief = "\n".join(f"- {c.cid}|{c.name}|{c.summary}|章:{c.chapter}" for c in concepts)
    data = get_llm().chat_json(sys, brief, temperature=0.3, max_tokens=600)
    clusters = data.get("clusters", [])
    # 校验成员 cid 有效性
    valid_ids = {c.cid for c in concepts}
    for cl in clusters:
        cl["members"] = [m for m in cl.get("members", []) if m in valid_ids]
    cross = [cl for cl in clusters if cl.get("cross_chapter")]
    res = {"n_clusters": len(clusters), "n_cross_chapter": len(cross),
           "clusters": clusters[:6], "cross_chapter_insight": cross[:3]}
    _cluster_cache[sid] = res
    return res


@tool
def build_cognitive_scaffold(sid: str, cid: str) -> dict:
    """奥苏伯尔(Ausubel)有意义学习理论三件套：为考点生成『先行组织者+类比+边界案例』学习脚手架。

    先行组织者(advance_organizer)：比新知更抽象、更概括的引子，把新概念锚定到学生已有认知结构；
    类比(analogy)：用生活或已学概念解释新概念的同构关系；
    边界案例(boundary_case)：反例或极端情形，划清概念边界防过度泛化。
    结果落盘 scaffold_{sid}.json，供前端与后续复习复用。
    """
    import json, os
    from .. import storage
    from ..llm import get_llm
    cmap = {c.cid: c for c in storage.load_concepts()}
    if cid not in cmap:
        return {"ok": False, "msg": "考点不存在"}
    c = cmap[cid]
    # 读课件原文片段用于反幻觉锚定
    txt = ""
    p = f"data/uploads/{sid}.txt"
    if os.path.exists(p):
        txt = open(p, encoding="utf-8").read()
    sys = ("你是认知心理学专家，运用奥苏伯尔有意义学习理论为大学生构建学习脚手架。"
           f"对考点【{c.name}】({c.summary})生成三件套：\n"
           "1. advance_organizer: 先行组织者——比该概念更抽象更概括的引子(≤45字)，"
           "把新知锚定到学生已有认知结构（如『这其实是XX的特例/推广』）。\n"
           "2. analogy: 类比——用一个生活或已学概念解释新概念的同构关系(≤45字)。\n"
           "3. boundary_case: 边界案例——一个反例或极端情形(≤45字)，划清概念边界。\n"
           "只输出JSON: {\"advance_organizer\":\"\",\"analogy\":\"\",\"boundary_case\":\"\"}")
    data = get_llm().chat_json(sys, f"考点：{c.name}\n定义：{c.summary}\n讲义片段：{txt[:600]}",
                               temperature=0.6, max_tokens=400)
    rec = {"cid": cid, "name": c.name, "advance_organizer": data.get("advance_organizer", ""),
           "analogy": data.get("analogy", ""), "boundary_case": data.get("boundary_case", ""),
           "ts": storage.now_iso()}
    # 落盘（去重替换同 cid 的旧脚手架）
    sp = os.path.join(storage.DATA_DIR, f"scaffold_{sid}.json")
    scaffolds = []
    if os.path.exists(sp):
        try:
            scaffolds = json.load(open(sp, encoding="utf-8"))
        except Exception:
            scaffolds = []
    scaffolds = [s for s in scaffolds if s.get("cid") != cid]
    scaffolds.append(rec)
    json.dump(scaffolds, open(sp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return {"ok": True, "scaffold": rec}


COURSEWARE_DECONSTRUCTOR_SYS = (
    "你是课件拆解师，运用认知心理学为大学生把课件拆成博士级学习蓝图。"
    "可调用工具：scan_courseware（查课件结构：考点数/章节/难度分层/依赖深度）、"
    "scan_weak_after_upload（查空白区和薄弱区）、"
    "diagnose_prerequisite_gaps（查『该学但缺先修』的危险断点+风险分）、"
    "mine_concept_clusters（语义聚类揭示隐性主题群+被章节割裂的同族概念）、"
    "retrieve_course_knowledge（混合检索课件原文段落：密集向量+BM25双路召回）。"
    "流程：先 scan_courseware 了解课件全貌，再 diagnose_prerequisite_gaps 找结构性断点，"
    "mine_concept_clusters 揭示学科深层主题结构，scan_weak_after_upload 看空白区，"
    "需要引用课件原文时用 retrieve_course_knowledge 取证。"
    "输出≤170字学习蓝图：①整体难度与基础/进阶/迁移比例 ②隐性主题结构洞察"
    "（哪些同族概念被章节割裂应打通）③最该优先补的前置缺口（用风险分说话）④推荐学习顺序。"
    "要用学术判断力而非罗列数字。")


def run_courseware_deconstructor(sid: str) -> dict:
    """📑 课件拆解师（博士级）：奥苏伯尔脚手架 + 语义聚类 + 前置缺口诊断 + 学习蓝图。

    返回 steps + answer（ReAct 轨迹与蓝图文本）+ structure（课件结构）+
    prerequisite_gaps（缺口诊断）+ clusters（语义聚类）+ scaffold（最危险考点的奥苏伯尔三件套）。
    """
    # 每次拆解重新聚类（防止换课件后用旧缓存）
    _cluster_cache.pop(sid, None)
    model = RouterChatModel(temperature=0.5, max_tokens=650)
    msg = f"学生{sid}上传了课件，请拆解并出具博士级学习蓝图。"
    try:
        steps, answer = _run_react(
            model, [scan_courseware, scan_weak_after_upload,
                    diagnose_prerequisite_gaps, mine_concept_clusters,
                    retrieve_course_knowledge],
            COURSEWARE_DECONSTRUCTOR_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"课件拆解失败：{e}"}
    # 过程式采集完整结构化数据（_run_react 截断工具结果到180字符，这里取全量）
    structure = scan_courseware.invoke({"sid": sid})
    gaps = diagnose_prerequisite_gaps.invoke({"sid": sid})
    clusters = mine_concept_clusters.invoke({"sid": sid})  # 缓存命中，不重复调 LLM
    # 为最危险考点生成奥苏伯尔认知脚手架（只生成1个最危险的，避免超时与 LLM 过载）
    scaffold = None
    top_gaps = gaps.get("top_gaps") or []
    if top_gaps:
        top_cid = top_gaps[0]["cid"]
        try:
            scaffold = build_cognitive_scaffold.invoke({"sid": sid, "cid": top_cid})
        except Exception as e:
            scaffold = {"ok": False, "msg": f"脚手架生成失败：{e}"}
    return {"steps": steps, "answer": answer, "structure": structure,
            "prerequisite_gaps": gaps, "clusters": clusters, "scaffold": scaffold}


# ============================================================
# 8. 📅 半月报告官
# ============================================================
@tool
def scan_15d(sid: str) -> dict:
    """扫描近 15 天学习历程：每日答题量/正确率走势/掌握度进步/弱点变化。"""
    import datetime as _dt
    from .. import storage
    from .. import mastery as M
    answers = storage.load_answers(sid)
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    now = _dt.date.today()
    cutoff = now - _dt.timedelta(days=15)
    # 每日统计
    daily: dict[str, list] = {}
    for a in answers:
        d = str(a.ts)[:10]
        try:
            if _dt.date.fromisoformat(d) >= cutoff:
                daily.setdefault(d, []).append(a)
        except ValueError:
            continue
    daily_stats = []
    for d in sorted(daily):
        ds = daily[d]
        acc_d = sum(1 for a in ds if a.correct) / len(ds) if ds else 0
        daily_stats.append({"date": d, "n": len(ds), "acc": round(acc_d, 2)})
    # 进步：前7天 vs 后8天正确率
    half_idx = len(answers) // 2
    recent_answers = [a for a in answers if str(a.ts)[:10] >= str(cutoff)]
    if len(recent_answers) >= 4:
        h = len(recent_answers) // 2
        early_acc = sum(1 for a in recent_answers[:h] if a.correct) / max(h, 1)
        late_acc = sum(1 for a in recent_answers[h:] if a.correct) / max(len(recent_answers) - h, 1)
    else:
        early_acc = late_acc = None
    # 掌握度进步（用快照对比）
    snaps = storage.load_snapshots(sid)
    if len(snaps) >= 2:
        prev_m = snaps[-2].mastery
        cur_m = {c.cid: c.p for c in mastery.values()}
        common = set(prev_m) & set(cur_m)
        delta = sum(cur_m[c] - prev_m[c] for c in common) / max(len(common), 1)
    else:
        delta = None
    # 当前最薄弱
    weak = sorted(((c.name, round(mastery[c.cid].p, 2)) for c in concepts
                   if c.cid in mastery), key=lambda x: x[1])[:5]
    return {"daily": daily_stats, "n_days_active": len(daily_stats),
            "n_total": len(recent_answers),
            "early_acc": round(early_acc, 2) if early_acc is not None else None,
            "late_acc": round(late_acc, 2) if late_acc is not None else None,
            "mastery_delta": round(delta, 3) if delta is not None else None,
            "weakest": weak,
            "avg_mastery": round(M.avg_mastery(mastery), 2) if mastery else 0}


@tool
def trace_growth_trajectory(sid: str) -> dict:
    """成长轨迹证据：把15天历程切成3阶段(开周/中周/近周)，每阶段输出
    正确率/连对错峰值/活跃度/薄弱点聚类，并检测成长转折点(突破点/停滞点/下滑点)
    及其可能触发因素，供 LLM 自主推理成长模式而非流水账统计。

    工具只给轨迹证据不下结论——让 LLM 从『开周0.4连错5→第8天0.75连对4→近周0.45停滞』
    等证据自己推理『第8天突破(可能攻克了枢纽考点)但近周停滞(可能遭遇新瓶颈)』的成长叙事。"""
    import datetime as _dt
    from .. import storage
    answers = storage.load_answers(sid)
    now = _dt.date.today()
    cutoff = now - _dt.timedelta(days=15)
    recent = [a for a in answers if str(a.ts)[:10] >= str(cutoff)]
    if len(recent) < 5:
        return {"n": len(recent), "msg": "半月数据不足，无法绘成长轨迹"}
    # 切3阶段（各5天）
    stages = []
    seg_days = 5
    for k in range(3):
        seg_start = cutoff + _dt.timedelta(days=k * seg_days)
        seg_end = seg_start + _dt.timedelta(days=seg_days)
        seg = [a for a in recent
               if seg_start <= _dt.date.fromisoformat(str(a.ts)[:10]) < seg_end]
        if not seg:
            stages.append({"stage": k + 1, "n": 0})
            continue
        acc = sum(1 for a in seg if a.correct) / len(seg)
        bu = cd = cu = 0
        for a in seg:
            if a.correct:
                cu += 1; cd = 0; bu = max(bu, cu)
            else:
                cd += 1; cu = 0
        stages.append({"stage": k + 1, "label": ["开周", "中周", "近周"][k],
                       "n": len(seg), "acc": round(acc, 2),
                       "best_streak": bu, "worst_slump": cd})
    # 转折点检测：相邻阶段正确率差
    accs = [s.get("acc") for s in stages if s.get("n", 0) > 0]
    turning = []
    for i in range(1, len(accs)):
        d = round(accs[i] - accs[i - 1], 2)
        if d > 0.2:
            turning.append({"at_stage": i + 1, "type": "突破", "delta": d})
        elif d < -0.2:
            turning.append({"at_stage": i + 1, "type": "下滑", "delta": d})
    # 停滞检测：相邻阶段差<0.05 且正确率<0.6
    for i in range(1, len(accs)):
        if abs(accs[i] - accs[i - 1]) < 0.05 and accs[i] < 0.6:
            turning.append({"at_stage": i + 1, "type": "停滞", "delta": round(accs[i] - accs[i - 1], 2)})
    # 薄弱点转移：开周薄弱 vs 近周薄弱（是否攻克旧瓶颈、遭遇新瓶颈）
    cmap = {c.cid: c for c in storage.load_concepts()}
    mastery = storage.load_mastery(sid)
    early_cid = {a.cid for a in recent[:len(recent) // 3] if not a.correct}
    late_cid = {a.cid for a in recent[-len(recent) // 3:] if not a.correct}
    conquered = early_cid - late_cid  # 开周错但近周不错=攻克
    new_blocker = late_cid - early_cid  # 近周新错=新瓶颈
    return {"stages": stages, "turning_points": turning,
            "conquered_count": len(conquered),
            "new_blockers": [{"cid": cid, "name": cmap.get(cid, type("x",(),{"name":cid})()).name}
                             for cid in list(new_blocker)[:3]],
            "growth_pattern": ("突破后停滞" if any(t["type"] == "突破" for t in turning)
                               and any(t["type"] == "停滞" for t in turning)
                               else "持续上升" if all(t["type"] == "突破" for t in turning)
                               else "持续下滑" if all(t["type"] == "下滑" for t in turning)
                               else "波动" if len(turning) >= 2 else "平稳")}


HALF_MONTH_REPORTER_SYS = (
    "你是半月报告官，基于15天成长轨迹证据自主推理成长模式并出具有温度的成长叙事。"
    "可调用工具：scan_15d（查15天每日统计/进步走势/掌握度增量/当前薄弱点——统计快照）、"
    "trace_growth_trajectory（把历程切3阶段，给对错峰值/转折点/薄弱转移等轨迹证据）。"
    "流程：先 scan_15d 取统计快照，再 trace_growth_trajectory 取成长轨迹证据，然后综合推理。"
    "关键：不要流水账罗列数字，要基于轨迹证据自己推理成长叙事"
    "（如『第8天突破(攻克枢纽X)但近周停滞(遭遇新瓶颈Y)』），说出转折点与触发因素推测。"
    "输出≤170字成长档案：成长模式判断+转折点叙事+一个亮点+一个待改进+结尾鼓励。")


def run_half_month_reporter(sid: str) -> dict:
    """📅 半月报告官：LLM 自主推理成长叙事 + 报告（返回 steps + answer + daily）。"""
    model = RouterChatModel(temperature=0.6, max_tokens=600)
    msg = f"学生{sid}的15天成长报告，请基于成长轨迹证据推理成长叙事。"
    try:
        steps, answer = _run_react(model, [scan_15d, trace_growth_trajectory],
                                   HALF_MONTH_REPORTER_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"半月报告失败：{e}"}
    return {"steps": steps, "answer": answer, **scan_15d.invoke({"sid": sid}),
            "trajectory": trace_growth_trajectory.invoke({"sid": sid})}
