"""知识图谱因果推断（Causal Inference with do-calculus）。

博士级创新点：不只是先修依赖（相关），而是用 do-calculus 做因果推断——
学不会 A 的真正根因是 B 没掌握（而非表面相关的上游）。
学术前沿：Judea Pearl 因果革命——从相关(seeing)到干预(doing)，
用 do(P(y|do(x))) 估计干预效果，区分因果与伪相关。

工具：
  1. build_causal_graph — 从答题时序构建因果图：哪些考点掌握度变化导致其他变化
  2. do_calculus        — 干预效应估计：若干预B(提升至掌握)，A预期变化多少
                           （do(·) 工程近似：先修链因果路径+路径衰减；无路径→混杂相关）
  3. find_root_cause    — 沿因果图反向溯源真根因（不是先修链的表层相关）
  4. issue_causal_diagnosis — 落盘因果诊断书
"""
from __future__ import annotations
import json
from collections import defaultdict
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel
from ..llm import get_llm

# 模块级缓存：_run_react 截断步骤结果到 180 字，导致完整 JSON 无法回传；
# 工具执行时把完整结果写进缓存，循环后读取
_CAUSAL_CACHE: dict = {}


@tool
def build_causal_graph(sid: str) -> dict:
    """从答题时序构建因果图：基于时间先后的掌握度变化传导。

    方法：对每对考点(A,B)，若学生在答A之后B的掌握度也提升（时序传导），
    且 A 是 B 的先修依赖 → 记因果边 A→B。再统计传导强度。
    区别于先修依赖（静态声明）：这里用的是真实答题时序的传导证据。
    """
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    if not answers or not mastery:
        return {"nodes": [], "edges": [], "msg": "答题记录不足，无法构建因果图"}
    # 按时间排序的答题序列
    sorted_ans = sorted(answers, key=lambda a: a.ts)
    # 记录每个考点首次答对的时间位置
    first_correct_idx = {}
    for i, a in enumerate(sorted_ans):
        if a.correct and a.cid not in first_correct_idx:
            first_correct_idx[a.cid] = i
    # 构建因果边：A 先修于 B 且 A 的首对早于 B 的首对 → A→B（时序传导，强证据）
    edges = []
    edge_keys = set()  # (src,dst) 去重
    for b_cid, b_c in cmap.items():
        for pre_cid in b_c.prerequisites:
            if pre_cid not in cmap:
                continue
            key = (pre_cid, b_cid)
            if pre_cid in first_correct_idx and b_cid in first_correct_idx:
                a_idx = first_correct_idx[pre_cid]
                b_idx = first_correct_idx[b_cid]
                if a_idx < b_idx:
                    # 传导强度：时间间隔反比（越近越强）
                    gap = b_idx - a_idx
                    strength = round(1 / (1 + gap * 0.1), 2)
                    edges.append({"src": pre_cid, "src_name": cmap[pre_cid].name,
                                  "dst": b_cid, "dst_name": b_c.name,
                                  "strength": strength, "time_gap": gap,
                                  "evidence": "时序传导"})
                    edge_keys.add(key)
    # 补充结构因果边：先修关系 + 掌握度缺口（时序不足时用结构证据）
    for b_cid, b_c in cmap.items():
        for pre_cid in b_c.prerequisites:
            if pre_cid not in cmap or (pre_cid, b_cid) in edge_keys:
                continue
            pre_m = mastery.get(pre_cid)
            b_m = mastery.get(b_cid)
            if pre_m and b_m:
                # 缺口越大（先修越弱→下游也弱），因果强度越高
                gap = round((1 - pre_m.p) * (1 - b_m.p), 2)
                if gap > 0.1:
                    edges.append({"src": pre_cid, "src_name": cmap[pre_cid].name,
                                  "dst": b_cid, "dst_name": b_c.name,
                                  "strength": gap, "time_gap": -1,
                                  "evidence": "结构+缺口"})
                    edge_keys.add((pre_cid, b_cid))
    # 找反向因果：先修没掌握但下游掌握了（伪相关，可能是死记硬背）
    spurious = []
    for b_cid, b_c in cmap.items():
        for pre_cid in b_c.prerequisites:
            pre_m = mastery.get(pre_cid)
            b_m = mastery.get(b_cid)
            if pre_m and b_m and pre_m.p < 0.4 and b_m.p > 0.6:
                spurious.append({"cid": b_cid, "name": b_c.name,
                                 "root": pre_cid, "root_name": cmap[pre_cid].name,
                                 "note": f"{b_c.name}掌握但{cmap[pre_cid].name}未掌握→疑似死记硬背"})
    nodes = [{"cid": c.cid, "name": c.name,
              "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None}
             for c in concepts if c.cid in mastery]
    res = {"n_nodes": len(nodes), "n_edges": len(edges),
           "edges": edges[:20], "nodes": nodes[:30],
           "spurious_correlations": spurious[:10]}
    _CAUSAL_CACHE["graph"] = res
    return res


@tool
def do_calculus(sid: str, target_cid: str, intervention_cid: str) -> dict:
    """do-calculus 因果推断：若干预(提升至掌握)intervention_cid，
    target_cid 的掌握度预期变化多少——估计 P(target↑|do(intervention↑))。

    Pearl do(·) 的工程近似：沿先修链找 干预→目标 的因果路径（可达性检验），
    效应 = 干预提升空间 × 路径长度衰减 × 经验传导系数；
    无因果路径 → 仅混杂相关，效应≈0。确定性计算为准，LLM 复核仅作参考不覆盖。
    """
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    mastery = storage.load_mastery(sid)
    t_c = cmap.get(target_cid)
    i_c = cmap.get(intervention_cid)
    if not t_c or not i_c:
        return {"error": "考点不存在"}
    # 沿先修链判断是否有因果路径 intervention → ... → target
    def reachable(start: str, goal: str, visited: set) -> list[str]:
        if start == goal:
            return [start]
        visited.add(start)
        c = cmap.get(start)
        if not c:
            return []
        for nxt in c.prerequisites:
            if nxt not in visited and nxt in cmap:
                path = reachable(nxt, goal, visited)
                if path:
                    return [start] + path
        return []
    # reachable 沿 prerequisites 上溯，返回 [target,...,intervention]；
    # 反转为因果方向 [intervention,...,target]（干预在上游源头）
    path = list(reversed(reachable(target_cid, intervention_cid, set())))
    t_mastery = mastery.get(target_cid)
    i_mastery = mastery.get(intervention_cid)
    t_p = t_mastery.p if t_mastery else 0.3
    i_p = i_mastery.p if i_mastery else 0.3
    if path:
        # 有因果路径：干预效果 = 路径长度衰减 × 干预缺口
        gap = 1.0 - i_p  # 干预提升空间
        decay = 1.0 / (1.0 + 0.3 * (len(path) - 1))
        effect = round(gap * decay * 0.6, 3)  # 0.6 经验传导系数
        causal = True
        relation = f"因果路径: {' → '.join(cmap[x].name for x in path)}"
    else:
        # 无直接因果路径：相关但非因果，干预效果低
        effect = round((1.0 - t_p) * 0.1, 3)
        causal = False
        relation = "无因果路径，仅有弱相关"
    # 确定性计算为准（工具只产出原始证据，结论由上层 LLM 自主推理）；
    # 内部 LLM 复核仅作参考字段 llm_review，不得覆盖数值
    out = {"target": target_cid, "target_name": t_c.name,
           "intervention": intervention_cid, "intervention_name": i_c.name,
           "causal_path": path, "causal": causal,
           "expected_delta": effect, "reasoning": relation,
           "recommend_intervene": causal}
    try:
        llm = get_llm()
        user = (f"考点{target_cid}({t_c.name}，摘要:{t_c.summary})，当前掌握度{t_p:.2f}\n"
                f"干预考点{intervention_cid}({i_c.name}，摘要:{i_c.summary})，当前掌握度{i_p:.2f}\n"
                f"因果路径分析: {relation}\n"
                f"估算干预{intervention_cid}后{target_cid}的掌握度预期变化。")
        sys = ("你是因果推断复核员，基于工具给出的确定性路径分析复核干预效果。"
               "区分因果效应(有路径)与混杂相关(无路径)。"
               '只输出JSON：{"causal":true|false,"expected_delta":-0.3~0.5,'
               '"reasoning":"复核依据(≤50字)","recommend_intervene":true|false}。')
        res = llm.chat_json(sys, user, temperature=0.3, max_tokens=250)
        out["llm_review"] = {"causal": bool(res.get("causal", causal)),
                             "expected_delta": float(res.get("expected_delta", effect)),
                             "reasoning": res.get("reasoning", ""),
                             "recommend_intervene": bool(res.get("recommend_intervene", causal))}
    except Exception as e:
        out["note"] = f"LLM复核不可用:{e}"
    return out


@tool
def find_root_cause(sid: str, target_cid: str) -> dict:
    """沿因果图反向溯源真根因：不是先修链的表层相关，
    而是掌握度最低、传导最强的上游根节点。

    返回根因考点、因果证据链、根因类型(知识缺口/概念混淆/从未接触)。
    """
    graph = build_causal_graph.invoke({"sid": sid})
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    mastery = storage.load_mastery(sid)
    edges = graph.get("edges", [])
    # 反向溯源：沿入边 BFS 收集全部上游祖先（路径强度 = 各跳边强度连乘）；
    # 无入边的节点用先修链兜底（强度与 build_causal_graph 结构边同口径：两端缺口乘积）
    parent_of: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in edges:
        parent_of[e["dst"]].append((e["src"], e.get("strength", 0.3)))

    def parents_of(cid: str) -> list[tuple[str, float]]:
        ps = parent_of.get(cid)
        if ps:
            return ps
        c = cmap.get(cid)
        if not c:
            return []
        m_dst = mastery.get(cid)
        p_dst = m_dst.p if m_dst else 0.3
        return [(p, round((1 - (mastery[p].p if p in mastery else 0.3)) * (1 - p_dst), 3))
                for p in c.prerequisites if p in cmap]

    best_strength: dict[str, float] = {}
    queue = [(target_cid, 1.0)]
    seen = {target_cid}
    while queue:
        cur, acc = queue.pop(0)
        for p, s in parents_of(cur):
            if p not in seen:
                seen.add(p)
                best_strength[p] = acc * s  # 连乘：越深衰减越大（与 do_calculus 同向）
                queue.append((p, acc * s))
    if not best_strength:
        return {"target": target_cid, "root_cause": None,
                "msg": "无上游因果节点，根因可能是该考点本身未掌握或从未接触"}
    # 选掌握度最低 × 传导最强的祖先（缺口 × 路径传导强度）
    best = None
    for cid, strength in best_strength.items():
        m = mastery.get(cid)
        p = m.p if m else 0.3
        score = (1 - p) * strength
        if not best or score > best["score"]:
            best = {"cid": cid, "name": cmap[cid].name if cid in cmap else cid,
                    "mastery": round(p, 2), "strength": round(strength, 3),
                    "score": round(score, 3)}
    # 判断根因类型
    root_m = mastery.get(best["cid"]) if best else None
    root_p = root_m.p if root_m else None
    if root_p is None:
        cause_type = "从未接触"
    elif root_p < 0.2:
        cause_type = "知识缺口"
    elif root_p < 0.4:
        cause_type = "概念混淆"
    else:
        cause_type = "技能薄弱"
    ranked = sorted(best_strength.items(),
                    key=lambda kv: -(1 - (mastery[kv[0]].p if kv[0] in mastery else 0.3)) * kv[1])
    res = {"target": target_cid, "target_name": cmap[target_cid].name if target_cid in cmap else target_cid,
           "root_cause": best, "cause_type": cause_type,
           "evidence_chain": [{"node": cmap[c].name if c in cmap else c,
                               "strength": round(s, 3)} for c, s in ranked[:5]]}
    _CAUSAL_CACHE["root_cause"] = res
    return res


@tool
def issue_causal_diagnosis(sid: str, target: str, root_cause: str,
                           evidence: str, intervention: str) -> dict:
    """落盘因果诊断书。

    target=主诉考点，root_cause=真根因考点，evidence=因果证据(≤60字)，
    intervention=建议干预(≤40字)。
    """
    import os
    from .. import storage
    rec = {"sid": sid, "target": target, "root_cause": root_cause,
           "evidence": evidence, "intervention": intervention,
           "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"causal_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "diagnosis": rec}


CAUSAL_SYS = (
    "你是因果推断诊断官，运用 Judea Pearl do-calculus 框架诊断学生学不会的真根因。"
    "可调用工具：build_causal_graph（构建因果图）、do_calculus（干预效果推断）、"
    "find_root_cause（反向溯源真根因）、issue_causal_diagnosis（落盘诊断书，唯一入口）。"
    "流程：先 build_causal_graph 建因果图，再 find_root_cause 溯源根因，"
    "用 do_calculus 验证干预根因能否改善目标，最后 issue_causal_diagnosis 落盘。"
    "再用≤100字向学生说明：你以为是A学不会，真根因其实是B，干预B才能治本。"
)

CAUSAL_TOOLS = [build_causal_graph, do_calculus, find_root_cause, issue_causal_diagnosis]


def run_causal_diagnosis(sid: str, target_cid: str = "") -> dict:
    """因果推断诊断：找真根因 + 干预建议。

    返回 {steps, answer, causal_graph, root_cause, verdict}。
    """
    _CAUSAL_CACHE.clear()
    model = RouterChatModel(temperature=0.4, max_tokens=700)
    if not target_cid:
        # 默认取最薄弱考点
        from .diagnosis_agents import pull_mastery_map
        weakest = pull_mastery_map.invoke({"sid": sid})["weakest"]
        target_cid = weakest[0]["cid"] if weakest else ""
        target_name = weakest[0]["name"] if weakest else "未知"
    else:
        from .. import storage
        cmap = {c.cid: c for c in storage.load_concepts()}
        target_name = cmap.get(target_cid, type("X", (), {"name": target_cid})()).name
    msg = (f"学生{sid}主诉『{target_name}』学不会。"
           "请用 do-calculus 因果推断找真根因并出诊断书。")
    try:
        steps, answer = _run_react(model, CAUSAL_TOOLS, CAUSAL_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"因果推断引擎故障：{e}",
                "causal_graph": _CAUSAL_CACHE.get("graph", {}),
                "root_cause": (_CAUSAL_CACHE.get("root_cause", {}) or {}).get("root_cause")}
    causal_graph = _CAUSAL_CACHE.get("graph", {})
    rc_full = _CAUSAL_CACHE.get("root_cause", {}) or {}
    root_cause = rc_full.get("root_cause")
    if not root_cause:
        # 兜底：从步骤文本解析
        for s in steps:
            r = s.get("result")
            if not r:
                continue
            try:
                d = json.loads(r)
            except Exception:
                continue
            if s["tool"] == "find_root_cause":
                root_cause = d.get("root_cause")
    return {"steps": steps, "answer": answer,
            "causal_graph": causal_graph, "root_cause": root_cause,
            "sid": sid, "target": target_cid}
