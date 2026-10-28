"""诊断双智能体（LangChain ReAct）：🩺 认知断层 CT 室 + 🧠 认知法医。
把规则流水线升级为真正的 create_react_agent：模型自主决定查什么证据、
下什么结论，处方/鉴定必须经 prescribe_ct / issue_findings 落盘才生效。
共用 5 个真实数据 @tool（掌握度图谱/先修链扫描/答题时序/易混淆对比/到期复习）。
"""
from __future__ import annotations
from langchain_core.tools import tool
from ..lc_model import RouterChatModel


@tool
def pull_mastery_map(sid: str) -> dict:
    """调取掌握度图谱：全部考点按掌握度从低到高排名（含未作答标记）。"""
    from .. import storage
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    rows = sorted(({"cid": c.cid, "name": c.name,
                    "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None}
                   for c in concepts),
                  key=lambda x: (x["mastery"] if x["mastery"] is not None else 2))
    return {"weakest": rows[:10], "strongest": rows[-3:]}


@tool
def trace_chain(sid: str, cid: str) -> dict:
    """沿先修依赖链完整扫描该考点及其全部上游（含分支）：每环节返回掌握度，定位断在哪一层。"""
    from .. import storage
    cmap = {c.cid: c for c in storage.load_concepts()}
    mastery = storage.load_mastery(sid)
    if cid not in cmap:
        return {"error": "考点不存在"}
    layers, seen, queue = [], set(), [cid]
    while queue:
        cur = queue.pop(0)
        if cur in seen or cur not in cmap:
            continue
        seen.add(cur)
        c = cmap[cur]
        layers.append({"cid": c.cid, "name": c.name, "depth": 0,
                       "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None})
        queue.extend(c.prerequisites)
    return {"chain": layers, "n_upstream": len(layers) - 1}


@tool
def predict_chain_failure_risk(sid: str, cid: str) -> dict:
    """先修链风险传染预测：沿先修链基于上游掌握度梯度，预测目标考点的失败概率
    及风险传染路径——哪个上游薄弱将传染到目标，哪层是风险源。

    从『查当前链状态』升级为『预测哪里将断裂』——供 LLM 自主推理预防性补强哪层，
    而非等到失败再诊断。"""
    from .. import storage
    cmap = {c.cid: c for c in storage.load_concepts()}
    mastery = storage.load_mastery(sid)
    if cid not in cmap:
        return {"error": "考点不存在"}
    # BFS 沿先修链采集每层掌握度
    chain, seen, queue = [], set(), [cid]
    while queue:
        cur = queue.pop(0)
        if cur in seen or cur not in cmap:
            continue
        seen.add(cur)
        c = cmap[cur]
        p = mastery[cur].p if cur in mastery else None
        chain.append({"cid": cur, "name": c.name, "mastery": round(p, 2) if p is not None else None,
                      "is_target": cur == cid})
        queue.extend(c.prerequisites)
    # 风险传染：上游掌握度低→传染给下游。每层风险=自身低 + 上游传染
    risk_chain = []
    upstream_below = 0
    for i, node in enumerate(chain):
        p = node["mastery"]
        # 该层风险：掌握度越低风险越高(0-1)，加上上游低掌握的传染加成
        own_risk = round(1 - (p if p is not None else 0.5), 2)
        if p is not None and p < 0.4 and not node["is_target"]:
            upstream_below += 1
        # 传染加成：每个上游低掌握节点给目标+0.1传染风险
        contagion = round(min(upstream_below * 0.12, 0.5), 2) if node["is_target"] else 0
        total_risk = round(min(own_risk + contagion, 1.0), 2)
        risk_chain.append({**node, "own_risk": own_risk,
                           "contagion_risk": contagion if node["is_target"] else None,
                           "total_risk": total_risk})
    # 风险源：上游掌握度最低的环节
    upstream = [n for n in risk_chain if not n["is_target"] and n["mastery"] is not None]
    risk_source = min(upstream, key=lambda x: x["mastery"]) if upstream else None
    target_node = next((n for n in risk_chain if n["is_target"]), None)
    target_risk = target_node["total_risk"] if target_node else None
    verdict = ("高危(将断裂)" if target_risk and target_risk >= 0.7
               else "中危(需预防)" if target_risk and target_risk >= 0.4
               else "低危(稳定)")
    return {"target": cid, "target_name": cmap[cid].name,
            "target_failure_risk": target_risk,
            "risk_source": {"cid": risk_source["cid"], "name": risk_source["name"],
                            "mastery": risk_source["mastery"]}
                           if risk_source else None,
            "n_upstream_below_threshold": upstream_below,
            "risk_chain": risk_chain[:8],
            "verdict": verdict,
            "evidence_brief": (f"目标{cmap[cid].name}失败风险{target_risk},"
                               f"风险源{'无' if not risk_source else risk_source['name']+'(掌握'+str(risk_source['mastery'])+')'},"
                               f"{upstream_below}个上游低于阈值")}


@tool
def locate_root_cause_node(sid: str, cid: str) -> dict:
    """根因定位：沿先修链递归向最上游追溯，找到最底层未掌握的根因考点
    （掌握度<0.4 的最上游节点）——链的真正源头，而非表层断点。

    从『列整条链』升级为『定位链的源头病灶』——供 LLM 推理该补的真正起点，
    而非治标不治本补中间层。"""
    from .. import storage
    cmap = {c.cid: c for c in storage.load_concepts()}
    mastery = storage.load_mastery(sid)
    if cid not in cmap:
        return {"error": "考点不存在"}
    # 递归找最上游未掌握节点
    roots = []
    visited = set()

    def trace_up(cur, path):
        if cur in visited or cur not in cmap:
            return
        visited.add(cur)
        c = cmap[cur]
        p = mastery[cur].p if cur in mastery else None
        path.append({"cid": cur, "name": c.name, "mastery": round(p, 2) if p is not None else None})
        # 是根因候选：未掌握(<0.4)且无前置（最上游）或前置都掌握
        is_below = p is not None and p < 0.4
        has_untracked_prereq = any(pp in cmap and pp not in visited for pp in c.prerequisites)
        if is_below and (not c.prerequisites or all(
                pp not in mastery or mastery[pp].p >= 0.4 for pp in c.prerequisites if pp in cmap)):
            roots.append({"cid": cur, "name": c.name, "mastery": round(p, 2),
                          "depth_from_target": len(path) - 1})
        for pp in c.prerequisites:
            trace_up(pp, path[:])

    trace_up(cid, [])
    # 选最上游（depth最大）的根因
    roots.sort(key=lambda x: -x["depth_from_target"])
    root = roots[0] if roots else None
    return {"target": cid, "target_name": cmap[cid].name,
            "root_cause_node": root,
            "all_candidates": roots[:4],
            "is_target_itself_root": root and root["cid"] == cid,
            "note": (f"根因病灶:{root['name']}(距目标{root['depth_from_target']}层,掌握{root['mastery']})"
                     if root else "链上无未掌握节点,根因可能在概念混淆或技能,非先修链断供")}


@tool
def error_timeline(sid: str, cid: str) -> dict:
    """调取该考点答题时序：对错序列、最近错题原文、错误是否发生在上游之后。"""
    from .. import storage
    answers = [a for a in storage.load_answers(sid) if a.cid == cid]
    if not answers:
        return {"n": 0, "msg": "无答题记录（可能从未接触该考点）"}
    marks = "".join("✔" if a.correct else "✘" for a in answers)
    last_wrong = next((a for a in reversed(answers) if not a.correct), None)
    return {"n": len(answers), "timeline": marks,
            "last_wrong_q": (last_wrong.question or "")[:60] if last_wrong else "",
            "last_wrong_ref": (last_wrong.reference or "")[:60] if last_wrong else ""}


@tool
def confusion_pair(cid: str) -> dict:
    """查易混淆概念：同章节、同难度的相邻考点及其摘要（用于判断是不是概念混淆）。"""
    from .. import storage
    cmap = {c.cid: c for c in storage.load_concepts()}
    cur = cmap.get(cid)
    if not cur:
        return {"pairs": []}
    sibs = [c for c in cmap.values() if c.cid != cid and (c.chapter == cur.chapter or c.tier == cur.tier)]
    return {"pairs": [{"cid": c.cid, "name": c.name, "summary": c.summary[:50]}
                      for c in sibs[:4]]}


@tool
def due_review_count(sid: str) -> dict:
    """查今日到期复习项数量与最紧急的考点名。"""
    from .. import storage
    from ..core.spaced_repetition import due_review
    due = due_review(storage.load_mastery(sid))
    return {"n_due": len(due), "items": [str(x) for x in due[:5]]}


@tool
def identify_misdiagnosis_pattern(sid: str) -> dict:
    """误诊模式识别：检测学生『自评掌握度高但实际反复错』的过度自信误判区——
    这些是隐性知识盲区（自以为会实则不会），比显性薄弱更危险。

    从『查掌握度图谱』升级为『识别认知盲区』——供 LLM 自主推理哪些考点
    需要打破学生的虚假自信，而非只补显性低分。"""
    from .. import storage
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in storage.load_concepts()}
    if not answers or not mastery:
        return {"n": 0, "msg": "数据不足无法识别误诊模式"}
    # 按考点聚合答题记录，算实际正确率
    by_cid: dict[str, list] = {}
    for a in answers:
        by_cid.setdefault(a.cid, []).append(a.correct)
    misdiagnosed = []
    for cid, m in mastery.items():
        if cid not in cmap or cid not in by_cid:
            continue
        actual_acc = sum(1 for c in by_cid[cid] if c) / len(by_cid[cid])
        # 误诊：自评掌握度>=0.6 但实际正确率<0.4（高自信低实绩）
        if m.p >= 0.6 and actual_acc < 0.4 and len(by_cid[cid]) >= 2:
            misdiagnosed.append({"cid": cid, "name": cmap[cid].name,
                                 "self_assessed": round(m.p, 2),
                                 "actual_acc": round(actual_acc, 2),
                                 "gap": round(m.p - actual_acc, 2),
                                 "n_attempts": len(by_cid[cid])})
    misdiagnosed.sort(key=lambda x: -x["gap"])
    # 反向误诊：自评低但实际全对（过度谨慎，可能浪费复习时间）
    under_confident = []
    for cid, m in mastery.items():
        if cid not in cmap or cid not in by_cid:
            continue
        actual_acc = sum(1 for c in by_cid[cid] if c) / len(by_cid[cid])
        if m.p < 0.3 and actual_acc >= 0.8 and len(by_cid[cid]) >= 2:
            under_confident.append({"cid": cid, "name": cmap[cid].name,
                                     "self_assessed": round(m.p, 2),
                                     "actual_acc": round(actual_acc, 2)})
    return {"n_misdiagnosed": len(misdiagnosed),
            "misdiagnosed_zones": misdiagnosed[:5],
            "n_under_confident": len(under_confident),
            "under_confident_zones": under_confident[:3],
            "pattern": ("过度自信误判" if len(misdiagnosed) > len(under_confident)
                        else "过度谨慎" if under_confident else "校准良好"),
            "evidence_brief": (f"{len(misdiagnosed)}个隐性盲区(自评高实绩低),"
                               f"{len(under_confident)}个过度谨慎区;模式:{'过度自信误判' if len(misdiagnosed) > len(under_confident) else '过度谨慎' if under_confident else '校准良好'}")}


@tool
def audit_cross_confusion(sid: str) -> dict:
    """跨概念混淆审计：检测学生是否把跨章节/跨tier但功能相近的概念搞混
    （如把『认证』和『授权』『签名』和『加密』混淆）——超越相邻易混淆，
    找到跨域的概念错配模式。

    从『查相邻易混淆』升级为『审计跨域混淆模式』——供 LLM 自主推理
    学生是否存在系统性的概念边界模糊，而非偶发混淆。"""
    from .. import storage
    concepts = storage.load_concepts()
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in concepts}
    if not answers:
        return {"n": 0, "msg": "无答题数据"}
    # 找功能相近但跨章节的概念对（名字共享字符/词根，但章节不同）
    pairs = []
    c_list = list(cmap.values())
    for i, a in enumerate(c_list):
        for b in c_list[i + 1:]:
            if a.chapter and b.chapter and a.chapter != b.chapter:
                # 名字共享>=2字符（疑似功能相近但不同概念）
                common = set(a.name) & set(b.name)
                if len(common) >= 2 and len(a.name) >= 2 and len(b.name) >= 2:
                    # 检查是否两个都低掌握且学生在此两考点答错模式相似
                    a_p = mastery[a.cid].p if a.cid in mastery else None
                    b_p = mastery[b.cid].p if b.cid in mastery else None
                    a_wrong = sum(1 for x in answers if x.cid == a.cid and not x.correct)
                    b_wrong = sum(1 for x in answers if x.cid == b.cid and not x.correct)
                    # 混淆嫌疑：两个都答错过 且 至少一个低掌握
                    confusion_suspect = (a_wrong > 0 and b_wrong > 0 and
                                          ((a_p is not None and a_p < 0.5) or
                                           (b_p is not None and b_p < 0.5)))
                    if confusion_suspect:
                        pairs.append({"a": {"cid": a.cid, "name": a.name,
                                             "chapter": a.chapter, "mastery": round(a_p, 2) if a_p else None,
                                             "n_wrong": a_wrong},
                                      "b": {"cid": b.cid, "name": b.name,
                                            "chapter": b.chapter, "mastery": round(b_p, 2) if b_p else None,
                                            "n_wrong": b_wrong},
                                      "shared_chars": "".join(common)})
    pairs.sort(key=lambda x: -(x["a"]["n_wrong"] + x["b"]["n_wrong"]))
    return {"n_confusion_pairs": len(pairs),
            "cross_confusion_pairs": pairs[:4],
            "pattern": (f"跨章节概念边界模糊({len(pairs)}对)" if len(pairs) >= 2
                        else "偶发混淆" if pairs else "无跨域混淆"),
            "evidence_brief": (f"{len(pairs)}对跨章节混淆嫌疑;" +
                               ";".join(f"{p['a']['name']}↔{p['b']['name']}" for p in pairs[:2]) or "无")}


TOOLS = [pull_mastery_map, trace_chain, error_timeline, confusion_pair, due_review_count]


def _strip_markdown(text: str) -> str:
    """轻量 markdown 清洗：LLM 爱用表格/加粗/标题，前端是 innerHTML 直接渲染，
    统一压成短段落纯文本。"""
    import re
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|"):  # 表格行 → 顿号连接
            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            if cells and not set(cells[0]) <= {"-", " ", ":"}:
                lines.append(" · ".join(cells))
        else:
            s = re.sub(r"^#{1,6}\s*", "", s)
            s = s.replace("**", "").replace("__", "")
            if s:
                lines.append(s)
    return "\n".join(lines)


def _run_react(model, tools, sys_prompt, user_msg) -> tuple[list, str]:
    """通用 ReAct 循环：返回 (steps, answer)。
    兜底：模型偶发把工具意图当纯文本输出（无 tool_calls）导致循环提前结束，
    此时用已采集的证据强制直答一次，保证面板总能拿到像样的结论。"""
    import re
    from langgraph.prebuilt import create_react_agent
    try:
        agent = create_react_agent(model, tools, prompt=sys_prompt)
    except TypeError:
        agent = create_react_agent(model, tools, state_modifier=sys_prompt)
    res = agent.invoke({"messages": [("user", user_msg)]})
    steps, answer = [], ""
    for m in res["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                steps.append({"tool": tc["name"], "args": tc["args"]})
        elif type(m).__name__ == "ToolMessage":
            if steps:
                steps[-1]["result"] = str(m.content)[:180]
        elif type(m).__name__ == "AIMessage" and m.content:
            answer = str(m.content)
    # 截断特征：结论其实是一条工具意图文本（"我决定调用工具: xxx(...)"）
    truncated = (not answer) or re.search(r"调用工具\s*[:：]", answer)
    if truncated:
        facts = "；".join(f"[{s['tool']}] {s.get('result') or ''}" for s in steps if s.get("result")) or "（无工具结果）"
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            final = model.invoke([SystemMessage(content=f"{sys_prompt}\n证据采集已结束，禁止再提调用工具，直接输出最终结论（纯文本短段落）。"),
                                  HumanMessage(content=f"{user_msg}\n\n已取证结果：{facts}")])
            answer = str(final.content)
        except Exception:
            answer = steps[-1]["result"] if steps and steps[-1].get("result") else (answer or "本次结论生成失败")
    return steps, _strip_markdown(answer)


# ---------- 🩺 认知断层 CT 室 ----------
CT_SYS = ("你是认知断层 CT 室的主诊医生，从『查链状态』升级为『预测风险传染+定位源头病灶』"
          "的预防性深度诊断。学生主诉某个考点学不会。可调用工具："
          "trace_chain（沿先修链扫描每层掌握度——查现状）、"
          "predict_chain_failure_risk（预测目标失败概率+风险传染路径+风险源——预测将断裂）、"
          "locate_root_cause_node（沿链追溯最底层未掌握的根因考点——定位源头病灶而非表层断点）、"
          "error_timeline（查答题时序）、confusion_pair（查易混淆概念）、"
          "prescribe_ct（开具诊断处方，唯一生效入口）。"
          "流程：先 trace_chain 看链全貌，必须 predict_chain_failure_risk 预测风险传染，"
          "必须 locate_root_cause_node 定位真正源头病灶（别把表层断点当病灶），"
          "必要时查时序/混淆判断病因类型，最后必须 prescribe_ct 开处方。"
          "关键：处方里的 break_point 应是 locate_root_cause_node 找到的根因（治本），"
          "cause 要结合风险预测判断（风险传染型/源头缺失型/概念混淆型）。"
          "再用≤90字向学生说明：根因病灶在哪（不是表层断点）、预测的风险、先补什么源头。")


@tool
def prescribe_ct(sid: str, target: str, break_point: str, cause: str, plan: str) -> dict:
    """落盘 CT 诊断处方。target=学生主诉考点，break_point=真断点考点（应为locate_root_cause_node找到的根因），
    cause=断层类型（风险传染型/源头缺失型/概念混淆型/技能缺失/从未接触），plan=修复方案(一句话)。"""
    import json, os
    from .. import storage
    rec = {"target": target, "break_point": break_point, "cause": cause,
           "plan": plan, "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"ct_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "prescription": rec}


CT_TOOL_LIST = TOOLS + [predict_chain_failure_risk, locate_root_cause_node, prescribe_ct]


def run_ct(sid: str, target: str = "") -> dict:
    """断层 CT：对主诉考点（默认取最薄弱）扫描真断点并开方。"""
    model = RouterChatModel(temperature=0.4, max_tokens=520)
    target_cid = ""
    if not target:
        weakest = pull_mastery_map.invoke({"sid": sid})["weakest"]
        if weakest:
            target_cid = weakest[0].get("cid", "")
            target = weakest[0]["name"]
    # 同时给 LLM cid 和名字：trace_chain/predict/locate 需要 cid 参数
    cid_hint = f"（cid={target_cid}）" if target_cid else "（用 trace_chain 等工具时需先从 pull_mastery_map 取该考点 cid）"
    msg = (f"学生{sid}主诉『{target}』{cid_hint}学不会。请 CT 扫描先修链，"
           "predict 预测风险传染，locate 定位源头病灶，开出处方并给出诊断说明。"
           f"调用 trace_chain/predict_chain_failure_risk/locate_root_cause_node 时 cid 参数={target_cid}。")
    try:
        steps, answer = _run_react(model, CT_TOOL_LIST, CT_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"CT 室设备故障：{e}", "target": target}
    return {"steps": steps, "answer": answer, "target": target}


# ---------- 🧠 认知法医（AI 深度诊断） ----------
FORENSIC_SYS = ("你是认知法医，从『查数据快照』升级为『识别隐性盲区+审计跨域混淆』"
                "的深度尸检。对学生全部学习数据进行全景尸检。可调用工具："
                "pull_mastery_map（掌握度图谱——显性薄弱）、"
                "identify_misdiagnosis_pattern（识别『自评高实绩低』的隐性盲区——过度自信误判区）、"
                "audit_cross_confusion（审计跨章节概念边界模糊模式——系统性混淆而非偶发）、"
                "error_timeline（答题时序）、confusion_pair（相邻易混淆）、"
                "due_review_count（到期复习）、issue_findings（出具鉴定书，唯一生效入口）。"
                "流程：先 pull_mastery_map 看显性薄弱，必须 identify_misdiagnosis_pattern 挖隐性盲区"
                "（自以为会实则不会的，比显性薄弱更危险），"
                "必须 audit_cross_confusion 审计跨域混淆模式，"
                "对最致命的2-3个考点查时序/混淆定死因，"
                "最后必须 issue_findings 出具鉴定书（每条含 cid/cause/evidence/advice）。"
                "关键：死因要区分隐性盲区(过度自信型)/跨域混淆/显性缺失，证据要来自误诊识别或混淆审计。"
                "再输出≤110字总鉴定意见（法医口吻，客观但关怀，点出最危险的隐性盲区）。")


@tool
def issue_findings(sid: str, findings: str) -> dict:
    """出具法医鉴定书。findings 为 JSON 数组字符串，每项：
    {"cid":"考点id","name":"考点名","cause":"死因(过度自信型隐性盲区/跨域混淆/显性缺失/审题习惯/记忆模糊)","evidence":"证据(≤30字)","advice":"建议(≤30字)"}。
    至少 1 条、至多 3 条。"""
    import json, os
    from .. import storage
    try:
        arr = json.loads(findings)
        if not isinstance(arr, list) or not arr:
            raise ValueError
    except Exception:
        return {"error": "findings 必须是非空 JSON 数组"}
    arr = arr[:3]
    rec = {"findings": arr, "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"forensic_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "n_findings": len(arr)}


FORENSIC_TOOL_LIST = TOOLS + [identify_misdiagnosis_pattern, audit_cross_confusion, issue_findings]


def run_forensic(sid: str) -> dict:
    """认知法医全景尸检：返回 {steps, answer}。"""
    model = RouterChatModel(temperature=0.4, max_tokens=600)
    msg = f"请对学生{sid}的全部学习数据做法医尸检，出具鉴定书。"
    try:
        steps, answer = _run_react(model, FORENSIC_TOOL_LIST, FORENSIC_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"法医失联：{e}"}
    return {"steps": steps, "answer": answer}
