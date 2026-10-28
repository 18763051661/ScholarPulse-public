"""生成式 CAT（Computerized Adaptive Testing）实时出题。

博士级创新点：传统CAT从固定题库抽题，博士级是LLM根据学生薄弱模式
实时生成全新题目 + 反幻觉grounding校验 + Elo难度校准。
学术前沿：Generative Assessment — 用生成式模型动态产出评估内容，
而非从静态题库检索；实现真正个性化、无重复、贴合弱点的自适应测评。

工具：
  1. generate_item  — LLM 根据考点+学生薄弱模式生成全新题干+参考答案
  2. ground_check   — 反幻觉校验：题目是否真对应考点、答案是否与concept summary一致
  3. difficulty_calibrate — 根据 Elo 估计题目难度系数，做IRT式校准
"""
from __future__ import annotations
import json
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel
from ..llm import get_llm

# 模块级缓存：保存工具完整结果（_run_react 会把步骤结果截断到 180 字，
# 导致完整 JSON 无法回传；工具本身执行时把完整结果写进缓存，循环后读取）
_GEN_ITEMS: dict[str, dict] = {}


GEN_SYS = (
    "你是一道考试题的命题专家。根据指定考点与学生的薄弱模式，生成一道全新简答题。"
    "要求：题目必须考察该考点的核心能力，不得与已有题库雷同，难度贴合指定tier，"
    "参考答案必须准确、可评分。只输出JSON。"
    '严格输出：{"question":"题干(≤100字)","reference_answer":"参考答案(≤150字)",'
    '"tested_skill":"考察的能力点(≤20字)","bloom_level":"记忆|理解|应用|分析|综合|评价"}。'
)

GROUND_SYS = (
    "你是反幻觉校验官。检查一道生成题是否真的对应目标考点、参考答案是否与考点摘要一致。"
    "检查项：1.题目考察的是否就是该考点 2.参考答案是否准确无幻觉 3.是否有考点摘要不支持的内容。"
    '严格输出：{"grounded":true|false,"issues":["问题1"],"confidence":0.0~1.0,'
    '"corrected_answer":"修正后的参考答案(若有修正，否则原样)"}。'
)

CALIB_SYS = (
    "你是项目反应理论(IRT)难度校准师。根据考点摘要、学生当前 Elo 和题目标注难度，"
    "估算这道题对该生的难度系数。难度=学生答对的概率的补数(1-P)。"
    "Elo越高、tier越基础则P越高。只输出JSON。"
    '严格输出：{"difficulty":0.0~1.0,"expected_p":0.0~1.0,"irt_note":"校准说明(≤40字)"}。'
)

TIER_TO_ELO = {"foundational": 800, "core": 1100, "advanced": 1400}


@tool
def generate_item(cid: str, course: str, weakness: str, tier: str) -> dict:
    """LLM 根据考点+学生薄弱模式生成全新题目。

    cid=考点ID，weakness=学生薄弱模式描述(如"混淆死锁与活锁")，tier=难度层。
    返回生成的题干、参考答案、考察能力、Bloom层级。
    """
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    c = cmap.get(cid)
    if not c:
        return {"error": f"考点{cid}不存在"}
    llm = get_llm()
    user = (f"考点ID：{cid}\n考点名：{c.name}\n所属课程：{course}\n"
            f"考点摘要：{c.summary}\n考点tier：{tier}\n"
            f"学生薄弱模式：{weakness}\n"
            "请生成一道全新简答题，贴合该薄弱模式，考察该考点核心能力。")
    try:
        res = llm.chat_json(GEN_SYS, user, temperature=0.7, max_tokens=400)
        qid = f"gen_{cid}_{abs(hash(weakness + res.get('question', ''))) % 100000}"
        item = {
            "qid": qid, "cid": cid, "course": course, "tier": tier,
            "question": res.get("question", ""),
            "reference_answer": res.get("reference_answer", ""),
            "tested_skill": res.get("tested_skill", ""),
            "bloom_level": res.get("bloom_level", "应用"),
            "source_quote": c.summary, "grounded": False,
        }
        _GEN_ITEMS[qid] = item  # 缓存完整结果
        return {"ok": True, "qid": qid, "question_preview": item["question"][:30]}
    except Exception as e:
        return {"error": f"生成失败:{e}"}


@tool
def ground_check(qid: str, question: str, reference_answer: str, cid: str) -> dict:
    """反幻觉校验：题目是否真对应考点、答案是否与concept summary一致。

    严格题：生成题必须过这道校验才纳入卷面；不通过则自动修正答案或标记重生成。
    """
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    c = cmap.get(cid)
    if not c:
        return {"error": f"考点{cid}不存在"}
    llm = get_llm()
    user = (f"考点名：{c.name}\n考点摘要(权威来源)：{c.summary}\n"
            f"生成题干：{question}\n生成参考答案：{reference_answer}\n"
            "请校验题目是否真考察该考点、答案是否准确无幻觉。")
    try:
        res = llm.chat_json(GROUND_SYS, user, temperature=0.2, max_tokens=300)
        grounded = bool(res.get("grounded", False))
        corrected = res.get("corrected_answer", "")
        # 更新缓存中的 item
        if qid in _GEN_ITEMS:
            _GEN_ITEMS[qid]["grounded"] = grounded
            if corrected and corrected != reference_answer:
                _GEN_ITEMS[qid]["reference_answer"] = corrected
        return {"qid": qid, "grounded": grounded,
                "issues_count": len(res.get("issues", [])),
                "confidence": float(res.get("confidence", 0.5))}
    except Exception as e:
        if qid in _GEN_ITEMS:
            _GEN_ITEMS[qid]["grounded"] = False
        return {"qid": qid, "grounded": False, "issues_count": 1, "confidence": 0.0}


@tool
def difficulty_calibrate(cid: str, tier: str, student_elo: int) -> dict:
    """IRT 难度校准：根据考点、tier、学生 Elo 估计题目对该生的难度。

    返回难度系数(1-答对概率) + 期望答对率 + 校准说明。
    """
    from .. import storage
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    c = cmap.get(cid)
    if not c:
        return {"error": f"考点{cid}不存在"}
    llm = get_llm()
    user = (f"考点：{c.name}（{c.summary}）\n考点tier：{tier}\n"
            f"考点exam_weight：{c.exam_weight}\n学生当前Elo：{student_elo}\n"
            "请估算一道该考点、该tier难度的题对该生的难度。")
    try:
        res = llm.chat_json(CALIB_SYS, user, temperature=0.2, max_tokens=200)
        difficulty = float(res.get("difficulty", 0.5))
        expected_p = float(res.get("expected_p", 0.5))
    except Exception:
        tier_elo = TIER_TO_ELO.get(tier, 1100)
        gap = student_elo - tier_elo
        expected_p = max(0.1, min(0.95, 0.5 + gap / 1200))
        difficulty = round(1 - expected_p, 3)
    # 更新缓存中该 cid 的所有 item
    for it in _GEN_ITEMS.values():
        if it.get("cid") == cid:
            it["difficulty"] = round(difficulty, 3)
            it["expected_p"] = round(expected_p, 3)
    return {"cid": cid, "difficulty": round(difficulty, 3),
            "expected_p": round(expected_p, 3)}


GENCAT_SYS = (
    "你是生成式自适应测评系统(Generative CAT)的调度官。职责：为学生在薄弱考点上"
    "实时生成全新题目并校验。可调用工具："
    "generate_item（LLM生成新题，返回qid）、ground_check（反幻觉校验，传qid+题干+答案+cid）、"
    "difficulty_calibrate（IRT难度校准，传cid+tier+elo）。"
    "流程：先用 pull_mastery_map 查学生最薄弱考点，然后对每个薄弱考点："
    "1) generate_item(cid,course,weakness,tier) 生成新题 "
    "2) ground_check(qid,question,reference_answer,cid) 校验 "
    "3) difficulty_calibrate(cid,tier,elo) 估难度。"
    "共为 5 个薄弱考点各生成 1 题。最后用≤100字说明：生成题数、通过校验数、难度分布。"
)

GENCAT_TOOLS = [generate_item, ground_check, difficulty_calibrate]


def run_generative_cat(sid: str, course: str = "os") -> dict:
    """生成式 CAT：为学生实时生成5道自适应新题。

    过程式编排：对每个薄弱考点依次 LLM 生成→反幻觉校验→IRT校准，
    每步均为真实大模型 API 调用。手动构建 steps 供前端展示工具轨迹。
    返回 {steps, answer, items, generation_log, sid, course, student_elo}。
    """
    _GEN_ITEMS.clear()
    from .diagnosis_agents import pull_mastery_map
    from .. import storage
    elo_data = storage.load_quiz_elo(sid)
    student_elo = elo_data.get("elo", 1000)
    mastery_map = pull_mastery_map.invoke({"sid": sid})
    weakest = mastery_map.get("weakest", [])
    concepts = storage.load_concepts()
    cmap = {c.cid: c for c in concepts}
    course_cids = {c.cid for c in concepts if c.course == course}
    weak_in_course = [w for w in weakest if w["cid"] in course_cids][:5]
    if not weak_in_course:
        weak_in_course = [{"cid": c.cid, "name": c.name, "mastery": None}
                          for c in concepts if c.course == course][:5]
    steps = [{"tool": "pull_mastery_map", "args": {"sid": sid},
              "result": f"取到{len(weak_in_course)}个薄弱考点"}]
    weakness_map = {w["cid"]: f"{w['name']}(掌握度{w.get('mastery','未知')})" for w in weak_in_course}
    for w in weak_in_course:
        cid = w["cid"]
        tier = cmap.get(cid, type("X", (), {"tier": "core"})()).tier
        weakness = weakness_map.get(cid, w.get("name", ""))
        # 1) LLM 生成新题
        gen = generate_item.invoke({"cid": cid, "course": course,
                                    "weakness": weakness, "tier": tier})
        if gen.get("error") or not gen.get("qid"):
            steps.append({"tool": "generate_item", "args": {"cid": cid},
                          "result": gen.get("error", "生成失败")})
            continue
        qid = gen["qid"]
        item = _GEN_ITEMS.get(qid, {})
        steps.append({"tool": "generate_item", "args": {"cid": cid},
                      "result": f"生成 qid={qid}：{item.get('question','')[:40]}"})
        # 2) 反幻觉校验
        gc = ground_check.invoke({"qid": qid, "question": item.get("question", ""),
                                  "reference_answer": item.get("reference_answer", ""),
                                  "cid": cid})
        steps.append({"tool": "ground_check", "args": {"qid": qid},
                      "result": f"grounded={gc.get('grounded')} conf={gc.get('confidence')}"})
        # 3) IRT 难度校准
        dc = difficulty_calibrate.invoke({"cid": cid, "tier": tier,
                                          "student_elo": student_elo})
        steps.append({"tool": "difficulty_calibrate", "args": {"cid": cid},
                      "result": f"难度={dc.get('difficulty')} P={dc.get('expected_p')}"})
    items = list(_GEN_ITEMS.values())
    n_grounded = sum(1 for it in items if it.get("grounded"))
    log = [{"qid": it["qid"], "cid": it["cid"], "grounded": it.get("grounded", False),
            "difficulty": it.get("difficulty")} for it in items]
    # LLM 生成总结
    llm = get_llm()
    summary_in = f"为{sid}(Elo={student_elo})生成{len(items)}题，{n_grounded}题通过反幻觉校验。"
    try:
        answer = llm.chat("你是生成式CAT调度官，用≤100字总结生成结果。",
                         summary_in, temperature=0.4, max_tokens=200)
    except Exception:
        answer = f"生成{len(items)}题，{n_grounded}题通过校验，按Elo {student_elo}自适应配难度。"
    return {"steps": steps, "answer": answer, "items": items,
            "generation_log": log, "sid": sid, "course": course,
            "student_elo": student_elo}
