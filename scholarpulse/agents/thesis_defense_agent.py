"""多智能体博士答辩委员会（Multi-Agent Debate）。

博士级创新点：单一LLM批改 → 3个AI考官独立评分 → 2轮交叉辩论 → 主持人裁决。
学术前沿：Multi-Agent Debate (Du et al., 2023) — 多个LLM智能体通过辩论
提升推理质量、消除单一模型偏见；Society of Mind (Minsky) 多视角仲裁。
每个考官有独立人格化评判倾向，必须经历对抗后达成相对共识。

流程：
  Round0 — 三派考官独立评分（严格派/宽容派/方法论派）
  Round1 — 互相看对方评分+理由，各自调整自己的评分
  Round2 — 聚焦分歧最大的题，三方激辩
  裁决   — 主持人加权综合 + 分歧度标注 + 仲裁理由
"""
from __future__ import annotations
import json
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel
from ..llm import get_llm


# ---------- 三派考官系统提示 ----------
STRICT_SYS = (
    "你是博士答辩委员会的【严格派考官】。评判原则：以学术严谨为准绳，扣分导向——"
    "只要有逻辑跳跃、概念不精确、关键步骤缺失就扣分。对含糊其辞、似是而非零容忍。"
    '严格只输出一个JSON：{"score":0.0~1.0,"reason":"评分理由(≤50字)",'
    '"flaws":["具体扣分点1","扣分点2"]}。'
)
LENIENT_SYS = (
    "你是博士答辩委员会的【宽容派考官】。评判原则：鼓励创新思维与发散理解，"
    "只要答到核心思想、思路正确即使表述不完美也酌情给分。重创新轻刻板，独到见解给加分。"
    '严格只输出一个JSON：{"score":0.0~1.0,"reason":"评分理由(≤50字)",'
    '"highlights":["加分点1","加分点2"]}。'
)
METHOD_SYS = (
    "你是博士答辩委员会的【方法论派考官】。评判原则：看推导过程与论证方法，"
    "不看结论对错而看论证是否扎实、是否遵循学科方法论。过程正确但结论偏差 > 结论正确但过程混乱。"
    '严格只输出一个JSON：{"score":0.0~1.0,"reason":"评分理由(≤50字)",'
    '"method_notes":["方法学评价1","评价2"]}。'
)

DEBATE_SYS = (
    "你正在参加博士答辩委员会的交叉辩论环节。你会看到另外两位考官对该生的评分与理由，"
    "以及你自己上一轮的评分。请认真审视同行的观点：如果你认同就维持，如果对方说服了你"
    "可以调整评分（调整幅度≤0.2），如果你坚持己见就说明为何对方的批评不成立。"
    '严格只输出一个JSON：{"score":0.0~1.0,"reason":"调整理由(≤60字)",'
    '"concede":["接受的同行观点"],"hold":["坚持的观点"]}。'
)

CHAIRMAN_SYS = (
    "你是博士答辩委员会的【主席/主持人】。职责：综合三位考官最终评分与两轮辩论记录，"
    "出具最终裁决。方法：加权平均（严格派0.35 / 宽容派0.25 / 方法论派0.40）+ 分歧度标注。"
    "分歧度=三方评分极差；极差>0.2记为『有争议』，需在verdict中说明仲裁理由。"
    '严格只输出一个JSON：{"final_score":0.0~1.0,"grade":"A|B|C|D",'
    '"divergence":0.0~1.0,"consensus":true|false,"verdict":"裁决意见(≤80字)"}。'
)

ROLE_WEIGHTS = {"strict": 0.35, "lenient": 0.25, "methodologist": 0.40}
ROLES = [
    ("strict", "严格派", STRICT_SYS),
    ("lenient", "宽容派", LENIENT_SYS),
    ("methodologist", "方法论派", METHOD_SYS),
]


def _build_qa_text(qa_set: list[dict]) -> str:
    """把问答集压成评审文本，最多 5 题。"""
    lines = []
    for i, x in enumerate(qa_set[:5], 1):
        q = (x.get("q") or x.get("question") or "")[:80]
        ans = (x.get("ans") or "")[:120]
        ref = (x.get("ref") or x.get("reference") or "")[:120]
        lines.append(f"【题{i}】{q}\n  学生答：{ans}\n  参考答案：{ref}")
    return "\n".join(lines)


@tool
def convene_examiners(qa_json: str) -> dict:
    """召集三派考官独立评审（Round0）。

    qa_json 为 JSON 字符串，每项 {"q":题干,"ans":学生答,"ref":参考答案}。
    返回三位考官的独立评分与理由。每个考官内部独立调用大模型 API。
    """
    try:
        qa_set = json.loads(qa_json)
    except Exception:
        return {"error": "qa_json 非合法 JSON"}
    if not isinstance(qa_set, list) or not qa_set:
        return {"error": "问答集为空"}
    qa_text = _build_qa_text(qa_set)
    round0 = []
    for role_key, role_name, sys_prompt in ROLES:
        llm = get_llm()
        try:
            res = llm.chat_json(sys_prompt, qa_text, temperature=0.3, max_tokens=400)
            score = float(res.get("score", 0.5))
        except Exception as e:
            score, res = 0.5, {"reason": f"评审异常:{e}"}
        round0.append({
            "role": role_key, "role_name": role_name,
            "score": round(score, 3),
            "reason": res.get("reason", ""),
            "notes": (res.get("flaws") or res.get("highlights")
                      or res.get("method_notes") or []),
        })
    return {"round": 0, "examiners": round0}


@tool
def debate_round(round_json: str, round_num: int = 1) -> dict:
    """交叉辩论轮（Round1/Round2）。每位考官审视另外两位的评分与理由后调整。

    round_json 为上一轮结果 JSON，含 examiners 列表（每项含 role/score/reason/notes）。
    round_num=1 为全景辩论，round_num=2 聚焦分歧最大的题二次激辩。
    """
    try:
        prev = json.loads(round_json)
    except Exception:
        return {"error": "round_json 非合法 JSON"}
    examiners = prev.get("examiners", [])
    if not examiners:
        return {"error": "无考官评分记录"}
    # 找分歧最大的题索引（默认第1题，Round2 聚焦）
    focus_hint = ""
    if round_num >= 2:
        scores = [e["score"] for e in examiners]
        spread = max(scores) - min(scores)
        focus_hint = f"（本轮聚焦最大分歧点，当前极差={spread:.2f}，请就最争议之处激辩）\n"
    new_examiners = []
    llm = get_llm()
    for i, me in enumerate(examiners):
        others = [e for j, e in enumerate(examiners) if j != i]
        others_text = "\n".join(
            f"  {o['role_name']}考官 评分{o['score']}：{o['reason']}；要点：{o.get('notes', [])}"
            for o in others)
        my_text = f"我（{me['role_name']}考官）上轮评分{me['score']}：{me['reason']}；要点：{me.get('notes', [])}"
        user = (f"{focus_hint}当前学生答辩评审交叉辩论第{round_num}轮。\n"
                f"以下是同行考官的观点，请审视后调整你的评分：\n{my_text}\n{others_text}")
        role_name = me["role_name"]
        # 复用辩论系统提示
        try:
            res = llm.chat_json(DEBATE_SYS, user, temperature=0.35, max_tokens=350)
            score = float(res.get("score", me["score"]))
            # 调整幅度限制 ±0.2
            delta = max(-0.2, min(0.2, score - me["score"]))
            score = round(me["score"] + delta, 3)
        except Exception as e:
            score, res = me["score"], {"reason": f"辩论异常:{e}"}
        new_examiners.append({
            "role": me["role"], "role_name": role_name,
            "score": score, "reason": res.get("reason", ""),
            "concede": res.get("concede", []), "hold": res.get("hold", []),
            "prev_score": me["score"],
        })
    return {"round": round_num, "examiners": new_examiners}


@tool
def chairman_arbitrate(final_json: str) -> dict:
    """主持人综合裁决：加权平均三派最终评分 + 分歧度标注，并落盘答辩裁决书。

    final_json 含 examiners 列表（Round2 结果）。
    """
    try:
        data = json.loads(final_json)
    except Exception:
        return {"error": "final_json 非合法 JSON"}
    examiners = data.get("examiners", [])
    if not examiners:
        return {"error": "无最终评分"}
    llm = get_llm()
    summary = "\n".join(
        f"  {e['role_name']}考官 最终评分{e['score']}：{e['reason']}"
        for e in examiners)
    try:
        res = llm.chat_json(CHAIRMAN_SYS, summary, temperature=0.3, max_tokens=300)
        final_score = float(res.get("final_score", 0.5))
        grade = res.get("grade", "C")
        divergence = float(res.get("divergence", 0.0))
        verdict = res.get("verdict", "")
        consensus = bool(res.get("consensus", True))
    except Exception as e:
        # 兜底：纯加权
        weighted = sum(e["score"] * ROLE_WEIGHTS.get(e["role"], 1/3) for e in examiners)
        final_score = round(weighted, 3)
        scores = [e["score"] for e in examiners]
        divergence = round(max(scores) - min(scores), 3)
        grade = "A" if final_score >= 0.85 else "B" if final_score >= 0.7 else "C" if final_score >= 0.5 else "D"
        verdict = f"主席兜底裁决（LLM异常:{e}）：加权均分{final_score}"
        consensus = divergence < 0.2
    rec = {"final_score": final_score, "grade": grade,
           "divergence": divergence, "consensus": consensus, "verdict": verdict,
           "examiners": examiners, "ts": _now_iso()}
    # 落盘
    import os
    from .. import storage
    sid = data.get("sid", "unknown")
    p = os.path.join(storage.DATA_DIR, f"defense_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, **rec}


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ---------- ReAct 协调器 ----------
DEFENSE_SYS = (
    "你是博士答辩委员会的协调官。职责：按流程组织多智能体辩论式答辩。"
    "可调用工具：convene_examiners（召集三派考官独立评分Round0）、"
    "debate_round（交叉辩论，须执行两轮：round_num=1全景辩论，round_num=2聚焦分歧）、"
    "chairman_arbitrate（主持人最终裁决，唯一落盘入口）。"
    "流程：先 convene_examiners 召集三派独立评分，"
    "再 debate_round(round0, 1) 第一轮辩论，"
    "再 debate_round(round1, 2) 第二轮聚焦辩论，"
    "最后 chairman_arbitrate(round2) 出具主席裁决。"
    "最后用≤100字向学生宣布：答辩等级、关键分歧、最终裁定。"
)

DEFENSE_TOOLS = [convene_examiners, debate_round, chairman_arbitrate]


def run_thesis_defense(sid: str, course: str, qa_set: list[dict]) -> dict:
    """多智能体博士答辩委员会。

    过程式编排：Round0三派独立评分→Round1全景辩论→Round2聚焦辩论→主席裁决。
    每步均为真实大模型 API 调用，手动构建 steps 供前端展示工具轨迹。
    绕过 _run_react（其会截断工具结果到180字符，导致完整JSON无法回传、
    ReAct循环因看不全上一轮JSON而提前终止、verdict解析失败）。
    返回 {steps, answer, rounds, verdict, sid, course}。
    """
    qa_json = json.dumps(qa_set[:5], ensure_ascii=False)
    steps = []
    rounds = []
    verdict = {}

    # Round0 — 三派考官独立评分（3 次独立 LLM 调用）
    try:
        round0 = convene_examiners.invoke({"qa_json": qa_json})
    except Exception as e:
        return {"steps": [], "answer": f"答辩Round0故障：{e}",
                "rounds": [], "verdict": {}, "sid": sid, "course": course}
    if round0.get("error"):
        return {"steps": [], "answer": f"答辩Round0失败：{round0['error']}",
                "rounds": [], "verdict": {}, "sid": sid, "course": course}
    rounds.append(round0)
    scores0 = [e.get("score") for e in round0.get("examiners", [])]
    steps.append({"tool": "convene_examiners", "args": {"n_q": len(qa_set[:5])},
                  "result": f"三派独立评分：严格{scores0[0] if len(scores0)>0 else '?'} "
                            f"宽容{scores0[1] if len(scores0)>1 else '?'} "
                            f"方法论{scores0[2] if len(scores0)>2 else '?'}"})

    # Round1 — 全景辩论（3 次 LLM 调用，每位考官审视另两位）
    try:
        round1 = debate_round.invoke({"round_json": json.dumps(round0, ensure_ascii=False),
                                      "round_num": 1})
    except Exception as e:
        round1 = {"round": 1, "examiners": round0.get("examiners", []),
                  "error": f"Round1故障:{e}"}
    rounds.append(round1)
    scores1 = [e.get("score") for e in round1.get("examiners", [])]
    steps.append({"tool": "debate_round", "args": {"round_num": 1},
                  "result": f"R1辩论后调整：{scores1}"})

    # Round2 — 聚焦分歧最大点激辩（3 次 LLM 调用）
    try:
        round2 = debate_round.invoke({"round_json": json.dumps(round1, ensure_ascii=False),
                                      "round_num": 2})
    except Exception as e:
        round2 = {"round": 2, "examiners": round1.get("examiners", []),
                  "error": f"Round2故障:{e}"}
    rounds.append(round2)
    scores2 = [e.get("score") for e in round2.get("examiners", [])]
    steps.append({"tool": "debate_round", "args": {"round_num": 2},
                  "result": f"R2聚焦激辩后：{scores2}"})

    # 主席裁决（1 次 LLM 调用 + 落盘 defense_{sid}.json）
    final_payload = {"examiners": round2.get("examiners", []), "sid": sid}
    try:
        verdict = chairman_arbitrate.invoke(
            {"final_json": json.dumps(final_payload, ensure_ascii=False)})
    except Exception as e:
        verdict = {"error": f"主席裁决故障：{e}"}
    v = verdict if isinstance(verdict, dict) else {}
    steps.append({"tool": "chairman_arbitrate", "args": {},
                  "result": f"主席裁决：等级{v.get('grade','?')} "
                            f"最终分{v.get('final_score','?')} "
                            f"分歧{v.get('divergence','?')} "
                            f"共识={v.get('consensus','?')}"})

    # LLM 生成总结
    llm = get_llm()
    summary_in = (f"学生{sid}《{course}》答辩完成。\n"
                  f"Round0评分：{scores0}\nRound1调整：{scores1}\nRound2聚焦：{scores2}\n"
                  f"主席裁决：等级{v.get('grade','?')} 最终分{v.get('final_score','?')} "
                  f"分歧度{v.get('divergence','?')} 共识={v.get('consensus','?')}\n"
                  f"裁决意见：{v.get('verdict','')}")
    try:
        answer = llm.chat("你是博士答辩委员会主席，用≤100字向学生宣布答辩结果。",
                         summary_in, temperature=0.4, max_tokens=250)
    except Exception:
        answer = (f"答辩完成：等级{v.get('grade','?')}，"
                  f"最终分{v.get('final_score','?')}，{v.get('verdict','')}")

    return {"steps": steps, "answer": answer, "rounds": rounds,
            "verdict": verdict, "sid": sid, "course": course}
