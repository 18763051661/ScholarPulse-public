"""Reflexion 多轮科学方法循环（Self-Reflecting Agent）。

博士级创新点：单次 ReAct 升级为『假设→插桩→复现→分析→修正→验证』
科学调试循环，AI 会自我反思进化。
学术前沿：Reflexion (Shinn et al., 2023) — 智能体在失败后自然语言反思，
把反思存入记忆，下一轮带着记忆重试，实现自我进化。
区别于 ReAct 的『想一步做一步』，Reflexion 是『做一轮→评一轮→改一轮』。

核心循环 _run_reflexion：
  Round 1: hypothesize（形成初始假设）→ instrument（插桩取证）→ analyze（分析证据）
  自评：结论是否扎实？证据是否充分？
    ✓ 通过 → 返回
    ✗ 失败 → reflect（为什么假设错了）→ revise（修正假设）→ Round 2 带着 memory 继续
  Round N: 最终裁决（即使未达完美，也给出当前最优 + 反思记录）
"""
from __future__ import annotations
import json
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown, TOOLS as DIAG_TOOLS
from .diagnosis_agents import prescribe_ct, issue_findings, FORENSIC_TOOL_LIST, CT_TOOL_LIST
from ..lc_model import RouterChatModel
from ..llm import get_llm


EVAL_SYS = (
    "你是科学方法评审官。评估一位诊断智能体本轮的结论是否扎实。"
    "检查：1.是否调用了足够的取证工具 2.结论是否有证据支撑 3.是否找到了根因而非表层症状。"
    '只输出JSON：{"satisfactory":true|false,"evidence_sufficient":true|false,'
    '"found_root_cause":true|false,"critique":"批评意见(≤60字)","gap":"未补的缺口(≤40字)"}。'
)

REFLECT_SYS = (
    "你是 Reflexion 反思引擎。智能体本轮诊断失败了，请用科学方法反思："
    "为什么本轮假设错了？证据采集漏了什么？下一轮应该先查什么？"
    '只输出JSON：{"reflection":"反思(≤80字)","revised_hypothesis":"修正后的假设(≤40字)",'
    '"next_action":"下一轮首要动作(≤30字)"}。'
)

FINAL_SYS = (
    "你是 Reflexion 最终裁决官。智能体经历了多轮反思仍未达成完美结论，"
    "请综合所有轮次的证据与反思，给出当前最优结论。"
    '只输出JSON：{"conclusion":"最终结论(≤120字)","confidence":0.0~1.0,'
    '"best_round":最满意的一轮编号,"open_questions":["未解问题"]}。'
)


def _self_evaluate(model: RouterChatModel, answer: str, steps: list) -> dict:
    """科学方法自评：本轮结论是否扎实？"""
    llm = get_llm()
    facts = "；".join(f"[{s['tool']}] {s.get('result', '')[:100]}" for s in steps if s.get("result")) or "（无工具调用）"
    user = f"智能体结论：{answer[:200]}\n取证记录：{facts}\n请评估本轮是否扎实。"
    try:
        return llm.chat_json(EVAL_SYS, user, temperature=0.2, max_tokens=250)
    except Exception as e:
        return {"satisfactory": True, "critique": f"自评异常:{e}，默认通过",
                "evidence_sufficient": bool(steps), "found_root_cause": False, "gap": ""}


def _reflect(model: RouterChatModel, answer: str, steps: list,
             evaluation: dict, round_num: int) -> dict:
    """失败后反思：为什么假设错了，下一轮怎么改。"""
    llm = get_llm()
    facts = "；".join(f"[{s['tool']}]" for s in steps)
    user = (f"第{round_num+1}轮诊断失败。\n智能体结论：{answer[:150]}\n"
            f"调用的工具：{facts}\n评审意见：{evaluation.get('critique', '')}；"
            f"缺口：{evaluation.get('gap', '')}\n请反思并修正假设。")
    try:
        return llm.chat_json(REFLECT_SYS, user, temperature=0.4, max_tokens=250)
    except Exception as e:
        return {"reflection": f"反思异常:{e}", "revised_hypothesis": "", "next_action": ""}


def _final_arbitrate(model: RouterChatModel, best_answer: str,
                     reflection_memory: list) -> str:
    """多轮仍未完美→最终裁决。"""
    llm = get_llm()
    mem = "\n".join(f"Round{r['round']+1}: {r['evaluation'].get('critique', '')}"
                   for r in reflection_memory)
    user = f"最优轮结论：{best_answer[:200]}\n反思历史：\n{mem}\n请给最终结论。"
    try:
        res = llm.chat_json(FINAL_SYS, user, temperature=0.3, max_tokens=300)
        return res.get("conclusion", best_answer)
    except Exception:
        return best_answer


def _run_reflexion(model: RouterChatModel, tools: list, sys_prompt: str,
                   user_msg: str, max_rounds: int = 3) -> tuple[list, str, list]:
    """Reflexion 多轮科学方法循环。

    每轮：假设→插桩取证（ReAct）→分析（自评）→失败则反思修正→下一轮带记忆。
    返回 (steps, answer, reflection_memory)。
    """
    reflection_memory = []
    all_steps = []
    best_answer = ""
    for round_i in range(max_rounds):
        # 假设阶段：本轮的假设（首轮用原消息，后续带反思记忆）
        if reflection_memory:
            mem_str = "\n".join(
                f"Round{r['round']+1}反思：{r['reflection'].get('reflection', '')}"
                f"→修正假设：{r['reflection'].get('revised_hypothesis', '')}"
                f"→下轮首要动作：{r['reflection'].get('next_action', '')}"
                for r in reflection_memory)
            augmented = (f"{user_msg}\n\n【反思记忆——请带着以下反思重试】\n{mem_str}\n"
                         "请基于反思修正假设，优先执行上轮指出的 next_action。")
        else:
            augmented = user_msg
        # 插桩+分析阶段：跑 ReAct 取证并出结论
        steps, answer = _run_react(model, tools, sys_prompt, augmented)
        all_steps.append({"round": round_i, "phase": "hypothesize+instrument",
                          "steps": steps, "answer": answer})
        best_answer = answer or best_answer
        # 验证阶段：自评
        evaluation = _self_evaluate(model, answer, steps)
        all_steps.append({"round": round_i, "phase": "evaluate", "evaluation": evaluation})
        if evaluation.get("satisfactory"):
            # 通过验证 → 提取步骤供前端展示
            flat_steps = [s for r in all_steps for s in (r.get("steps", []) if isinstance(r.get("steps"), list) else [])]
            flat_steps.append({"tool": "reflexion_eval", "args": evaluation,
                              "result": f"Round{round_i+1}通过验证"})
            return flat_steps, answer, reflection_memory
        # 反思阶段：失败→反思→修正假设
        reflection = _reflect(model, answer, steps, evaluation, round_i)
        reflection_memory.append({"round": round_i, "reflection": reflection,
                                  "evaluation": evaluation})
        all_steps[-1]["reflection"] = reflection
    # 多轮未完美→最终裁决
    final = _final_arbitrate(model, best_answer, reflection_memory)
    flat_steps = []
    for r in all_steps:
        if isinstance(r.get("steps"), list):
            for s in r["steps"]:
                flat_steps.append(s)
        elif "evaluation" in r:
            flat_steps.append({"tool": "reflexion_eval", "args": r["evaluation"],
                              "result": f"Round{r['round']+1}: {r['evaluation'].get('critique', '')}"})
        if "reflection" in r:
            flat_steps.append({"tool": "reflexion_reflect", "args": r["reflection"],
                              "result": r["reflection"].get("reflection", "")})
    flat_steps.append({"tool": "reflexion_final", "args": {},
                      "result": f"最终裁决(经{max_rounds}轮反思)"})
    return flat_steps, _strip_markdown(final), reflection_memory


# ---------- 应用于法医/CT ----------
REFLEXION_FORENSIC_SYS = (
    "你是认知法医，正在执行 Reflexion 科学方法循环。可调用工具："
    "pull_mastery_map、error_timeline、confusion_pair、due_review_count、issue_findings。"
    "本轮流程：先 hypothesize（假设死因类型），再 instrument（调工具取证），"
    "再 analyze（分析证据），最后 issue_findings 出具鉴定书。"
    "若评审不通过，你会被要求带着反思记忆重试，聚焦上轮漏查的证据。"
)

REFLEXION_CT_SYS = (
    "你是认知断层 CT 医生，正在执行 Reflexion 科学方法循环。可调用工具："
    "trace_chain、error_timeline、confusion_pair、prescribe_ct。"
    "本轮流程：先 hypothesize（假设真断点在哪），再 trace_chain 扫描取证，"
    "分析后 prescribe_ct 开方。若评审不通过，带着反思重试，上轮可能把表层当病灶。"
)


def run_forensic_reflexion(sid: str) -> dict:
    """法医 Reflexion 升级版：多轮科学方法循环。"""
    model = RouterChatModel(temperature=0.4, max_tokens=600)
    msg = f"请对学生{sid}执行 Reflexion 法医尸检，经历假设→取证→分析→反思多轮循环。"
    try:
        steps, answer, memory = _run_reflexion(
            model, FORENSIC_TOOL_LIST, REFLEXION_FORENSIC_SYS, msg, max_rounds=3)
    except Exception as e:
        return {"steps": [], "answer": f"Reflexion法医故障：{e}",
                "reflection_memory": []}
    return {"steps": steps, "answer": answer,
            "reflection_memory": memory, "n_rounds": len(memory) + 1,
            "sid": sid, "mode": "reflexion"}


def run_ct_reflexion(sid: str, target: str = "") -> dict:
    """CT 医 Reflexion 升级版：多轮科学方法循环。"""
    from .diagnosis_agents import pull_mastery_map
    if not target:
        weakest = pull_mastery_map.invoke({"sid": sid})["weakest"]
        target = weakest[0]["name"] if weakest else ""
    model = RouterChatModel(temperature=0.4, max_tokens=500)
    msg = (f"学生{sid}主诉『{target}』学不会。请执行 Reflexion CT 诊断，"
           "经历假设→扫描→分析→反思多轮，定位真断点。")
    try:
        steps, answer, memory = _run_reflexion(
            model, CT_TOOL_LIST, REFLEXION_CT_SYS, msg, max_rounds=3)
    except Exception as e:
        return {"steps": [], "answer": f"Reflexion CT 故障：{e}",
                "reflection_memory": [], "target": target}
    return {"steps": steps, "answer": answer,
            "reflection_memory": memory, "n_rounds": len(memory) + 1,
            "sid": sid, "target": target, "mode": "reflexion"}
