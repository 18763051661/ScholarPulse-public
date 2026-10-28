"""
【L4b 政策听证委员会】—— 多智能体三属性之一：社会性（分工 + 对抗 + 合议）
沙盘给出统计证据，但"该不该推行"是价值判断：同一份数据，效率派、公平派、风险派
会读出不同的结论。委员会让三个立场显式对抗——
  第1轮  各派基于反事实实验数据独立陈词（都要引用数字）
  第2轮  各派看到前面全部发言，交叉质询：支持谁、反驳谁、是否让步
  主持人 综合两轮辩论与统计证据裁决：通过 / 有条件通过 / 暂缓，并给出前置条件
LangGraph 七节点线性流水线：r1_eff → r1_eq → r1_risk → r2_eff → r2_eq → r2_risk → chair。
线性序保证每位发言者能看到所有前序发言（社会性 = 有序的相互看见）。
创新点：把"政策合法性"从单人叙事（policy_sandbox 的顾问）升级为程序正义的合议过程；
LLM 异常时逐角色规则降级——立场模板依然让辩论完整可审计。
"""
from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from .. import storage
from ..llm import get_llm
from ..models import PolicyExperiment, PolicyDebate, DebateSpeech

FACTION_NAME = {"efficiency": "效率派", "equity": "公平派", "risk": "风险派"}

SYS_EFF = """你是教育经济学"效率派"委员。你的天职是单位资源的产出最大化。
关注：效应量大小、成本收益比、规模化的边际收益。发言要求：引用实验数据中的具体数字，
3-4句，立场鲜明但不失风度。"""

SYS_EQ = """你是教育社会学"公平派"委员。你的天职是不让任何一个弱势群体掉队。
关注：分层差异（by_tier）、溢出效应、政策是否加剧两极分化。发言要求：引用实验数据中的具体数字，
3-4句，立场鲜明但不失风度。"""

SYS_RISK = """你是"风险控制派"委员。你的天职是防止不可逆的决策灾难。
关注：置信区间是否跨0、合成人口的外推性、干预的副作用与执行走样。发言要求：引用实验数据中的具体数字，
3-4句，立场鲜明但不失风度。"""

SYS_CHAIR = """你是听证会主持人。给你一份政策的反事实实验统计结果，与效率/公平/风险三派
两轮共六段发言。请综合证据与辩论做出裁决：
  verdict: "通过" / "有条件通过" / "暂缓" 三选一
  rationale: 3-5句裁决理由——哪派的论据更硬、分歧的根源是什么
  conditions: 2-3条推行前置条件（如"先在某分层试点 30 天"）
输出JSON: {"verdict":"...","rationale":"...","conditions":["...","..."]}。只依据给定材料。只输出JSON。"""

SIG = {"pos": "效应显著为正", "cross": "置信区间跨 0、证据不足", "neg": "效应为负"}


class DebateState(TypedDict, total=False):
    exp: dict            # PolicyExperiment.model_dump()
    speeches: list[dict] # [{faction, round, speech}] 线性序追加
    verdict: str         # 主持人裁决：通过 / 有条件通过 / 暂缓
    rationale: str       # 裁决理由
    conditions: list[str]  # 推行前置条件


# ---------- 发言公共逻辑 ----------
def _effect_brief(exp: dict) -> str:
    eff = exp.get("effect", {})
    intent = exp.get("policy_json", {}).get("intent", exp.get("policy_text", ""))
    return (f"政策:{exp.get('policy_text','')}\n意图:{intent}\n"
            f"效应量Δ:{eff.get('delta_avg', 0.0):+.4f}  95%CI:{eff.get('ci95', [0.0, 0.0])}\n"
            f"分层效应:{eff.get('by_tier', {})}\n对照组溢出:{eff.get('spillover_avg', 0.0)}")


def _prior_speeches(state: DebateState) -> str:
    return "\n".join(f"[{FACTION_NAME[s['faction']]}·第{s['round']}轮] {s['speech']}"
                     for s in state.get("speeches", []))


def _speak(state: DebateState, faction: str, rnd: int) -> dict:
    sys_map = {"efficiency": SYS_EFF, "equity": SYS_EQ, "risk": SYS_RISK}
    line = _effect_brief(state["exp"])
    if rnd == 1:
        user = f"第1轮发言。基于以下反事实实验数据陈述你的立场：\n{line}"
    else:
        user = (f"第2轮发言。前面委员的发言：\n{_prior_speeches(state)}\n\n"
                f"实验数据：\n{line}\n请交叉质询：明确支持或反驳某位委员，并给出你的坚持或让步。")
    try:
        text = get_llm().chat(sys_map[faction], user, temperature=0.7, max_tokens=400)
        text = text.strip() or _fallback_speech(state, faction, rnd)
    except Exception:
        text = _fallback_speech(state, faction, rnd)
    return {"speeches": state.get("speeches", []) + [{"faction": faction, "round": rnd,
                                                      "speech": text}]}


def _sig_of(eff: dict) -> str:
    lo, hi = (eff.get("ci95") or [0.0, 0.0])[:2]
    if lo > 0:
        return SIG["pos"]
    if hi <= 0:
        return SIG["neg"]
    return SIG["cross"]


def _fallback_speech(state: DebateState, faction: str, rnd: int) -> str:
    """LLM 异常时的立场模板降级：数字驱动，辩论依然完整。"""
    eff = state["exp"].get("effect", {})
    d = eff.get("delta_avg", 0.0)
    lo, hi = (eff.get("ci95") or [0.0, 0.0])[:2]
    sig = _sig_of(eff)
    tier = eff.get("by_tier") or {}
    weakest = min(tier, key=tier.get) if tier else "未知分层"
    if faction == "efficiency":
        if rnd == 1:
            return (f"（规则降级）实验显示效应量 {d:+.4f}，{sig}。以最小干预撬动该量级的掌握度提升，"
                    f"单位成本收益为正；若分层红利确实存在，规模化后边际成本还会摊薄。")
        return (f"（规则降级）回应质疑：{d:+.4f} 的均值效应已含全部分层；"
                f"效率不与公平对立——政策做大了蛋糕，分配问题才有的分。")
    if faction == "equity":
        if rnd == 1:
            return (f"（规则降级）分层效应里 {weakest} 层受益最小。均值的胜利可能是分层的灾难，"
                    f"请效率派解释：为什么不应为弱势层设计额外倾斜？")
        return (f"（规则降级）我不撤回担忧：溢出与分层证据都指向 {weakest} 层受益不足；"
                f"我愿意支持推行，但必须附带弱势层补偿条款。")
    lo4, hi4 = f"{lo:.4f}", f"{hi:.4f}"
    if rnd == 1:
        return (f"（规则降级）95%CI [{lo4}, {hi4}]，{sig}。且结论来自合成人口仿真，"
                f"外推到真实学区存在模型风险，执行走样会进一步放大不确定性。")
    return (f"（规则降级）两位委员的分歧本质是证据强度问题：CI {'跨 0' if lo <= 0 <= hi else '未跨 0'}。"
            f"我的让步是不否决，但坚持任何推行都必须绑定小范围试点与逐月监测。")


# ---------- 七个节点（线性图：每位发言者都能看见全部前序发言） ----------
def _r1_eff(state: DebateState) -> dict:
    return _speak(state, "efficiency", 1)


def _r1_eq(state: DebateState) -> dict:
    return _speak(state, "equity", 1)


def _r1_risk(state: DebateState) -> dict:
    return _speak(state, "risk", 1)


def _r2_eff(state: DebateState) -> dict:
    return _speak(state, "efficiency", 2)


def _r2_eq(state: DebateState) -> dict:
    return _speak(state, "equity", 2)


def _r2_risk(state: DebateState) -> dict:
    return _speak(state, "risk", 2)


def _chair(state: DebateState) -> dict:
    eff = state["exp"].get("effect", {})
    user = (f"统计证据：{_effect_brief(state['exp'])}\n\n"
            f"委员会发言记录：\n{_prior_speeches(state)}")
    try:
        data = get_llm().chat_json(SYS_CHAIR, user)
        verdict = str(data.get("verdict", "有条件通过"))
        rationale = str(data.get("rationale", ""))
        conditions = [str(c) for c in (data.get("conditions") or [])][:3]
    except Exception:
        # 规则降级裁决：证据强度决定裁决档位
        lo, hi = (eff.get("ci95") or [0.0, 0.0])[:2]
        d = eff.get("delta_avg", 0.0)
        if lo > 0:
            verdict = "通过"
            conditions = ["建立逐月效果监测与回滚预案", "满一年后做外部队复核"]
        elif hi <= 0:
            verdict = "暂缓"
            conditions = ["重新设计干预强度后再上会"]
        else:
            verdict = "有条件通过"
            conditions = ["先在风险最低的分层试点 30 天并复核 CI", "建立弱势层补偿机制"]
        rationale = (f"（规则降级）统计证据显示 Δ={d:+.4f}、{_sig_of(eff)}；"
                     f"效率派与公平派的分歧根源在'均值收益如何分配'，风险派提醒了证据边界。"
                     f"综合两轮辩论，以当前证据强度做出上述裁决。")
    return {"verdict": verdict, "rationale": rationale, "conditions": conditions}


def build_debate_graph():
    g = StateGraph(DebateState)
    for name, fn in [("r1_eff", _r1_eff), ("r1_eq", _r1_eq), ("r1_risk", _r1_risk),
                     ("r2_eff", _r2_eff), ("r2_eq", _r2_eq), ("r2_risk", _r2_risk),
                     ("chair", _chair)]:
        g.add_node(name, fn)
    g.add_edge(START, "r1_eff")
    g.add_edge("r1_eff", "r1_eq")
    g.add_edge("r1_eq", "r1_risk")
    g.add_edge("r1_risk", "r2_eff")
    g.add_edge("r2_eff", "r2_eq")
    g.add_edge("r2_eq", "r2_risk")
    g.add_edge("r2_risk", "chair")
    g.add_edge("chair", END)
    return g.compile()


def run_debate(exp: PolicyExperiment | None = None) -> dict:
    """对最近一次有效沙盘结果召开听证会并落盘 debate.json。无沙盘 → ValueError。"""
    exp = exp or storage.load_policy()
    if exp is None or not exp.effect:
        raise ValueError("无有效沙盘结果，请先运行政策沙盘")
    final = build_debate_graph().invoke({"exp": exp.model_dump(), "speeches": []})
    d = PolicyDebate(ts=storage.now_iso(), policy_text=exp.policy_text,
                     speeches=[DebateSpeech(**s) for s in final.get("speeches", [])],
                     verdict=final["verdict"], rationale=final["rationale"],
                     conditions=final["conditions"])
    storage.save_debate(d)
    return d.model_dump()
