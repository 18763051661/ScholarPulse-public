"""
【L4 反事实政策沙盘】—— 用 ABM 做"教育政策的物理引擎"
政策还没颁布，先在合成学区里做一场随机对照试验（RCT）。LangGraph 五级线性流水线：
  政策解析官(LLM)   → 把自然语言政策解析为结构化干预参数（失败返回固定错误结构）
  实验设计师        → 按能力先验排序配对 + 抛币分组（匹配对设计，两组能力分布对齐）
  双世界仿真        → 同 seed 跑基线世界与处理世界；干预只作用于处理组。
                      每生独立随机流 + 无状态传染哈希流 ⇒ 对照组在两个世界逐位一致，
                      任何组间差异只能来自政策本身（或经社交图的真实溢出）
  统计分析师(纯Python) → 处理组逐生 Δ、bootstrap 95% CI、按 tier 分解、对照组溢出量
  政策顾问(LLM)     → 把统计结果翻译为决策叙事与风险提示
创新点：因果效应不靠"前后对比"（会混入自然增长与回归均值），而靠"同 seed 平行世界"——
潜在结果框架（Rubin causal model）的工程化实现：Δ = E[Y(处理世界) - Y(基线世界)]。
"""
from __future__ import annotations
import random
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from .. import storage
from ..llm import get_llm
from ..models import PolicyExperiment
from ..society import abm

B_BOOTSTRAP = 500        # bootstrap 重采样次数

SYS_PARSE = """你是教育政策解析官。把学区总监的自然语言政策解析为结构化实验参数。
仅支持一类干预语义：对满足条件的学生，每天增加 extra_questions 道（1~5）强制复习题。
  condition 只能是 {"all": true}（全体学生）或 {"avg_below": 数值}（平均掌握度低于阈值的学生）
  action 固定为 "extra_review"，extra_questions 为 1~5 的整数
输出JSON: {"valid": true, "intent": "政策的一句话概括", "condition": {...}, "action": "extra_review", "extra_questions": n}
无法解析为上述语义时输出: {"valid": false, "reason": "原因"}
只输出JSON。"""

SYS_ADVISE = """你是教育政策顾问。给你一项政策在合成学区反事实实验（随机对照、同seed双世界）的统计结果。
请输出：
  narrative: 4-6句，面向学区总监的决策建议——效应大小是否值得推行、对哪些学生层最有效
  risk_notes: 2-3句风险提示——CI是否跨0、合成人口外推性、公平性
输出JSON: {"narrative":"...","risk_notes":"..."}。只依据给定数据。只输出JSON。"""


class SandboxState(TypedDict, total=False):
    policy_text: str
    days: int
    seed: int
    policy_json: dict       # 解析官产物（或外部注入，供测试/程序化调用）
    groups: dict            # {"control": [sid], "treatment": [sid]}
    worlds: dict            # {"baseline": run_world产物, "treatment": ...}
    analysis: dict          # {delta_avg, ci95, by_tier, spillover_avg, ...}
    narrative: str
    risk_notes: str


# ---------- 节点 1：政策解析官 ----------
def _node_parse(state: SandboxState) -> dict:
    injected = state.get("policy_json")
    if injected:                                     # 外部注入（测试/程序化调用），跳过 LLM
        return {"policy_json": injected}
    try:
        data = get_llm().chat_json(SYS_PARSE, f"政策文本：{state['policy_text']}")
        if not data.get("valid"):
            pj = {"valid": False, "reason": str(data.get("reason", "政策语义无法解析"))}
        else:
            pj = {"valid": True,
                  "intent": str(data.get("intent", state["policy_text"])),
                  "condition": data.get("condition") or {"all": True},
                  "action": "extra_review",
                  "extra_questions": max(0, min(5, int(data.get("extra_questions", 0))))}
    except Exception:
        pj = {"valid": False, "reason": "政策解析失败（LLM 不可用），实验已中止"}   # 固定错误结构
    return {"policy_json": pj}


# ---------- 节点 2：实验设计师（匹配对随机分组） ----------
def _node_design(state: SandboxState) -> dict:
    if not state.get("policy_json", {}).get("valid"):
        return {}
    soc = storage.load_society()
    if soc is None:
        abm._ensure_society()
        soc = storage.load_society()
    # 能力先验降序排列后相邻配对（匹配对），对内抛币决定进入处理组还是对照组
    ranked = sorted(soc.traits, key=lambda s: (-soc.traits[s].ability_prior, s))
    rng = random.Random(state["seed"])
    control, treatment = [], []
    for i in range(0, len(ranked) - 1, 2):
        a, b = ranked[i], ranked[i + 1]
        if rng.random() < 0.5:
            control.append(a)
            treatment.append(b)
        else:
            treatment.append(a)
            control.append(b)
    return {"groups": {"control": sorted(control), "treatment": sorted(treatment)}}


# ---------- 节点 3：双世界仿真（同 seed，干预仅处理组） ----------
def _node_run_both(state: SandboxState) -> dict:
    if not state.get("groups"):
        return {}
    tg = state["groups"]["treatment"]
    base = abm.run_world(days=state["days"], seed=state["seed"], track_sids=tg)
    treat = abm.run_world(days=state["days"], seed=state["seed"],
                          policy=state["policy_json"], treatment=tg, track_sids=tg)
    return {"worlds": {"baseline": base, "treatment": treat}}


# ---------- 节点 4：统计分析师（纯 Python，bootstrap CI + 分层分解） ----------
def _node_analyze(state: SandboxState) -> dict:
    if not state.get("worlds"):
        return {}
    w1 = state["worlds"]["baseline"]["avg"]
    w2 = state["worlds"]["treatment"]["avg"]
    tgroup = state["groups"]["treatment"]
    deltas = [w2[s] - w1[s] for s in tgroup]
    n = len(deltas)
    delta = sum(deltas) / n if n else 0.0
    rng = random.Random(state["seed"])               # bootstrap 也固定 seed，实验完全可复现
    boots = []
    for _ in range(B_BOOTSTRAP):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    ci95 = [boots[int(0.025 * B_BOOTSTRAP)], boots[min(int(0.975 * B_BOOTSTRAP), B_BOOTSTRAP - 1)]]
    # 按学生层级分解 + 对照组溢出效应（经社交图的间接影响）
    students = storage.load_students()
    by_tier: dict[str, list[float]] = {}
    for s in tgroup:
        tier = students[s].tier if s in students else "unknown"
        by_tier.setdefault(tier, []).append(w2[s] - w1[s])
    cgroup = state["groups"]["control"]
    spill = [w2[s] - w1[s] for s in cgroup]
    analysis = {"delta_avg": round(delta, 4),
                "ci95": [round(ci95[0], 4), round(ci95[1], 4)],
                "by_tier": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_tier.items())},
                "tier_n": {k: len(v) for k, v in sorted(by_tier.items())},
                "spillover_avg": round(sum(spill) / len(spill), 4) if spill else 0.0,
                "n_treatment": n,
                "curves": {"baseline": state["worlds"]["baseline"]["track"],
                           "treated": state["worlds"]["treatment"]["track"]}}
    return {"analysis": analysis}


# ---------- 节点 5：政策顾问 ----------
def _node_advise(state: SandboxState) -> dict:
    if not state.get("analysis"):
        return {}
    an = state["analysis"]
    pj = state["policy_json"]
    brief = {k: v for k, v in pj.items() if k != "valid"}
    stats = {k: an[k] for k in ("delta_avg", "ci95", "by_tier", "spillover_avg", "n_treatment")}
    user = (f"政策意图:{pj.get('intent', state['policy_text'])}\n干预参数:{brief}\n"
            f"统计结果:{stats}\n（days={state['days']}, seed={state['seed']}）")
    try:
        data = get_llm().chat_json(SYS_ADVISE, user)
        narrative, risk = str(data.get("narrative", "")), str(data.get("risk_notes", ""))
    except Exception:
        d, (lo, hi) = an["delta_avg"], an["ci95"]
        sig = "显著为正" if lo > 0 else ("不显著" if hi <= 0 else "方向为正但置信区间跨 0")
        narrative = (f"（LLM不可用，规则降级）该政策使处理组平均掌握度变化 {d:+.4f}"
                     f"（95% CI [{lo:.4f}, {hi:.4f}]），效应{sig}。分层看："
                     + "；".join(f"{k} 层 {v:+.4f}" for k, v in an["by_tier"].items())
                     + "。建议优先在效应最大的学生层试点。")
        risk_notes = ("CI 跨 0 说明证据不足，建议延长观察期或加大干预强度；" if lo <= 0 <= hi else
                      "效应显著，但需注意：") + "结论来自合成人口仿真，真实推行前应小范围试点验证。"
    return {"narrative": narrative, "risk_notes": risk}


def build_sandbox_graph():
    """五级线性流水线：解析 → 设计 → 双跑 → 分析 → 顾问。"""
    g = StateGraph(SandboxState)
    g.add_node("parse", _node_parse)
    g.add_node("design", _node_design)
    g.add_node("run_both", _node_run_both)
    g.add_node("analyze", _node_analyze)
    g.add_node("advise", _node_advise)
    g.add_edge(START, "parse")
    g.add_edge("parse", "design")
    g.add_edge("design", "run_both")
    g.add_edge("run_both", "analyze")
    g.add_edge("analyze", "advise")
    g.add_edge("advise", END)
    return g.compile()


def run_sandbox(policy_text: str, days: int = 30, seed: int = 42,
                policy_json: dict | None = None) -> dict:
    """跑完整反事实实验并落盘 policy.json。policy_json 可注入以跳过 LLM 解析（测试用）。"""
    graph = build_sandbox_graph()
    final = graph.invoke({"policy_text": policy_text, "days": max(1, days),
                          "seed": seed, "policy_json": policy_json})
    pj = final.get("policy_json", {})
    if not pj.get("valid"):                          # 解析失败：审计留痕，不执行实验
        exp = PolicyExperiment(policy_text=policy_text, policy_json=pj, days=days, seed=seed,
                               effect={}, narrative=f"实验未执行：{pj.get('reason', '解析失败')}",
                               risk_notes="")
        storage.save_policy(exp)
        return exp.model_dump()
    exp = PolicyExperiment(policy_text=policy_text, policy_json=pj, days=days, seed=seed,
                           effect=final["analysis"], narrative=final["narrative"],
                           risk_notes=final["risk_notes"])
    storage.save_policy(exp)
    return exp.model_dump()
