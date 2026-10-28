"""
【L3 误区流行病学】—— 把"知识点误区"当作传染病来防治
把每个 (学生, 概念) 抽象为一个 SIR 流行病学单元：
  S 易感（尚未形成该误区）→ I 感染（持有错误认知）→ R 康复（纠正/免疫）
三条传播动力：
  外源点燃 —— ABM 答题流中"首次答错"即自然感染日（误区在独立练习中自发产生）
  内源传播 —— 感染者沿社交边"讨论"，以 β·w·social_activity 概率把误区传染给易感同学
  自愈康复 —— 每天以 γ 概率自行纠正（偶尔复习到位）
三件流行病学武器（纯 Python 自实现，不引 networkx）：
  1. R0 = β_c·k̄·D̄ 逐概念估计 —— 找出"最容易流行"的误区
  2. 幂迭代求特征向量中心性 —— 找出"超级传播者"学生
  3. 反事实回测 design_intervention —— 拆组（破坏强连结构）/疫苗式复习（预免疫枢纽），
     同 seed 重放传播动力学，量化干预的真实效果
"""
from __future__ import annotations
import math
import random

from ..models import EpidemicState, R0Estimate
from .. import storage
from .population import seed_society

CONTAGION_BETA = 0.25   # 与 ABM 同源的全局传播系数
RECOVERY_GAMMA = 0.15   # 每日自愈概率（平均感染期 ≈ 1/γ ≈ 6.7 天）
REGROUP_FACTOR = 0.3    # 拆组干预：组内强连边传播系数衰减为 30%
VACCINE_TOP_N = 10      # 疫苗式复习默认覆盖的中心性枢纽人数


def _ensure_society():
    soc = storage.load_society()
    if soc is None:
        seed_society(42)
        soc = storage.load_society()
    return soc


def _adjacency(soc) -> dict[str, list[tuple[str, float, str]]]:
    """对称带权邻接表：adj[sid] = [(邻居, 边权, 边类型)]，双向注册。"""
    adj: dict[str, list[tuple[str, float, str]]] = {sid: [] for sid in soc.traits}
    for e in soc.edges:
        if e.src in adj and e.dst in adj:
            adj[e.src].append((e.dst, e.weight, e.kind))
            adj[e.dst].append((e.src, e.weight, e.kind))
    return adj


def _day_of(ts: str) -> int:
    """ABM 答题时间戳 'virtual-day-03' → 3；无法解析返回 -1。"""
    try:
        return int(ts.rsplit("-", 1)[1])
    except Exception:
        return -1


def init_states() -> dict[str, dict[str, int]]:
    """从 ABM 答题流提取自然感染种子：每对 (sid,cid) 首次答错的虚拟天 = 感染日。"""
    soc = _ensure_society()
    seeds: dict[str, dict[str, int]] = {}
    for sid in sorted(soc.traits):
        for rec in storage.load_answers(sid):
            if rec.correct:
                continue
            d = _day_of(rec.ts)
            if d < 0:
                continue
            cur = seeds.setdefault(rec.cid, {})
            if sid not in cur or d < cur[sid]:
                cur[sid] = d
    return seeds


def _power_iteration(adj: dict, max_iter: int = 100,
                     tol: float = 1e-10) -> tuple[dict[str, int], dict[str, float]]:
    """
    幂迭代求（度中心性, 特征向量中心性）。手写 (A+I)·v 迭代 + L2 归一化，不引 networkx。
    用移位矩阵 A+I 而非裸 A：星形图等二部图存在 ±λ 特征值对，裸幂迭代在两对称子空间间
    永久振荡；移位后特征向量不变但特征值整体为正，负分量按 |s-λ|/(s+λ) 指数衰减。
    adj 兼容 (邻居, 权重) 或 (邻居, 权重, 类型) 两种边元组。
    """
    SHIFT = 1.0
    nodes = sorted(adj)
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}, {}
    deg = {a: len(nbs) for a, nbs in adj.items()}
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(max_iter):
        Av = [SHIFT * v[i] for i in range(n)]
        for a in nodes:
            s = 0.0
            for e in adj[a]:
                s += e[1] * v[idx[e[0]]]
            Av[idx[a]] += s
        norm = math.sqrt(sum(x * x for x in Av))
        if norm == 0:
            break
        Av = [x / norm for x in Av]
        delta = max(abs(Av[i] - v[i]) for i in range(n))
        v = Av
        if delta < tol:
            break
    return deg, {nodes[i]: v[i] for i in range(n)}


def _run_sir(seeds: dict[str, dict[str, int]],
             adj: dict[str, list[tuple[str, float, str]]], traits: dict,
             days: int, seed: int = 42, beta: float = CONTAGION_BETA,
             gamma: float = RECOVERY_GAMMA, sg_factor: float = 1.0,
             vaccinated: set[tuple[str, str]] | None = None,
             focus_cid: str | None = None, total_cells: int = 0) -> dict:
    """
    核心 SIR 推进（纯函数，同 seed 严格可复现）。
    sg_factor<1 模拟拆组（组内强连边传播被削弱）；vaccinated 预免疫集合模拟疫苗式复习。
    返回 {"state": {cid:{sid:[状态,感染日,康复日]}}, "curve": 每日SIR,
          "durations": {cid: 平均感染期}, "focus": focus_cid 的疫情统计}。
    """
    rng = random.Random(seed)
    vac = vaccinated or set()
    cells: dict[str, dict[str, list]] = {}
    for sid, cid in vac:                       # 疫苗：预免疫直接 R（t_infect=-1 不计为感染史）
        cells.setdefault(cid, {})[sid] = ["R", -1, 0]
    for cid, smap in seeds.items():            # 种子：感染日取决于首次答错日
        for sid, t0 in smap.items():
            if (sid, cid) in vac:
                continue
            cells.setdefault(cid, {})[sid] = ["I", t0, -1]
    curve, focus_curve = [], []
    for d in range(1, days + 1):
        # 1) 外源点燃：该虚拟日首次答错的自然感染（已被传播/免疫者跳过）
        for cid in sorted(seeds):
            for sid in sorted(seeds[cid]):
                if seeds[cid][sid] == d and sid not in cells.get(cid, {}):
                    cells.setdefault(cid, {})[sid] = ["I", d, -1]
        # 2) 内源传播：感染者沿社交边击穿易感邻居（扫快照，当日新感次日才传播）
        newly: list[tuple[str, str]] = []
        for cid in sorted(cells):
            smap = cells[cid]
            for sid in sorted(smap):
                cell = smap[sid]
                if cell[0] != "I" or d <= cell[1]:
                    continue
                for nb, w, kind in adj.get(sid, ()):
                    t = smap.get(nb)
                    if t is not None and t[0] != "S":
                        continue
                    b = beta * (w * sg_factor if kind == "study_group" else w)
                    if rng.random() < b * traits[nb].social_activity:
                        newly.append((cid, nb))
        for cid, nb in newly:
            cells[cid][nb] = ["I", d, -1]
        # 3) 自愈康复：感染满一天后每天以 γ 概率纠正
        for cid in sorted(cells):
            smap = cells[cid]
            for sid in sorted(smap):
                cell = smap[sid]
                if cell[0] == "I" and d > cell[1] and rng.random() < gamma:
                    cell[0], cell[2] = "R", d
        # 4) 当日 SIR 曲线（S = 总单元 - I - R；总单元 = 学生数 × 考点数）
        i = sum(1 for smap in cells.values() for c in smap.values() if c[0] == "I")
        r = sum(1 for smap in cells.values() for c in smap.values() if c[0] == "R")
        curve.append({"day": d, "s": max(total_cells - i - r, 0), "i": i, "r": r})
        if focus_cid:
            fi = sum(1 for c in cells.get(focus_cid, {}).values() if c[0] == "I")
            focus_curve.append({"day": d, "i": fi})
    # 观测感染期：康复者取 (康复-感染)，未康复者按观察期删失
    durations: dict[str, float] = {}
    for cid, smap in cells.items():
        ds = []
        for c in smap.values():
            if c[0] == "R" and c[2] > c[1] >= 0:
                ds.append(c[2] - c[1])
            elif c[0] == "I" and c[1] >= 0:
                ds.append(days - c[1])
        if ds:
            durations[cid] = sum(ds) / len(ds)
    focus = None
    if focus_cid:
        fcells = cells.get(focus_cid, {})
        ever = sorted(sid for sid, c in fcells.items() if c[0] == "I" and c[1] >= 0
                      or (c[0] == "R" and c[1] >= 0))
        focus = {"peak_i": max((c["i"] for c in focus_curve), default=0),
                 "final_r": sum(1 for c in fcells.values() if c[0] == "R" and c[1] >= 0),
                 "ever_infected": ever, "infected_n": len(ever)}
    return {"state": cells, "curve": curve, "durations": durations, "focus": focus}


def propagate(days: int = 30, seed: int = 42, beta: float = CONTAGION_BETA,
              gamma: float = RECOVERY_GAMMA) -> dict:
    """在社交图上推进误区 SIR 传播并落盘 epidemic.json。同 seed 可复现。"""
    soc = _ensure_society()
    adj = _adjacency(soc)
    seeds = init_states()
    concepts = [c for c in storage.load_concepts() if c.noise != "filtered"]
    total_cells = len(soc.traits) * len(concepts)
    res = _run_sir(seeds, adj, soc.traits, days, seed, beta, gamma,
                   total_cells=total_cells)
    states = [EpidemicState(sid=sid, cid=cid, state=cell[0], t_infect=int(cell[1]))
              for cid, smap in sorted(res["state"].items())
              for sid, cell in sorted(smap.items())]
    epi = storage.load_epidemic()
    epi["states"] = [s.model_dump() for s in states]
    epi["sir_curve"] = res["curve"]
    meta = epi.get("meta", {})
    meta.update({"sir_days": days, "sir_seed": seed, "sir_beta": beta,
                 "durations": {k: round(v, 3) for k, v in res["durations"].items()}})
    epi["meta"] = meta
    storage.save_epidemic(epi)
    curve = res["curve"]
    return {"days": days, "peak_i": max((c["i"] for c in curve), default=0),
            "final_i": curve[-1]["i"] if curve else 0,
            "final_r": curve[-1]["r"] if curve else 0,
            "seeded_cells": sum(len(m) for m in seeds.values())}


def estimate_r0(beta: float = CONTAGION_BETA,
                gamma: float = RECOVERY_GAMMA) -> list[R0Estimate]:
    """
    逐概念估计基本传染数 R0 = β_c·k̄·D̄：
      β_c = β × 该概念感染者的平均边权（网络位置决定有效接触强度）
      k̄   = 感染者平均度；D̄ = 观测平均感染期（propagate 沉淀，缺失时退化为 1/γ）
    R0 对 β 单调（沙盘/单测依赖此性质）。
    """
    epi = storage.load_epidemic()
    if not epi.get("states"):
        propagate()
        epi = storage.load_epidemic()
    soc = _ensure_society()
    adj = _adjacency(soc)
    durations = epi.get("meta", {}).get("durations", {})
    by_cid: dict[str, list[dict]] = {}
    for st in epi.get("states", []):
        if st["state"] in ("I", "R") and st.get("t_infect", -1) >= 0:
            by_cid.setdefault(st["cid"], []).append(st)
    out: list[R0Estimate] = []
    for cid in sorted(by_cid):
        inf = by_cid[cid]
        sids = [s["sid"] for s in inf]
        ks = [len(adj.get(s, ())) for s in sids]
        k = sum(ks) / len(ks) if ks else 0.0
        ws = [w for s in sids for _, w, _ in adj.get(s, ())]
        w_mean = (sum(ws) / len(ws)) if ws else 0.0
        d = durations.get(cid) or (1.0 / gamma)
        b_c = beta * w_mean
        out.append(R0Estimate(cid=cid, beta=round(b_c, 4), k=round(k, 3),
                              d=round(d, 2), r0=round(b_c * k * d, 4),
                              infected_n=len(inf)))
    out.sort(key=lambda r: -r.r0)
    return out


def superspreaders(top_n: int = 10) -> list[dict]:
    """特征向量中心性 Top-N 学生 —— 传播网络枢纽（疫苗式复习的默认目标）。"""
    soc = _ensure_society()
    adj = _adjacency(soc)
    deg, eig = _power_iteration(adj)
    epi = storage.load_epidemic()
    cnt: dict[str, int] = {}
    for st in epi.get("states", []):
        if st["state"] in ("I", "R") and st.get("t_infect", -1) >= 0:
            cnt[st["sid"]] = cnt.get(st["sid"], 0) + 1
    students = storage.load_students()
    rank = sorted(soc.traits, key=lambda s: (-eig.get(s, 0.0), s))[:top_n]
    return [{"sid": s, "name": students[s].name if s in students else s,
             "degree": deg.get(s, 0), "eigen": round(eig.get(s, 0.0), 4),
             "infected_concepts": cnt.get(s, 0)} for s in rank]


def design_intervention(cid: str, action: str = "regroup",
                        target_sids: list[str] | None = None) -> dict:
    """
    反事实回测：同 seed 重放传播动力学，对比干预前后该概念误区的疫情规模。
      regroup —— 拆散学习小组：组内强连边传播系数 ×0.3（接触结构被破坏）
      vaccine —— 疫苗式复习：预免疫（默认 = 该概念感染者中中心性最高的 VACCINE_TOP_N 人）
    返回 {before:{peak_i, final_r, infected_n}, after:{...},
          effect:{peak_reduction, infections_averted, averted_pct}}
    """
    soc = _ensure_society()
    adj = _adjacency(soc)
    epi = storage.load_epidemic()
    days = int(epi.get("meta", {}).get("sir_days", 30))
    seed = int(epi.get("meta", {}).get("sir_seed", 42))
    seeds = init_states()
    base = _run_sir(seeds, adj, soc.traits, days, seed, focus_cid=cid)
    if action == "regroup":
        alt = _run_sir(seeds, adj, soc.traits, days, seed,
                       sg_factor=REGROUP_FACTOR, focus_cid=cid)
    elif action == "vaccine":
        targets = list(target_sids or [])
        if not targets:
            _, eig = _power_iteration(adj)
            infected = sorted(base["state"].get(cid, {}),
                              key=lambda s: (-eig.get(s, 0.0), s))
            targets = infected[:VACCINE_TOP_N]
        alt = _run_sir(seeds, adj, soc.traits, days, seed,
                       vaccinated={(s, cid) for s in targets}, focus_cid=cid)
    else:
        raise ValueError(f"未知干预类型: {action}")
    b, a = base["focus"], alt["focus"]
    averted = len(set(b["ever_infected"]) - set(a["ever_infected"]))
    pct = round(averted / len(b["ever_infected"]), 4) if b["ever_infected"] else 0.0
    return {"cid": cid, "action": action,
            "before": {"peak_i": b["peak_i"], "final_r": b["final_r"],
                       "infected_n": b["infected_n"]},
            "after": {"peak_i": a["peak_i"], "final_r": a["final_r"],
                      "infected_n": a["infected_n"]},
            "effect": {"peak_reduction": b["peak_i"] - a["peak_i"],
                       "infections_averted": averted, "averted_pct": pct}}


def dashboard() -> dict:
    """疫情仪表盘聚合：SIR 曲线 + 逐概念 R0 + 超级传播者（无数据时自动补跑）。"""
    epi = storage.load_epidemic()
    if not epi.get("states"):
        propagate()
        epi = storage.load_epidemic()
    return {"sir_curve": epi.get("sir_curve", []),
            "r0_list": [r.model_dump() for r in estimate_r0()],
            "superspreaders": superspreaders(),
            "meta": epi.get("meta", {})}
