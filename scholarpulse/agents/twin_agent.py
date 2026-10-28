"""
【数字孪生对抗赛智能体】—— 本系统差异化创新
与其他"AI 教学生"的智能体相反，本模块让 AI 先学会"模仿学生"：
1. 认知建模师 build_profile：根据答题历史构建学生的数字孪生画像（误区+信心偏差+人格）
2. 红方出题人 make_bait：专门瞄准孪生认知盲区设计诱饵题（复用反幻觉校验）
3. 孪生本人 twin_answer：在看不到学生答案的情况下盲答同一题（函数签名物理双盲）
4. 揭晓判官 reveal_analysis：对比双方作答，分析预测偏差暴露了建模的哪点盲区
核心玩法：孪生猜错的地方 = 系统最不了解你的地方；对战结果回写画像，孪生随数据进化。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import TwinProfile, TwinMisconception, TwinBattle, Concept, QuizItem
from .. import storage
from . import verify_agent, quiz_agent

SYS_PROFILE = """你是认知建模师。根据学生的答题记录、各考点掌握度与考点信息，推断该学生的"数字孪生画像"。
输出JSON: {"persona":"2-3句人格化描述：这个学生怎么学、怎么错、怎么自估把握",
"misconceptions":[{"cid":"考点ID","name":"考点名","pattern":"他在该考点的典型错误模式"}],
"confidence_bias":"overconfident|underconfident|balanced"}
只依据给定数据推断，不要编造数据里没有的行为。只输出JSON。"""

SYS_BAIT = """你是狡猾的红方出题人，手里有一份学生的"数字孪生画像"（含他的典型错误模式）。
请针对画像中的认知盲区出一道诱饵简答题：题面看起来平常，但最容易诱导画像主体掉进他自己的误区。
输出JSON: {"question":"题干","reference_answer":"参考答案要点(2-4句)","trick":"诱饵设计意图(一句话)"}
题目必须能从给定讲义内容推出，不要超纲。只输出JSON。"""

SYS_PLAIN = """你是大学课程命题老师。就给定考点出一道标准简答题。
输出JSON: {"question":"题干","reference_answer":"参考答案要点(2-4句)","trick":""}
题目必须能从给定讲义内容推出，不要超纲。只输出JSON。"""

SYS_TWIN = """你就是下面这份画像描述的学生本人。请以第一人称作答这道题：
按画像中你的认知误区、学习习惯和信心风格自然作答（有误区就自然掉进去，不要故意答对或答错）。
输出JSON: {"answer":"你的作答","confidence":1到5的自报把握度}
只输出JSON。"""

SYS_REVEAL = """你是揭晓判官。一场"学生 vs 他的数字孪生"对战刚结束。
对比双方作答与参考答案，分析：(1)孪生为何这样答（是否复现了画像误区）；
(2)这场偏差暴露了系统对该学生认知建模的哪点盲区。
输出JSON: {"analysis":"2-3句偏差分析","insight":"给该学生的一句话洞察"}
只输出JSON。"""


def _brief(sid: str) -> tuple[str, bool]:
    """汇总该生的答题历史与掌握度简报。返回 (简报文本, 是否冷启动)。"""
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    cmap = {c.cid: c for c in storage.load_concepts()}
    if not answers and not mastery:
        names = "、".join(c.name for c in list(cmap.values())[:5]) or "（暂无考点）"
        return (f"该生还没有任何答题记录。考点列表：{names}。", True)
    ans_lines = [f"- {a.cid}({cmap[a.cid].name if a.cid in cmap else a.cid}):"
                 f"{'答对' if a.correct else '答错'}" for a in answers[-30:]]
    mas_lines = [f"- {cid}:{m.p:.2f}(对{m.n_correct}/错{m.n_wrong})" for cid, m in mastery.items()]
    return ("答题记录(最近30条)：\n" + "\n".join(ans_lines) +
            "\n各考点掌握度：\n" + "\n".join(mas_lines), False)


def build_profile(sid: str) -> TwinProfile:
    """认知建模师：构建（或进化重建）学生数字孪生画像。LLM 不可用时规则降级。"""
    brief, cold = _brief(sid)
    user = f"学生ID:{sid}\n数据简报:\n{brief}"
    try:
        data = get_llm().chat_json(SYS_PROFILE, user)
        mis = [TwinMisconception(cid=str(m.get("cid", "")), name=str(m.get("name", "")),
                                 pattern=str(m.get("pattern", "")))
               for m in data.get("misconceptions", []) if isinstance(m, dict)]
        bias = data.get("confidence_bias", "balanced")
        bias = bias if bias in ("overconfident", "underconfident", "balanced") else "balanced"
        return TwinProfile(sid=sid, persona=str(data.get("persona", "画像生成中")),
                           misconceptions=mis, confidence_bias=bias,
                           updated_at=storage.now_iso())
    except Exception:
        # 降级：按掌握度规则生成通用画像
        mastery = storage.load_mastery(sid)
        cmap = {c.cid: c for c in storage.load_concepts()}
        weakest = sorted(mastery.values(), key=lambda m: m.p)[:3]
        mis = [TwinMisconception(cid=m.cid,
                                 name=cmap[m.cid].name if m.cid in cmap else m.cid,
                                 pattern="掌握度偏低，概念似懂非懂，易混淆相近概念")
               for m in weakest]
        persona = ("数据不足，基于先验：该生答题记录较少，" +
                   ("在" + "、".join(m.name for m in mis) + "上掌握薄弱。" if mis else "尚无明确薄弱点。"))
        return TwinProfile(sid=sid, persona=persona, misconceptions=mis,
                           confidence_bias="balanced", updated_at=storage.now_iso())


def _one_question(profile: TwinProfile, concept: Concept, material: str, bait: bool) -> dict:
    sys_p = SYS_BAIT if bait else SYS_PLAIN
    mis_text = "; ".join(f"{m.name}:{m.pattern}" for m in profile.misconceptions) or "暂无明确误区"
    user = (f"孪生画像:{profile.persona}\n认知误区:{mis_text}\n信心偏差:{profile.confidence_bias}\n"
            f"目标考点:{concept.name}({concept.cid})：{concept.summary}\n"
            f"讲义原文（节选）：\n{material[:2500]}")
    return get_llm().chat_json(sys_p, user)


def make_bait(profile: TwinProfile, concept: Concept, material: str) -> dict | None:
    """红方出题：诱饵题 -> 反幻觉校验；重试2次不过则降级普通题；全失败返回 None。"""
    item: QuizItem | None = None
    trick = ""
    for attempt in range(3):
        bait = attempt < 2  # 前两次出诱饵题，最后降级为普通题
        try:
            data = _one_question(profile, concept, material, bait)
        except Exception:
            return None  # LLM 不可用，上层返回友好提示
        cand = QuizItem(qid=f"tq_{concept.cid}_{storage.now_iso().replace(':', '').replace(' ', '').replace('-', '')}",
                        cid=concept.cid, tier="battle",
                        question=data.get("question", ""),
                        reference_answer=data.get("reference_answer", ""))
        if not cand.question or not cand.reference_answer:
            continue
        try:
            (cand,) = verify_agent.verify_quiz([cand], [concept], material)
        except Exception:
            cand.grounded = False
        if cand.grounded:
            item, trick = cand, data.get("trick", "")
            break
        item, trick = cand, data.get("trick", "")  # 保留最后一次候选
    if item is None:
        return None
    return {"item": item, "trick": trick}


def twin_answer(profile: TwinProfile, question: str) -> dict:
    """孪生盲答：签名刻意不含学生答案与参考答案——双盲由接口物理保证。"""
    mis_text = "; ".join(f"{m.name}:{m.pattern}" for m in profile.misconceptions) or "暂无明确误区"
    user = f"画像:{profile.persona}\n你的认知误区:{mis_text}\n信心偏差:{profile.confidence_bias}\n题目:{question}"
    try:
        data = get_llm().chat_json(SYS_TWIN, user)
        return {"answer": str(data.get("answer", "")), "confidence": int(data.get("confidence", 3))}
    except Exception:
        # 降级：按画像误区拼模板作答
        pat = profile.misconceptions[0].pattern if profile.misconceptions else "概念记得不太牢"
        return {"answer": f"我印象里这道题应该就是……具体有点说不清（{pat}）", "confidence": 3}


def reveal_analysis(question: str, reference: str, s_ans: str, t_ans: str,
                    profile: TwinProfile) -> dict:
    """揭晓判官：偏差分析。LLM 不可用返回固定文案。"""
    user = (f"题目:{question}\n参考答案:{reference}\n学生作答:{s_ans or '(空)'}\n"
            f"孪生作答:{t_ans or '(空)'}\n孪生画像:{profile.persona}")
    try:
        return get_llm().chat_json(SYS_REVEAL, user)
    except Exception:
        return {"analysis": "（LLM不可用）对比双方作答与参考答案即可看出：孪生是否复现了你的误区，"
                            "以及系统对你的建模还有哪里不准。",
                "insight": "孪生答错的题，往往正是你真正掌握的地方。"}


# ---------- 纯函数（可单测） ----------
def judge_winner(s_correct: bool, t_correct: bool) -> str:
    """胜负判定：学生对&孪生错=student；学生错&孪生对=twin；同对同错=tie。"""
    if s_correct and not t_correct:
        return "student"
    if t_correct and not s_correct:
        return "twin"
    return "tie"


def calc_similarity(history: list[TwinBattle]) -> dict:
    """孪生相似度 = 孪生与学生结果一致率；sim_trend 为近5场滑窗逐点趋势。"""
    if not history:
        return {"sim": -1.0, "sim_trend": []}
    same = sum(1 for b in history if b.s_correct == b.t_correct)
    trend = []
    for i in range(1, len(history) + 1):
        win = history[max(0, i - 5):i]
        w = sum(1 for b in win if b.s_correct == b.t_correct)
        trend.append({"ts": win[-1].ts, "sim": round(w / len(win), 2)})
    return {"sim": round(same / len(history), 2), "sim_trend": trend[-5:]}
