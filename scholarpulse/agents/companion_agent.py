"""💬 AI 助教（LangChain ReAct 智能体）+ ✍️ 考官开局。
助教不走 if/else 意图路由：把 8 个真实工具（项目知识库 RAG、课件 RAG、6 个学情工具）
暴露给模型，由 create_react_agent 自主决定查什么、答什么，前端展示工具轨迹与引用。
考官开局：学生进自测面板时，考官智能体查 Elo/薄弱/到期复习后下发今日训练策略。"""
from __future__ import annotations
import os

from langchain_core.tools import tool
from ..lc_model import RouterChatModel
from ..core.rag_retriever import HybridRetriever, chunk_text, get_student_retriever
from ..core.grounding import check as ground_check
from .web_search import web_search as _web_search
from .tool_calling_agent import query_weak_points, cognitive_diagnose, generate_report
from .diagnosis_agents import due_review_count, _run_react


@tool
def mastery_overview(sid: str) -> str:
    """一句话数据包：平均掌握度、最薄弱考点、到期复习数、累计错题数。"""
    from .. import storage
    from ..mastery import avg_mastery
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(sid)
    weak = sorted(concepts, key=lambda c: mastery[c.cid].p if c.cid in mastery else 1)[:2]
    n_wrong = sum(1 for a in storage.load_answers(sid) if not a.correct)
    return (f"平均掌握 {avg_mastery(mastery):.2f}；最薄弱 "
            f"{ '、'.join(name_of.get(c.cid, c.cid) for c in weak) or '无' }；"
            f"今日到期复习 {due_review_count.invoke({'sid': sid})['n_due']} 项；累计错题 {n_wrong} 道")


@tool
def list_wrong_book(sid: str) -> str:
    """查错题本：按错误次数排名的考点与最近错题原文。"""
    from .. import storage
    name_of = {c.cid: c.name for c in storage.load_concepts()}
    acc: dict[str, int] = {}
    last_q: dict[str, str] = {}
    for a in storage.load_answers(sid):
        if not a.correct:
            acc[a.cid] = acc.get(a.cid, 0) + 1
            if a.question:
                last_q[a.cid] = a.question[:40]
    rows = sorted(acc.items(), key=lambda x: -x[1])[:4]
    return "；".join(f"{name_of.get(k, k)}×{v}（最近错：{last_q.get(k, '')}）" for k, v in rows) or "错题本为空"


# ---------- 项目知识库（覆盖全部功能，随文件变化失效） ----------
_GUIDE_CACHE: dict = {}
_GUIDE_PATH = os.path.join("data", "assistant_kb", "project_guide.md")
# 最近一次课件检索的证据块：sid -> citations（供前端展示引用，答完即取走）
_LAST_CITES: dict[str, list] = {}
# 最近一次联网检索结果（仅在本地无答案时产生）
_LAST_WEB: list = []


def _bigrams(text: str) -> set[str]:
    """中文按二元字组、英文按词切分，用于证据覆盖率判定。"""
    cjk = "".join(ch if "\u4e00" <= ch <= "\u9fff" else " " for ch in text)
    out: set[str] = set()
    for seg in cjk.split():
        out.update(seg[i:i + 2] for i in range(len(seg) - 1))
        if len(seg) == 1:
            out.add(seg)
    for w in text.replace("\n", " ").split():
        if len(w) >= 2 and w.isascii():
            out.add(w.lower())
    return out


def _coverage(question: str, text: str) -> float:
    q = _bigrams(question)
    if not q:
        return 0.0
    return len(q & _bigrams(text)) / len(q)


def _guide_retriever():
    """项目功能地图检索器；按文件 mtime+size 缓存。"""
    if not os.path.exists(_GUIDE_PATH):
        return None
    st = os.stat(_GUIDE_PATH)
    key = (st.st_mtime_ns, st.st_size)
    hit = _GUIDE_CACHE.get("r")
    if hit and hit[0] == key:
        return hit[1]
    text = open(_GUIDE_PATH, encoding="utf-8", errors="ignore").read()
    chunks = [c for c in chunk_text(text) if len(c.strip()) >= 20]
    r = HybridRetriever(chunks) if chunks else None
    _GUIDE_CACHE["r"] = (key, r)
    return r


@tool
def search_project_guide(query: str) -> str:
    """检索本项目功能地图：页面入口、功能操作、算法原理、数据口径、接口。问“某功能在哪/怎么用/是什么”时先调它。"""
    r = _guide_retriever()
    if r is None:
        return "项目知识库未就绪（缺少 data/assistant_kb/project_guide.md）"
    hits = r.search(query, top_k=3)
    if not hits:
        return "知识库中没有匹配条目"
    return "\n".join(f"[资料{i}] {r.chunks[h['chunk_id']][:320]}"
                     for i, h in enumerate(hits, 1))


@tool
def search_course_material(sid: str, query: str) -> str:
    """检索该生课件原文（三通道混合检索 + RRF），返回带编号的证据片段。课件知识、定义、公式类问题必须先调它。"""
    r, meta = get_student_retriever(sid)
    if r is None:
        _LAST_CITES.pop(sid, None)
        return f"该生没有可检索的课件：{meta.get('msg', '无课件')}"
    hits = r.search(query, top_k=4)
    if not hits:
        _LAST_CITES.pop(sid, None)
        return "课件中没有找到相关内容。"
    cites = [{
        "n": i,
        "chunk_id": h["chunk_id"],
        "rrf": h["rrf"],
        "bm25": round(h["sparse"]["bm25"], 3),
        "text": r.chunks[h["chunk_id"]][:300],
    } for i, h in enumerate(hits, 1)]
    best = hits[0]
    if best["sparse"]["bm25"] <= 0 and _coverage(query, cites[0]["text"]) < 0.3:
        _LAST_CITES.pop(sid, None)
        return "课件中没有与该问题相关的内容（检索无有效信号），不得基于常识猜测。"
    _LAST_CITES[sid] = cites
    return "\n".join(f"[片段{c['n']}] {c['text']}" for c in cites)


@tool
def search_web(query: str) -> str:
    """联网兜底检索：仅当本地知识库与课件都明确没有相关内容时才调用，返回标题、摘要与链接。"""
    res = _web_search(query, 5)
    _LAST_WEB.clear()
    if not res:
        return "联网检索不可用（网络不通或搜索源无结果）。"
    _LAST_WEB.extend(res)
    return "\n".join(
        f"[网络{i}] {x['title']}\n{x['snippet']}\n链接: {x['url']}"
        for i, x in enumerate(res, 1)
    )


COMPANION_TOOLS = [search_project_guide, search_course_material, search_web,
                   mastery_overview, query_weak_points, list_wrong_book,
                   cognitive_diagnose, due_review_count, generate_report]

COMPANION_SYS = ("你是 ScholarPulse 的 AI 助教，覆盖本项目全部内容。可调用工具："
                 "search_project_guide（项目功能地图：页面入口、操作、算法、口径）、"
                 "search_course_material（该生课件原文，带片段编号）、"
                 "search_web（联网兜底）、"
                 "mastery_overview（掌握度）、query_weak_points（薄弱点）、"
                 "list_wrong_book（错题本）、cognitive_diagnose（断层问诊）、"
                 "due_review_count（到期复习）、generate_report（15天报告）。"
                 "路由规则：问功能/入口/怎么用/算法/数据口径，先调 search_project_guide；"
                 "问课件知识/定义/公式，先调 search_course_material；"
                 "问个人学情，调对应学情工具。"
                 "RAG 约束：只依据工具返回的证据回答；引用课件时必须标注 [片段n]；"
                 "证据不足时直接说『资料中没有』，禁止编造入口、参数、数字。"
                 "同一工具对同一问题最多调用一次，不要重复检索。"
                 "联网规则：只有本地工具明确返回『没有找到/资料中没有/无有效信号』时，"
                 "才调用 search_web 兜底；网络内容必须给出链接并注明来自网络，"
                 "不得把网络摘要伪装成课件原文或项目内部数据。"
                 "回答≤160字，先给结论，再给依据或下一步动作。")

EXAMINER_SYS = ("你是考官，学生即将进入今日自测。可调用工具：mastery_overview、"
                "query_weak_points、list_wrong_book、due_review_count。"
                "先调 2-3 个工具看清学情，再下发≤80字的今日训练策略："
                "建议主攻考点、难度层（基础/进阶/迁移）、要先复习什么。考官口吻，严格但鼓劲。")

# 联网兜底重答：本地检索无信号时由代码强制触发，不依赖模型自己决定调不调工具
_WEB_FALLBACK_SYS = ("你是课程助教。本地课件与项目知识库没有相关内容，已自动转为联网检索。"
                     "只能依据给出的网络结果回答；结论后必须给出来源链接（形如 链接：https://…）；"
                     "不得添加网络结果里没有的事实或数字；网络结果也不足时明确说明。≤160字。")
_REFUSE_HINTS = ("没有", "未找到", "资料中", "无法", "不足", "无有效信号")

EXAMINER_TOOLS = [mastery_overview, query_weak_points, list_wrong_book, due_review_count]


def run_companion(sid: str, message: str) -> dict:
    """AI 助教 ReAct：返回 {answer, steps, citations, web}。"""
    model = RouterChatModel(temperature=0.6, max_tokens=400)
    try:
        steps, answer = _run_react(model, COMPANION_TOOLS, COMPANION_SYS,
                                   f"学生{sid}说：{message}")
    except Exception as e:
        _LAST_CITES.pop(sid, None)
        _LAST_WEB.clear()
        return {"steps": [], "answer": f"助教离线：{e}", "citations": [], "web": []}

    cites = _LAST_CITES.pop(sid, [])

    # 代码级联网兜底：本地确实没有证据（无课件片段、无联网结果）且答案落成"找不到"时，
    # 强制联网重答一次。已有证据时不触发，避免把本来有据的部分答案覆盖掉。
    if (not cites and not _LAST_WEB
            and not any(s.get("tool") == "search_web" for s in steps)
            and any(k in answer for k in _REFUSE_HINTS)):
        hits = _web_search(message, 5)
        if hits:
            ctx = "\n".join(f"[网络{i}] {x['title']}：{x['snippet']}（链接：{x['url']}）"
                            for i, x in enumerate(hits, 1))
            try:
                from ..llm import get_llm
                answer = get_llm().chat(_WEB_FALLBACK_SYS,
                                        f"网络结果：\n{ctx}\n\n问题：{message}",
                                        temperature=0.3, max_tokens=360)
                steps.append({"tool": "search_web", "args": {"query": message},
                              "result": f"本地无信号 → 联网兜底命中 {len(hits)} 条"})
                _LAST_WEB.clear()
                _LAST_WEB.extend(hits)
            except Exception:
                pass

    web = [{"n": i, "title": x["title"], "url": x["url"], "text": x["snippet"]}
           for i, x in enumerate(_LAST_WEB, 1)]
    _LAST_WEB.clear()

    # 生成后硬校验：数字与接口路径必须能在证据里落地，无依据的显式标注（第三层反编造）
    g = ground_check(answer, cites, web, steps, question=message, extra=COMPANION_SYS)
    if g["note"]:
        answer = answer + " " + g["note"]

    return {"steps": steps, "answer": answer, "citations": cites, "web": web,
            "grounding": g}


def run_examiner(sid: str) -> dict:
    """考官开局：返回 {answer, steps}。"""
    model = RouterChatModel(temperature=0.5, max_tokens=350)
    try:
        steps, answer = _run_react(model, EXAMINER_TOOLS, EXAMINER_SYS,
                                   f"学生{sid}来了，给我今日训练策略。")
    except Exception as e:
        return {"steps": [], "answer": f"考官未到场：{e}"}
    return {"steps": steps, "answer": answer}
