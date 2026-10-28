"""创意内容生成工坊：短剧大纲→扩写→润色→双主持脚本→后期清单→多格式导出+视频。

对标需求：创意内容生成（自定义生成，不套模板）；双主持脚本 + TTS + 后期 + 短剧视频。

设计：
  - 五段流水线全部走大模型 API（故障转移链），支持自定义要求 brief
    （角色/风格/集数/时长/特殊设定）注入每一段的提示词
  - 不提供固定模板降级：生成失败返回 error 让前端提示重试（拒绝"套模板"假生成）
  - TTS 适配：dual_host 返回逐句 lines（甲/乙交替），前端用 Web Speech API 双声线朗读；
    speak_text 输出净化后的朗读文本
  - 短剧视频由前端 Canvas 动画 + MediaRecorder + AudioContext 合成 BGM 直接产出
    WebM（无需任何服务端依赖）；本模块提供 split_scenes 场景切分纯函数供其使用
  - 导出 md/html/txt 三格式，落盘 data/exports/
"""
from __future__ import annotations
import os
import re
from datetime import datetime

from .report_exporter import EXPORT_DIR

OUTLINE_SYS = (
    "你是校园短剧策划。基于主题与自定义要求输出一份 4 幕短剧大纲（起承转合），"
    "每幕一行，格式：『第X幕·标题：剧情概要（2 句内）』。必须吸收自定义要求中的"
    "角色、风格、设定；只输出 4 行大纲，不要输出任何解释。")
EXPAND_SYS = (
    "你是短剧编剧。把大纲扩写为一部完整长篇剧本（将合成为 5 分钟真人风短剧视频）："
    "共 8-12 场，每场以『【场景X · 地点】』开头（地点如 教室/图书馆/操场/宿舍/天台）；"
    "每场含 12-20 行对白，每行格式严格为『角色名：台词』；"
    "全剧固定 2-3 个主要角色（名字 2-3 字、全剧一致），可穿插少量『旁白：…』行；"
    "对白口语化、每行 6-20 字，有冲突、误会、反转与温暖结局。"
    "严格遵守自定义要求。直接输出剧本，不要任何解说。")
POLISH_SYS = (
    "你是文学润色编辑。在不改变剧情与结构的前提下润色剧本：对白更口语生动、"
    "描写更有画面感、删掉冗词。直接输出润色后的完整剧本，不要任何解说。")
HOST_SYS = (
    "你是双主持栏目编导。把剧本改编成两位主持人（甲/乙）的口播对话脚本："
    "两人轮流输出，每人每次 1-3 句，用『甲：』『乙：』开头逐行输出，"
    "开头有栏目式开场白，结尾有互动引导，内容必须紧扣剧本剧情。")
POST_SYS = (
    "你是短视频后期导演。基于剧本输出后期制作清单：1) 分镜建议（4-6 个镜头，"
    "每个一行：景别+画面+时长）；2) BGM 建议（分段情绪）；3) 字幕/花字要点；"
    "4) 转场与音效提示。分四节输出，每节 2-4 行。")


def _llm():
    from ..llm import get_llm
    return get_llm()


def _brief_line(brief: str) -> str:
    brief = (brief or "").strip()
    return f"\n自定义要求（必须遵守）：{brief}" if brief else ""


def _fail(stage: str, e: Exception) -> dict:
    return {"stage": stage, "error": f"{stage} 生成失败：{e}（大模型链路不可用，请稍后重试）"}


# ---------- TTS / 视频适配（纯函数，可离线测试） ----------

def speak_text(text: str, limit: int = 600) -> str:
    """净化成适合 Web Speech API 朗读的文本：去 markdown/舞台提示符号并限长。"""
    t = re.sub(r"[#*`>\[\]（）()【】]|（[^）]*）|\([^)]*\)", "", text or "")
    t = re.sub(r"^(甲|乙)[:：]\s*", "", t, flags=re.M)   # 朗读时不念"甲/乙"
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def _split_host_lines(text: str) -> list[dict]:
    """把『甲：…』『乙：…』逐行脚本解析为 [{host, text}]；无前缀行并入上一条。"""
    lines: list[dict] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^(甲|乙)[:：]\s*(.+)$", s)
        if m:
            lines.append({"host": m.group(1), "text": m.group(2).strip()})
        elif lines:
            lines[-1]["text"] += " " + s
        else:
            lines.append({"host": "甲", "text": s})
    return lines


def split_scenes(script: str, max_scenes: int = 6) -> list[dict]:
    """把剧本切成视频场景（纯函数）：优先按【场景…】切块，否则按空行分段。

    返回 [{"title": 场景标题, "lines": [台词/动作行]}]，供前端 Canvas 视频渲染。
    """
    text = (script or "").strip()
    if not text:
        return []
    blocks: list[str]
    if "【" in text and "】" in text:
        blocks, cur = [], None
        for raw in text.splitlines():
            s = raw.strip()
            if s.startswith("【") and "】" in s:
                if cur:
                    blocks.append("\n".join(cur))
                cur = [s]
            elif cur is not None and s:
                cur.append(s)
        if cur:
            blocks.append("\n".join(cur))
    else:
        blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) == 1:
            blocks = [b for b in text.splitlines() if b.strip()]
    scenes = []
    for b in blocks[:max_scenes]:
        ls = [x.strip() for x in b.splitlines() if x.strip()]
        title = ls[0]
        if title.startswith("【") and "】" in title:
            title = title.split("】")[0].strip("【")
        scenes.append({"title": title[:24], "lines": ls[1:][:6] or [title]})
    return scenes


# ---------- 五段流水线（全部大模型生成，失败返回 error） ----------

def gen_outline(topic: str, genre: str = "校园短剧", brief: str = "") -> dict:
    """第 1 段：短剧大纲（4 幕起承转合，吸收自定义要求）。"""
    topic = (topic or "").strip() or "大学生的一周"
    try:
        outline = _llm().chat(OUTLINE_SYS, f"题材：{genre}；主题：{topic}{_brief_line(brief)}",
                              temperature=0.85, max_tokens=500)
        return {"stage": "outline", "topic": topic, "text": outline.strip(), "mode": "llm"}
    except Exception as e:
        return _fail("outline", e)


def expand_script(topic: str, outline: str, brief: str = "") -> dict:
    """第 2 段：大纲 → 完整剧本（遵守自定义要求）。"""
    try:
        script = _llm().chat(EXPAND_SYS, f"主题：{topic}{_brief_line(brief)}\n大纲：\n{outline}",
                             temperature=0.85, max_tokens=3800)
        return {"stage": "expand", "text": script.strip(), "mode": "llm"}
    except Exception as e:
        return _fail("expand", e)


def polish_script(topic: str, script: str, brief: str = "") -> dict:
    """第 3 段：剧本润色。"""
    try:
        polished = _llm().chat(POLISH_SYS, f"主题：{topic}{_brief_line(brief)}\n剧本：\n{script}",
                               temperature=0.7, max_tokens=1100)
        return {"stage": "polish", "text": polished.strip(), "mode": "llm"}
    except Exception as e:
        return _fail("polish", e)


def dual_host_script(topic: str, script: str, brief: str = "") -> dict:
    """第 4 段：双主持口播脚本（甲/乙逐行 + TTS 逐句 lines）。"""
    try:
        host = _llm().chat(HOST_SYS, f"主题：{topic}{_brief_line(brief)}\n剧本：\n{script}",
                           temperature=0.75, max_tokens=900)
        return {"stage": "host", "text": host.strip(), "lines": _split_host_lines(host),
                "speak": speak_text(host), "mode": "llm"}
    except Exception as e:
        return _fail("host", e)


def post_production(topic: str, script: str, brief: str = "") -> dict:
    """第 5 段：后期制作清单（分镜/BGM/字幕/转场）。"""
    try:
        post = _llm().chat(POST_SYS, f"主题：{topic}{_brief_line(brief)}\n剧本：\n{script[:1500]}",
                           temperature=0.7, max_tokens=800)
        return {"stage": "post", "text": post.strip(), "mode": "llm"}
    except Exception as e:
        return _fail("post", e)


# ---------- 多格式导出 ----------

def _sections(outline: str, script: str, host: str, post: str) -> list[tuple[str, str]]:
    return [("一、短剧大纲", outline), ("二、完整剧本", script),
            ("三、双主持口播脚本", host), ("四、后期制作清单", post)]


def export_creative(title: str, outline: str, script: str,
                    host: str, post: str, fmt: str = "md") -> dict:
    """导出创意脚本包：fmt ∈ md/html/txt，落盘 data/exports/。"""
    title = (title or "校园短剧").strip()[:40]
    fmt = fmt.lower()
    os.makedirs(EXPORT_DIR, exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fff-]", "_", title)
    path = os.path.join(EXPORT_DIR, f"creative_{safe}.{fmt}")
    if fmt == "md":
        body = f"# 创意脚本包 · {title}\n> 生成时间：{datetime.now():%Y-%m-%d %H:%M}\n\n"
        body += "\n\n".join(f"## {h}\n\n{c}" for h, c in _sections(outline, script, host, post))
        open(path, "w", encoding="utf-8").write(body)
    elif fmt == "txt":
        body = f"创意脚本包 · {title}\n" + "=" * 30 + "\n\n"
        body += "\n\n".join(f"{h}\n{'-' * 20}\n{c}" for h, c in _sections(outline, script, host, post))
        open(path, "w", encoding="utf-8").write(body)
    elif fmt == "html":
        body = "".join(f"<h2>{h}</h2><pre class='sec'>{c}</pre>"
                       for h, c in _sections(outline, script, host, post))
        open(path, "w", encoding="utf-8").write(
            "<!doctype html><meta charset='utf-8'><title>" + title + "</title>"
            "<style>body{font-family:'Microsoft YaHei',sans-serif;max-width:760px;margin:40px auto;"
            "color:#1a1a2e}h1{color:#4a1a7a}h2{color:#6b2da0;border-left:4px solid #8e44ad;padding-left:8px}"
            ".sec{white-space:pre-wrap;font-family:inherit;background:#faf7fe;padding:12px;border-radius:8px}</style>"
            f"<h1>创意脚本包 · {title}</h1>{body}")
    else:
        return {"error": f"不支持的格式：{fmt}（可选 md/html/txt）"}
    return {"title": title, "fmt": fmt, "path": path, "size": os.path.getsize(path)}
