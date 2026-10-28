"""生成后硬校验（反编造第三层：从提示词约束升级为代码级约束）。

系统提示里的"禁止编造"只是口头约束，模型仍可能写出没依据的数字或接口。
这里在答案生成后做一遍可核验声明抽取——数字（含百分数/毫秒等单位）与 /api/ 接口
路径——逐个回查证据（检索片段、联网摘要、工具返回）。证据里找不到的列为"未验证"，
附在答案末尾提示，而不是静默丢弃（静默删字会让答案读不通）。

同一份逻辑同时供 AI 助教（生产路径）与生成质量抽检（评测路径）使用，
保证评测测的就是线上真实执行的检查。
"""
from __future__ import annotations
import re

_API_RE = re.compile(r"/api/[A-Za-z0-9_/\-]+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_UNIT_RE = re.compile(r"\s*(?:%|ms|个|项|道|条|页|天|倍|分|次|种|人|小时|秒)")


def _claim_numbers(text: str) -> set[str]:
    """抽取可核验的数字声明：带单位的数字，或位数 ≥ 2 的数字。
    裸个位数（列举序号 1./2./3.）不计入，避免把结构化排版误判为编造。"""
    out: set[str] = set()
    for m in _NUM_RE.finditer(text):
        tok = m.group(0)
        tail = text[m.end():m.end() + 3]
        if _UNIT_RE.match(tail) or len(tok.replace(".", "")) >= 2:
            out.add(tok)
    return out


def _num_supported(n: str, ev_nums: set[str]) -> bool:
    """数字是否被证据支撑：精确命中，或落在舍入/量级容差内（0.8 与 0.82 视为同一事实）。"""
    if n in ev_nums:
        return True
    try:
        v = float(n)
    except ValueError:
        return False
    dec = len(n.split(".")[1]) if "." in n else 0
    for e in ev_nums:
        try:
            ev = float(e)
        except ValueError:
            continue
        if abs(ev - v) <= max(0.05, abs(v) * 0.01):
            return True
        if round(ev, dec) == round(v, dec):
            return True
    return False


def _evidence_text(cites=(), web=(), steps=(), extra: str = "") -> str:
    parts = [str(extra)]
    parts += [c.get("text", "") for c in cites if isinstance(c, dict)]
    parts += [f"{w.get('title', '')} {w.get('text', '')} {w.get('url', '')}"
              for w in web if isinstance(w, dict)]
    parts += [str(s.get("result") or "") for s in steps if isinstance(s, dict)]
    return " ".join(parts)


def check(answer: str, cites=(), web=(), steps=(), question: str = "",
          extra: str = "", exclude=()) -> dict:
    """核对答案中的数字与接口路径能否在证据中落地。

    question：学生原问题——问题里已给出的数字/路径若被复述，不算编造。
    extra：系统提示等"产品事实"文本，其中的数字（如 15 天报表周期）可被引用。
    exclude：额外要豁免的字符串。
    返回 {ok, unverified, note, n_claims}。
    """
    ev = _evidence_text(cites, web, steps, extra)
    ev_nums = set(_NUM_RE.findall(ev))
    ev_apis = {p.rstrip("/") for p in _API_RE.findall(ev)}
    exclude = {str(x) for x in exclude}
    exclude |= _claim_numbers(question) | {p.rstrip("/") for p in _API_RE.findall(question)}

    bad_nums = sorted(n for n in _claim_numbers(answer)
                      if n not in exclude and not _num_supported(n, ev_nums))
    bad_apis = sorted(a for a in {p.rstrip("/") for p in _API_RE.findall(answer)}
                      if a not in exclude and a not in ev_apis)
    unverified = bad_nums + bad_apis
    note = ("（未在证据中找到依据，请核实：" + "、".join(unverified[:4]) + "）"
            if unverified else "")
    return {"ok": not unverified, "unverified": unverified, "note": note,
            "n_claims": len(_claim_numbers(answer)) + len(set(_API_RE.findall(answer)))}
