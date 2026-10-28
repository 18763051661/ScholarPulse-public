"""联网检索兜底：本地知识库与课件都没有答案时才使用。

按 Bing → DuckDuckGo 顺序尝试，任一成功即返回；
不引入第三方 HTML 解析依赖，网络不通时明确返回空列表，由调用方告知用户。
"""
from __future__ import annotations

import html as _html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _clean(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _normalize_url(u: str) -> str:
    u = _html.unescape(u or "")
    if u.startswith("//"):
        u = "https:" + u
    if "duckduckgo.com/l/" in u:
        qs = parse_qs(urlparse(u).query)
        if qs.get("uddg"):
            u = unquote(qs["uddg"][0])
    return u


def _bing(query: str, n: int) -> list[dict]:
    r = httpx.get("https://www.bing.com/search",
                  params={"q": query, "setlang": "zh-CN", "count": n},
                  headers={"User-Agent": UA}, timeout=8, follow_redirects=True)
    r.raise_for_status()
    out = []
    for block in re.findall(r'<li class="b_algo".*?</li>', r.text, re.S)[:n]:
        m = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        sn = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        out.append({
            "title": _clean(m.group(2)),
            "url": _normalize_url(m.group(1)),
            "snippet": _clean(sn.group(1)) if sn else "",
        })
    return out


def _ddg(query: str, n: int) -> list[dict]:
    r = httpx.post("https://html.duckduckgo.com/html/", data={"q": query},
                   headers={"User-Agent": UA}, timeout=8, follow_redirects=True)
    r.raise_for_status()
    titles = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                        r.text, re.S)
    snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
    out = []
    for i, (u, t) in enumerate(titles[:n]):
        out.append({
            "title": _clean(t),
            "url": _normalize_url(u),
            "snippet": _clean(snippets[i]) if i < len(snippets) else "",
        })
    return out


def web_search(query: str, n: int = 5) -> list[dict]:
    """返回 [{title, url, snippet}]；两个搜索源都失败时返回空列表。"""
    q = _strip_question(query) or query
    for fn in (_bing, _ddg):
        try:
            res = [x for x in fn(q, n) if x["url"].startswith("http")]
            if res:
                return res
        except Exception:
            continue
    return []


_QUESTION_WORDS = re.compile(
    r"(什么是|什么叫做|何为|是什么意思|是什么|为什么|怎么样|怎么|如何|有哪些|"
    r"请问|解释一下|解释|一下|的定义)")


def _strip_question(q: str) -> str:
    """去掉疑问词与标点，提高搜索源对中文问句的命中率。"""
    q = re.sub(r"[？?。！!，,]", " ", q or "")
    q = _QUESTION_WORDS.sub(" ", q)
    return " ".join(q.split())
