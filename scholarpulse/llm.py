"""
多供应商故障转移 LLM 客户端。
- 按 settings.providers 顺序调用 (DeepSeek -> Qwen -> Doubao)
- 任一供应商超时/HTTP错误/解析失败，自动切下一家
- 全部失败时抛出最后一个异常（上层可决定是否降级 DEMO）
所有三家都是 OpenAI Chat Completions 兼容协议。
"""
from __future__ import annotations
import json
import os
import time
from urllib.parse import urlparse
import httpx
from .config import settings, Provider

_STATE_PATH = os.path.join("data", "llm_state.json")


def _est_tokens(text: str) -> int:
    """无 usage 字段时的粗略估算：中文约 1.5 字/ token。"""
    return max(1, round(len(text or "") / 1.5))


def _mask(key: str) -> str:
    return (key[:5] + "****" + key[-4:]) if len(key) > 12 else "****"


def _host(base: str) -> str:
    u = urlparse(base)
    return f"{u.scheme}://{u.netloc}" if u.netloc else base


class LLMClient:
    def __init__(self) -> None:
        self.providers = settings.providers
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        # 记录最近一次成功用的是哪家，便于观测
        self.last_provider: str | None = None
        # 运行期选择的供应商与 token 记账，落盘在 data/llm_state.json
        self._state = self._load_state()
        # 每家供应商的调用统计（成功/失败次数），随状态落盘，重启不清零
        saved = self._state.get("stats") or {}
        self.stats: dict[str, dict[str, int]] = {
            p.name: {"ok": int(saved.get(p.name, {}).get("ok", 0)),
                     "fail": int(saved.get(p.name, {}).get("fail", 0))}
            for p in self.providers}
        for n, s in saved.items():  # 运行期添加过的历史供应商
            self.stats.setdefault(n, {"ok": int(s.get("ok", 0)),
                                      "fail": int(s.get("fail", 0))})
        self.active: str = self._state.get("active") or (
            self.providers[0].name if self.providers else "")
        self._apply_active()

    # ---------- 配置：切换 / 展示 ----------
    def _load_state(self) -> dict:
        try:
            d = json.load(open(_STATE_PATH, encoding="utf-8"))
            d.setdefault("usage", {"total": {}, "daily": {}})
            return d
        except Exception:
            return {"active": None, "usage": {"total": {}, "daily": {}}}

    def _save(self) -> None:
        os.makedirs("data", exist_ok=True)
        try:
            json.dump(self._state, open(_STATE_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False)
        except Exception:
            pass

    def _apply_active(self) -> None:
        """选中的供应商提到链首，其余回到配置里的原始顺序。"""
        if not self.active:
            return
        names = [p.name for p in self.providers]
        order = [self.active] + [n for n in settings.base_names
                                 if n != self.active and n in names]
        order += [n for n in names if n not in order]
        self.providers[:] = [next(p for p in self.providers if p.name == n)
                             for n in order]

    def set_active(self, name: str) -> dict:
        if not any(p.name == name for p in self.providers):
            raise ValueError(f"没有供应商 {name}")
        # 保存前先读盘，避免覆盖其他进程刚写入的 token 记账
        fresh = self._load_state().get("usage") or {}
        if fresh.get("total"):
            self._state["usage"] = fresh
        self.active = name
        self._state["active"] = name
        self._apply_active()
        self._save()
        return self.config()

    def config(self) -> dict:
        """配置信息：模型、端点、密钥掩码、超时、今日与累计用量。"""
        return {
            "active": self.active,
            "last": self.last_provider,
            "chain": [p.name for p in self.providers],
            "providers": [{
                "name": p.name, "model": p.model, "base": _host(p.base),
                "key": _mask(p.key), "timeout_ms": p.timeout_ms,
                "no_thinking": p.no_thinking,
                "runtime": p.name in settings.runtime_names,
                "probe": (self._state.get("probes") or {}).get(p.name),
                "stats": self.stats.get(p.name, {"ok": 0, "fail": 0}),
                "today": self._bucket("daily", p.name),
                "total": self._bucket("total", p.name),
            } for p in self.providers],
            "summary": self.usage(),
        }

    # ---------- token 记账 ----------
    def _bucket(self, kind: str, name: str) -> dict:
        u = self._state["usage"]
        src = u["daily"].get(time.strftime("%Y-%m-%d"), {}) if kind == "daily" else u["total"]
        return dict(src.get(name, {"req": 0, "pt": 0, "ct": 0}))

    def _record(self, name: str, pt: int, ct: int) -> None:
        u = self._state["usage"]
        day = time.strftime("%Y-%m-%d")
        for src in (u["total"], u["daily"].setdefault(day, {})):
            d = src.setdefault(name, {"req": 0, "pt": 0, "ct": 0})
            d["req"] += 1
            d["pt"] += pt
            d["ct"] += ct
        # 只保留最近 14 天，避免状态文件膨胀
        for old in sorted(u["daily"])[:-14]:
            u["daily"].pop(old, None)
        self._state["stats"] = self.stats
        self._save()

    def usage(self) -> dict:
        total = {"req": 0, "pt": 0, "ct": 0}
        for d in self._state["usage"]["total"].values():
            for k in total:
                total[k] += d.get(k, 0)
        total["tokens"] = total["pt"] + total["ct"]
        days = {d: sum(v.get("pt", 0) + v.get("ct", 0) for v in m.values())
                for d, m in self._state["usage"]["daily"].items()}
        return {"total": total, "today": self._bucket("daily", self.active),
                "daily": dict(sorted(days.items())[-7:]),
                "today_tokens": days.get(time.strftime("%Y-%m-%d"), 0)}

    # ---------- 额度（供应商余额接口） ----------
    def quota(self, name: str | None = None) -> list[dict]:
        """查询各供应商余额。DeepSeek 有官方余额接口，其余按本地记账估算。"""
        out = []
        for p in self.providers:
            if name and p.name != name:
                continue
            if "api.deepseek.com" in p.base:
                try:
                    with httpx.Client(timeout=6) as c:
                        r = c.get("https://api.deepseek.com/user/balance",
                                  headers={"Authorization": f"Bearer {p.key}"})
                    r.raise_for_status()
                    info = (r.json().get("balance_infos") or [{}])[0]
                    out.append({"name": p.name, "supported": True,
                                "currency": info.get("currency", "CNY"),
                                "total": info.get("total_balance"),
                                "granted": info.get("granted_balance"),
                                "topped_up": info.get("topped_up_balance")})
                except Exception as e:  # noqa: BLE001
                    out.append({"name": p.name, "supported": True,
                                "error": str(e)[:120]})
            else:
                out.append({"name": p.name, "supported": False,
                            "msg": "该供应商无公开余额接口，按本地 token 记账估算"})
        return out

    def status(self) -> dict:
        """供应商链路与调用统计（前端徽章/成本监控用）。"""
        return {"chain": [p.name for p in self.providers],
                "models": {p.name: p.model for p in self.providers},
                "last": self.last_provider, "stats": self.stats}

    # ---------- 连通性实测：真的发一次请求 ----------
    def probe(self, name: str) -> dict:
        """对指定供应商单独发一次极短请求（不走故障转移），返回真实延迟与回复。"""
        p = next((x for x in self.providers if x.name == name), None)
        if p is None:
            raise ValueError(f"没有供应商 {name}")
        body = {"model": p.model, "temperature": 0, "max_tokens": 16,
                "messages": [{"role": "user", "content": "只回复两个字：在线"}]}
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=p.timeout_ms / 1000) as c:
                r = c.post(p.base, headers={"Authorization": f"Bearer {p.key}",
                                            "Content-Type": "application/json"}, json=body)
            r.raise_for_status()
            d = r.json()
            txt = (d["choices"][0]["message"]["content"] or "").strip()
            ms = round((time.perf_counter() - t0) * 1000)
            u = d.get("usage") or {}
            pt = int(u.get("prompt_tokens") or _est_tokens(body["messages"][0]["content"]))
            ct = int(u.get("completion_tokens") or _est_tokens(txt))
            self.stats.setdefault(name, {"ok": 0, "fail": 0})["ok"] += 1
            self._state.setdefault("probes", {})[name] = {
                "ok": True, "ms": ms, "reply": txt[:40], "tokens": pt + ct,
                "ts": time.strftime("%m-%d %H:%M")}
            self._record(name, pt, ct)
            return {"name": name, "model": p.model, "ok": True, "ms": ms,
                    "reply": txt[:40], "tokens": pt + ct}
        except Exception as e:  # noqa: BLE001
            ms = round((time.perf_counter() - t0) * 1000)
            self.stats.setdefault(name, {"ok": 0, "fail": 0})["fail"] += 1
            self._state.setdefault("probes", {})[name] = {
                "ok": False, "ms": ms, "error": str(e)[:120],
                "ts": time.strftime("%m-%d %H:%M")}
            self._save()
            return {"name": name, "model": p.model, "ok": False, "ms": ms,
                    "error": str(e)[:160]}

    def chat(self, system: str, user: str, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        """同步聊天，返回纯文本 content。自动故障转移。"""
        if not self.providers:
            raise RuntimeError("未配置任何 LLM 供应商（.env 中缺少 KEY）")

        payload_base = {
            "model": None,  # 按供应商填
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }

        last_err: Exception | None = None
        for p in self.providers:
            body = dict(payload_base)
            body["model"] = p.model
            if p.no_thinking:
                # 火山方舟关闭思维链，避免超长延迟与 reasoning_content
                body["thinking"] = {"type": "disabled"}
            try:
                with httpx.Client(timeout=p.timeout_ms / 1000.0) as client:
                    resp = client.post(
                        p.base,
                        headers={
                            "Authorization": f"Bearer {p.key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                self.last_provider = p.name
                self.stats.setdefault(p.name, {"ok": 0, "fail": 0})["ok"] += 1
                # token 记账：优先用返回的 usage，缺失则按字数估算
                u = data.get("usage") or {}
                pt = int(u.get("prompt_tokens") or 0)
                ct = int(u.get("completion_tokens") or 0)
                if not (pt or ct):
                    pt = _est_tokens(system) + _est_tokens(user)
                    ct = _est_tokens(content)
                self._record(p.name, pt, ct)
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.stats.setdefault(p.name, {"ok": 0, "fail": 0})["fail"] += 1
                print(f"  [LLM] 供应商 {p.name} 调用失败: {e!r} -> 切换下一家")
                continue
        raise RuntimeError(f"所有 LLM 供应商均不可用，最后错误: {last_err!r}")

    def chat_json(self, system: str, user: str, **kw) -> dict:
        """要求模型返回 JSON，自动剥离代码块、容错截断、修复后解析。"""
        raw = self.chat(system, user, **kw).strip()
        return _tolerant_json(raw)


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        # 去掉首尾 ``` 包裹
        body = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        return body.strip()
    return raw


def _balance_and_parse(raw: str) -> dict:
    """从第一个 { 开始，引号感知地扫描括号；若 EOF 时未闭合则补全后再解析。"""
    start = raw.find("{")
    if start == -1:
        raise ValueError("响应中没有 JSON 对象")
    s = raw[start:]
    depth = 0
    in_str = False
    escape = False
    closes = []  # 栈记录待闭合的括号类型
    for ch in s:
        if escape:
            escape = False; continue
        if ch == "\\":
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch in "{[":
            closes.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if closes:
                closes.pop()
    # 补全：截断在字符串内部时先补引号，再补未闭合的括号
    fixed = s + ('"' if in_str else "") + "".join(reversed(closes))
    return json.loads(fixed)


def _tolerant_json(raw: str) -> dict:
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return _balance_and_parse(s)
    except Exception:
        pass
    # 最后兜底：正则抓出概念/题目数组
    import re
    m = re.search(r'"(concepts|items|labels|results)"\s*:\s*\[', s)
    if m:
        key = m.group(1)
        arr_start = s.find("[", m.start())
        # 截取到 EOF，补 ]
        frag = s[arr_start:]
        try:
            arr = json.loads(frag + "]")
            return {key: arr}
        except Exception:
            pass
    raise ValueError(f"无法解析模型返回为 JSON: {s[:300]}")


_llm_singleton: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton
