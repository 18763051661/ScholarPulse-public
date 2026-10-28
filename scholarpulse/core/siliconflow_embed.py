"""硅基流动 Embedding 集成：BAAI/bge-m3-zh 语义向量 + 全路径离线优雅降级。

『RAG × 硅基流动平台』双向集成的语义向量层：
  · embed_texts / embed_query —— POST /v1/embeddings（批量、去重、内存缓存、单次重试）
  · enhance_retriever         —— 把课件块的语义向量矩阵注入 HybridRetriever 第三通道
  · hybrid_search             —— 三通道（dense + CSR-BM25 + remote）RRF 融合检索；
    key 未配置 / 网络失败 / 维度不齐 / 任何异常 → 静默降级纯离线 2 通道（稳定性要求：
    远程增强永远只是增益，绝不成为系统可用性的单点依赖）。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

_EMBED_BATCH = 16
_q_cache: dict[str, list[float]] = {}        # md5(text) -> 向量（查询级缓存）
_doc_cache: dict[tuple, "object"] = {}       # (sid, n_chunks, 语料指纹) -> np 矩阵


def _cfg() -> dict:
    return {
        "url": os.getenv("SILICONFLOW_EMBED_BASE",
                         "https://api.siliconflow.cn/v1/embeddings").strip(),
        "key": os.getenv("SILICONFLOW_KEY", "").strip(),
        "model": os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3").strip(),
        "timeout_s": int(os.getenv("SILICONFLOW_EMBED_TIMEOUT_MS", "6000")) / 1000,
    }


def embed_available() -> bool:
    """硅基流动 Embedding 是否可用（key 已配置即视为可用，网络失败再降级）。"""
    return bool(_cfg()["key"])


def _post_embed(texts: list[str]) -> list[list[float]] | None:
    """调用一次 embeddings API；任何失败返回 None（不抛出——降级语义）。"""
    c = _cfg()
    body = json.dumps({"model": c["model"], "input": texts,
                       "encoding_format": "float"}).encode("utf-8")
    req = urllib.request.Request(
        c["url"], data=body, method="POST",
        headers={"Authorization": f"Bearer {c['key']}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=c["timeout_s"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        vecs = [d.get("embedding") for d in items]
        if len(vecs) != len(texts) or any(not v for v in vecs):
            return None
        return vecs
    except Exception:
        return None


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """批量嵌入（自动分批 + 内存缓存 + 单次重试）。任一批失败 → None（整体降级）。"""
    if not texts or not embed_available():
        return None
    uniq: dict[str, int] = {}
    for t in texts:
        k = hashlib.md5(t.encode("utf-8")).hexdigest()
        if k not in uniq:
            uniq[k] = len(uniq)
    inv = list(uniq.keys())
    raw: dict[str, list[float]] = {}
    for i in range(0, len(inv), _EMBED_BATCH):
        batch = inv[i:i + _EMBED_BATCH]
        got = None
        for attempt in (1, 2):   # 单次重试
            got = _post_embed([t[:2000] for t in batch])
            if got:
                break
            if attempt == 1:
                time.sleep(0.4)
        if not got:
            return None
        raw.update(dict(zip(batch, got)))
    out = []
    for t in texts:
        k = hashlib.md5(t.encode("utf-8")).hexdigest()
        _q_cache[k] = raw[k]
        out.append(raw[k])
    return out


def embed_query(text: str) -> list[float] | None:
    """单条查询嵌入（带缓存）。失败 → None。"""
    if not text.strip() or not embed_available():
        return None
    k = hashlib.md5(text.encode("utf-8")).hexdigest()
    if k in _q_cache:
        return _q_cache[k]
    got = embed_texts([text])
    return got[0] if got else None


def enhance_retriever(sid: str, retriever) -> bool:
    """把该生课件块的语义向量矩阵注入检索器第三通道（按语料指纹缓存，重复调用零开销）。"""
    import numpy as np
    if retriever is None or not embed_available() or retriever.remote_vecs is not None:
        return retriever is not None and retriever.remote_vecs is not None
    fp = hashlib.md5(" ".join(retriever.chunks).encode("utf-8")).hexdigest()
    key = (sid, retriever.n_chunks, fp)
    mat = _doc_cache.get(key)
    if mat is None:
        vecs = embed_texts(retriever.chunks)
        if not vecs:
            return False
        mat = np.asarray(vecs, dtype=float)
        _doc_cache[key] = mat
    return retriever.set_remote_vecs(mat)


def hybrid_search(sid: str, query: str, top_k: int = 5, retriever=None) -> dict:
    """三通道融合检索（硅基流动语义 + 特征哈希密集 + CSR BM25 → RRF）。
    任何远程环节失败自动降级纯离线 2 通道。返回 {hits, channels, embed_model?}。"""
    from .rag_retriever import get_student_retriever
    if retriever is None:
        retriever, meta = get_student_retriever(sid)
        if retriever is None:
            return {"hits": [], "channels": 0, **meta}
    channels, extra, qv = 2, {}, None
    if embed_available():
        try:
            if enhance_retriever(sid, retriever):
                qv = embed_query(query)
                if qv is not None:
                    channels = 3
                    extra = {"embed_model": _cfg()["model"]}
        except Exception:
            channels, qv = 2, None
    hits = retriever.search(query, top_k=top_k, remote_qv=qv if channels == 3 else None)
    return {"hits": hits, "channels": channels, **extra}
