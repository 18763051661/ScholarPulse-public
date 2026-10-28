"""RAG 稀疏矩阵层 + 硅基流动嵌入集成稳定性测试（全离线：远程通道用 monkeypatch 模拟）。

覆盖四类稳定性要求：
  1. 算法正确性     —— CSR 向量化 BM25 与参考循环实现逐值等价
  2. 边界与降级     —— OOV 查询/空语料/维度不齐/网络失败 → 静默降级不崩溃
  3. 双向数据流     —— add_texts / index_llm_knowledge 回写后立即可检索
  4. 基准判据       —— 正确性等价、延迟 p95、召回率、远程通道上限
"""
import numpy as np
import pytest

from scholarpulse.core import siliconflow_embed as SE
from scholarpulse.core.rag_benchmark import run_benchmark
from scholarpulse.core.rag_retriever import (
    HybridRetriever,
    _bm25_scores,
    _csr_bm25_index,
    _csr_query_scores,
    _gen_chunks,
    _tokenize,
    _cache,
    chunk_text,
    get_student_retriever,
    index_llm_knowledge,
)

_DOCS = ["进程同步的临界区互斥实现", "线程调度与时间片轮转策略",
         "死锁预防需要破坏循环等待条件", "虚拟内存页面置换算法LRU",
         "TCP三次握手建立可靠连接"] * 6


@pytest.fixture(autouse=True)
def _reset_caches():
    _cache.clear()
    _gen_chunks.clear()
    SE._q_cache.clear()
    SE._doc_cache.clear()
    yield
    _cache.clear()
    _gen_chunks.clear()
    SE._q_cache.clear()
    SE._doc_cache.clear()


# ---------------- 1. 算法正确性 ----------------
def test_csr_bm25_equivalent_to_reference_loop():
    """CSR 稀疏矩阵向量积与参考循环实现逐值等价（20 组随机查询）。"""
    import random
    doc_toks = [_tokenize(d) for d in _DOCS]
    W, vocab, _idf, _avgdl = _csr_bm25_index(doc_toks)
    rng = random.Random(7)
    for _ in range(20):
        q = "".join(rng.choice("进程同步调度死锁内存连接握手算法页面置换") for _ in range(5))
        ref = _bm25_scores(_tokenize(q), doc_toks)
        csr = [float(x) for x in _csr_query_scores(_tokenize(q), W, vocab)]
        assert all(abs(x - y) < 1e-3 for x, y in zip(ref, csr))


def test_csr_edge_cases_no_crash():
    """边界输入：OOV 查询/空查询/空语料 → 全零或空结果，绝不抛异常。"""
    W, vocab, _idf, _avgdl = _csr_bm25_index([_tokenize(d) for d in _DOCS[:3]])
    assert (_csr_query_scores(_tokenize(" zzxxx 不存在的词 qqq "), W, vocab) == 0).all()
    assert (_csr_query_scores([], W, vocab) == 0).all()
    W0, _v, _i, _a = _csr_bm25_index([])
    assert W0.shape[0] == 0
    assert (_csr_query_scores(["进程"], W0, _v) == 0).all()


def test_tfidf_view_rows_l2_normalized():
    """TF-IDF 第二视图行向量 L2 归一化（非零行范数=1）。"""
    chunks = chunk_text(" ".join(_DOCS), 120, 20)
    r = HybridRetriever(chunks)
    sq = np.asarray(r._tfidf.multiply(r._tfidf).sum(axis=1)).ravel()
    norms = np.sqrt(sq)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)


# ---------------- 2. 通道与降级 ----------------
def test_two_channel_offline_and_remote_degradation():
    """离线 2 通道；维度不齐/行数不匹配 → 静默降级；匹配 → 第三通道出现。"""
    r = HybridRetriever(chunk_text(" ".join(_DOCS), 120, 20))
    q = "死锁预防 循环等待"
    hits2 = r.search(q, top_k=3)
    assert hits2 and "remote" not in hits2[0]
    assert r.set_remote_vecs(np.ones((r.n_chunks + 5, 8))) is False      # 行数不齐
    assert r.remote_vecs is None
    assert r.set_remote_vecs(np.random.RandomState(0).rand(r.n_chunks, 8)) is True
    hits_bad = r.search(q, top_k=3, remote_qv=np.ones(3))                 # 维度不齐
    assert "remote" not in hits_bad[0]
    hits3 = r.search(q, top_k=3, remote_qv=np.random.RandomState(1).rand(8))
    assert "remote" in hits3[0]
    assert {"rank", "chunk_id", "rrf", "text", "dense", "sparse"} <= set(hits3[0])


def test_embed_module_offline_fallbacks(monkeypatch):
    """硅基流动未配置/网络失败 → 一律返回 None 或降级 2 通道，绝不抛异常。"""
    monkeypatch.setenv("SILICONFLOW_KEY", "")
    assert SE.embed_available() is False
    assert SE.embed_texts(["任意文本"]) is None
    assert SE.embed_query("查询") is None
    # key 存在但网络失败
    monkeypatch.setenv("SILICONFLOW_KEY", "sk-test")
    def _boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(SE.urllib.request, "urlopen", _boom)
    assert SE.embed_texts(["任意文本"]) is None
    # hybrid_search 全链路降级：channels=2，仍有检索结果
    r = HybridRetriever(chunk_text(" ".join(_DOCS), 120, 20))
    out = SE.hybrid_search("s900", "死锁预防", retriever=r)
    assert out["channels"] == 2 and out["hits"], out


def test_embed_module_success_path_mocked(monkeypatch):
    """mock 远程嵌入（2 维语义向量：死锁相关→[0,1]）→ 三通道融合生效。"""
    monkeypatch.setenv("SILICONFLOW_KEY", "sk-test")
    def _fake_post(texts):
        return [[0.0, 1.0] if "死锁" in t else [1.0, 0.0] for t in texts]
    monkeypatch.setattr(SE, "_post_embed", _fake_post)
    r = HybridRetriever(chunk_text(" ".join(_DOCS), 120, 20))
    out = SE.hybrid_search("s901", "死锁的必要条件", retriever=r)
    assert out["channels"] == 3 and "embed_model" in out
    assert "remote" in out["hits"][0]
    # 远程通道把死锁块顶到第一
    assert "死锁" in out["hits"][0]["text"]


# ---------------- 3. 双向数据流 ----------------
def test_add_texts_writeback_dedup_and_remote_invalidation():
    """回写：新内容立即可检索；重复回写去重；远程向量矩阵失效待重嵌入。"""
    r = HybridRetriever(chunk_text(" ".join(_DOCS), 120, 20))
    r.set_remote_vecs(np.random.RandomState(0).rand(r.n_chunks, 4))
    n0 = r.n_chunks
    assert r.add_texts(["全新补弱建议：死锁章节需要复习银行家算法"]) == n0 + 1
    assert r.remote_vecs is None                       # 维度失效，交由上层重嵌入
    assert any("银行家" in h["text"] for h in r.search("银行家算法 补弱建议", top_k=2))
    assert r.add_texts(["全新补弱建议：死锁章节需要复习银行家算法"]) == n0 + 1   # 去重
    assert r.add_texts(["", "   "]) == n0 + 1          # 空串忽略


def test_index_llm_knowledge_roundtrip(monkeypatch):
    """LLM 生成知识回写 → 下一次 get_student_retriever 可检索（生成-检索闭环）。"""
    from scholarpulse.core.rag_retriever import _gen_chunks as g
    sid = "s902"
    ret = index_llm_knowledge(sid, ["个性化补弱处方：该生在进程同步章节掌握薄弱，"
                                    "建议先复习信号量机制再练习银行家算法应用题。"])
    assert ret["indexed"] == 1 and ret["n_chunks"] >= 1
    r, meta = get_student_retriever(sid)
    assert r is not None
    hits = r.search("进程同步 补弱处方", top_k=3)
    assert hits and any("银行家" in h["text"] or "信号量" in h["text"] for h in hits)
    # 乱码被质量门拒收
    ret2 = index_llm_knowledge(sid, ["\x01\x02binary garbage \ufffd\ufffd"])
    assert ret2["indexed"] == 0


# ---------------- 4. 基准判据 ----------------
def test_benchmark_all_criteria_pass():
    """基准四类判据全过：等价/延迟 p95<50ms/召回≥90%/远程通道达上限。"""
    rep = run_benchmark(n_chunks=80, n_queries=10, seed=11)
    c = rep["criteria"]
    assert c["equivalent"]
    assert c["hybrid_p95_under_50ms"]
    assert c["recall_ok"]
    assert c["no_catastrophic_loss"]
    assert c["remote_channel_reaches_ceiling"]
    assert rep["correctness"]["max_abs_delta"] < 1e-3


def test_benchmark_deterministic():
    """同种子基准结果完全一致（离线可复现）。"""
    a = run_benchmark(n_chunks=40, n_queries=6, seed=3)
    b = run_benchmark(n_chunks=40, n_queries=6, seed=3)
    assert a["correctness"] == b["correctness"]
    assert a["quality"] == b["quality"]
