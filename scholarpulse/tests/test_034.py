"""混合检索 RAG 测试（全离线）：分块 / 密集向量 / BM25 / RRF 融合 / 学生索引缓存。"""
from scholarpulse.core.rag_retriever import (
    chunk_text, dense_vec, _tokenize, HybridRetriever, get_student_retriever,
)

CORPUS = ("子网掩码的作用是区分IP地址中的网络位与主机位，与IP地址按位与可得网络地址。\n"
          "TCP三次握手建立连接：客户端发SYN，服务端回SYN+ACK，客户端再发ACK。\n"
          "UDP是无连接的传输层协议，不保证可靠交付，但时延更低。\n"
          "路由器工作在网络层，按路由表转发IP数据报；交换机工作在数据链路层。")


def test_tokenize_cjk_bigram_and_words():
    t = _tokenize("子网掩码TCP_ip")
    assert "子网" in t and "网掩" in t and "掩码" in t  # CJK bigram
    assert "tcp_ip" in t                                # 下划线整词
    assert "子" not in t  # 连续汉字只出 bigram，不出单字


def test_chunk_overlap_and_hard_split():
    long_para = "网络" * 300  # 600 字超长段落 → 硬切
    chunks = chunk_text("第一段讲子网掩码。\n第二段讲TCP握手。\n" + long_para,
                        size=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].startswith("第一段")            # 段落装箱
    assert chunks[1].startswith(chunks[0][-20:])     # 相邻块尾部重叠
    assert all(len(c) <= 220 for c in chunks)        # size+overlap 上限


def test_dense_deterministic_self_similar():
    toks = _tokenize("子网掩码区分网络位与主机位 subnet mask")
    v1, v2 = dense_vec(toks), dense_vec(toks)
    assert v1 == v2                                   # 确定性（禁用随机化 hash）
    assert all(0 <= i < 256 for i in v1)              # 维度 = 256
    dot = sum(x * v1[i] for i, x in v1.items())
    assert abs(dot - 1.0) < 1e-6                      # L2 归一化后自点积 = 1


def test_dense_related_higher_than_unrelated():
    target = dense_vec(_tokenize("子网掩码的作用是区分IP地址中的网络位与主机位"))
    rel = dense_vec(_tokenize("子网掩码与网络位主机位的关系"))
    unr = dense_vec(_tokenize("费曼学习法用自己的话复述概念"))
    dot_rel = sum(x * target.get(i, 0.0) for i, x in rel.items())
    dot_unr = sum(x * target.get(i, 0.0) for i, x in unr.items())
    assert dot_rel > dot_unr          # 相关 > 不相关（不相关余弦可为 0）
    assert dot_rel > 0.1              # 相关对有实质重叠


def test_bm25_exact_term_ranks_top():
    r = HybridRetriever(chunk_text(CORPUS))
    hits = r.search("子网掩码", top_k=3)
    assert hits, "应至少命中一路"
    top = hits[0]
    assert "子网掩码" in top["text"]                  # 精确词命中排第一
    assert top["sparse"]["bm25"] > 0
    assert top["dense"]["score"] > 0


def test_rrf_fusion_consistency():
    r = HybridRetriever(chunk_text(CORPUS))
    hits = r.search("传输层协议 时延", top_k=3)
    for h in hits:
        expected = 0.0
        if h["dense"]["rank"]:
            expected += 1 / (60 + h["dense"]["rank"])
        if h["sparse"]["rank"]:
            expected += 1 / (60 + h["sparse"]["rank"])
        assert abs(h["rrf"] - round(expected, 5)) < 1e-9
        assert h["text"]                              # 原文证据非空


def test_student_retriever_fallback_and_cache():
    r1, m1 = get_student_retriever("__no_such_student__")
    if r1 is None:
        assert "msg" in m1                            # 无任何课件文本
        return
    assert m1["source"] == "sample_fallback" and m1["cached"] is False
    r2, m2 = get_student_retriever("__no_such_student__")
    assert m2["cached"] is True and r2 is r1          # 缓存命中同一实例


def test_garbage_binary_rejected(monkeypatch, tmp_path):
    """二进制乱码课件必须被质量门拒收，不能垃圾进垃圾出。"""
    import scholarpulse.core.rag_retriever as R
    monkeypatch.chdir(tmp_path)                       # 相对路径 data/uploads/
    d = tmp_path / "data" / "uploads"
    d.mkdir(parents=True)
    (d / "g001.txt").write_bytes(bytes(range(256)) * 40)
    r, m = R.get_student_retriever("g001")
    assert r is None and "不可读" in m["msg"]
