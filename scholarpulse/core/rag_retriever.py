"""混合检索 RAG 检索器：密集向量 + CSR 稀疏矩阵（BM25/TF-IDF）+ RRF 融合 + GraphRAG 图增强
+ 硅基流动远程嵌入第三通道（可选）+ 索引回写（双向数据流）。

多策略检索的分层实现：
  1. 分块 chunk_text   —— 段落感知分块，块间重叠防语义截断
  2. 密集向量 dense     —— 特征哈希(feature hashing)把字符 n-gram 投影到 256 维
                           定长向量，TF 加权 + L2 归一化，余弦相似度召回
                           （哈希技巧是工业界经典的免训练嵌入方案，
                             全程离线、确定性可复现，不依赖任何 embedding API）
  3. 稀疏矩阵 sparse    —— scipy CSR 稀疏词项-文档矩阵两视图：
                           · BM25 权重矩阵（k1=1.5, b=0.75），查询评分 = W @ q_vec
                             一次稀疏矩阵向量积，替代逐文档 Python 循环（_bm25_scores
                             保留为参考实现，供等价性测试）
                           · TF-IDF（L2 归一化）第二视图，作为算法对照元数据
  4. 远程嵌入 remote    —— 硅基流动 Embedding API（BAAI/bge-m3，1024 维语义向量），
                           文档矩阵由 set_remote_vecs 注入、查询向量经 remote_qv 传入；
                           网络失败/未配置时静默降级为纯离线 2 通道（稳定性要求）
  5. 融合 fusion        —— Reciprocal Rank Fusion (RRF, k=60)，通道数自适应（2/3 路）：
                           fused(i) = Σ 1/(60 + rank_ch(i))
  6. 双向数据流         —— 读：检索片段注入 LLM 提示词（retrieve_course_knowledge 工具）；
                           写：add_texts / index_llm_knowledge 把 LLM 生成知识回写索引
  7. 图增强 GraphRAG    —— expand_with_graph：命中片段识别涉及概念后，
                           沿概念先修图扩展一跳邻居（上游先修/下游依赖+掌握度），
                           向量检索 × 知识图谱融合

数据源：学生上传的课件原文 data/uploads/{sid}.txt（无则回退示例课件），
按文件 mtime+size 做缓存失效——重新上传课件后索引自动重建。
"""
from __future__ import annotations
import hashlib
import math
import os

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

_DIM = 256          # 密集向量维度
_K1, _B = 1.5, 0.75  # BM25 超参
_RRF_K = 60          # RRF 平滑常数
_CHUNK, _OVERLAP = 280, 60


# ---------- 分词：CJK 字符 bigram + 英文/数字词 ----------
def _tokenize(text: str) -> list[str]:
    toks: list[str] = []
    run: list[str] = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            run.append(ch)
        else:
            if len(run) == 1:
                toks.append(run[0])
            toks.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
            run = []
    if len(run) == 1:
        toks.append(run[0])
    toks.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
    # 英文/数字整词（长度≥2，小写化）
    word: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch == '_'):
            word.append(ch.lower())
        elif word:
            if len(word) >= 2:
                toks.append(''.join(word))
            word = []
    if len(word) >= 2:
        toks.append(''.join(word))
    return toks


# ---------- 分块：段落感知 + 重叠 ----------
def chunk_text(text: str, size: int = _CHUNK, overlap: int = _OVERLAP) -> list[str]:
    paras = [p.strip() for p in text.replace('\r\n', '\n').split('\n') if p.strip()]
    if not paras:
        return []
    # 段落贪心装箱
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > size:  # 超长段落硬切（带重叠）
            if buf:
                chunks.append(buf)
                buf = ""
            step = max(size - overlap, 1)
            for i in range(0, len(p), step):
                seg = p[i:i + size]
                chunks.append(seg)
                if i + size >= len(p):
                    break
            continue
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf} {p}".strip()
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    # 相邻块尾部重叠（语义缓冲）
    out = []
    for i, c in enumerate(chunks):
        if i > 0:
            tail = chunks[i - 1][-overlap:]
            c = f"{tail} {c}"
        out.append(c)
    return out


# ---------- 密集向量：特征哈希（确定性，禁用随机化 hash()） ----------
def _h_idx_sign(token: str) -> tuple[int, int]:
    h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
    return h % _DIM, (1 if (h >> 16) & 1 else -1)


def dense_vec(tokens: list[str]) -> dict[int, float]:
    v: dict[int, float] = {}
    for t in tokens:
        idx, sign = _h_idx_sign(t)
        v[idx] = v.get(idx, 0.0) + sign
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {i: x / norm for i, x in v.items()}


def _dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return round(sum(x * b.get(i, 0.0) for i, x in a.items()), 4)


# ---------- 稀疏检索：BM25 ----------
def _bm25_scores(q_toks: list[str], doc_toks: list[list[str]]) -> list[float]:
    n = len(doc_toks)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in doc_toks) / n or 1.0
    tf = [{} for _ in range(n)]
    df: dict[str, int] = {}
    for i, d in enumerate(doc_toks):
        for t in d:
            tf[i][t] = tf[i].get(t, 0) + 1
    for d in doc_toks:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for i in range(n):
        s = 0.0
        dl = len(doc_toks[i]) or 1
        for t in set(q_toks):
            if t not in tf[i]:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = tf[i][t]
            s += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / avgdl))
        scores.append(round(s, 4))
    return scores


def _ranks(scores: list[float]) -> dict[int, int]:
    """分数 → 排名（1 起，降序，仅保留非零分）。"""
    order = sorted((s, i) for i, s in enumerate(scores) if s > 0)
    return {i: r for r, (_, i) in enumerate(reversed(order), 1)}


# ---------- 稀疏矩阵算法层：CSR 向量化 BM25 / TF-IDF ----------
def _csr_bm25_index(doc_toks: list[list[str]]):
    """把整个语料库编码为 CSR 稀疏词项-文档矩阵 W（n_docs × vocab），
    非零元 = BM25 词项权重 idf(t)·tf·(k1+1)/(tf+k1·(1-b+b·dl/avgdl))。
    查询评分 = W @ q_vec —— 一次稀疏矩阵向量积替代逐文档 Python 循环，
    时间复杂度从 O(n_docs·|q|·dict 开销) 降为 O(nnz(W) 中命中列的子线性扫描)。
    返回 (W, vocab, idf, avgdl)。"""
    n = len(doc_toks)
    if n == 0:
        return csr_matrix((0, 0)), {}, np.zeros(0), 0.0
    avgdl = sum(len(d) for d in doc_toks) / n or 1.0
    # 词表 + 文档频率
    vocab: dict[str, int] = {}
    df_counts: dict[str, int] = {}
    per_doc_tf: list[dict[str, int]] = []
    for d in doc_toks:
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
            if t not in vocab:
                vocab[t] = len(vocab)
        per_doc_tf.append(tf)
        for t in tf:
            df_counts[t] = df_counts.get(t, 0) + 1
    idf = np.zeros(len(vocab))
    for t, j in vocab.items():
        df = df_counts[t]
        idf[j] = math.log(1 + (n - df + 0.5) / (df + 0.5))
    # COO 三元组 → CSR
    rows, cols, vals = [], [], []
    for i, tf in enumerate(per_doc_tf):
        dl = len(doc_toks[i]) or 1
        norm = _K1 * (1 - _B + _B * dl / avgdl)
        for t, f in tf.items():
            rows.append(i)
            cols.append(vocab[t])
            vals.append(idf[vocab[t]] * f * (_K1 + 1) / (f + norm))
    W = csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
                   shape=(n, len(vocab)))
    return W, vocab, idf, avgdl


def _csr_query_scores(q_toks: list[str], W: csr_matrix, vocab: dict[str, int]) -> np.ndarray:
    """查询评分：q_vec 为词频稀疏行向量，scores = W @ q_vec（一次 CSR 稀疏矩阵向量积）。
    与 _bm25_scores 参考实现数学等价（对 query 去重词项求和）。"""
    if W.shape[0] == 0 or not q_toks:
        return np.zeros(W.shape[0])
    qtf: dict[int, int] = {}
    for t in set(q_toks):
        j = vocab.get(t)
        if j is not None:
            qtf[j] = 1
    if not qtf:
        return np.zeros(W.shape[0])
    qv = csr_matrix((np.ones(len(qtf)), (np.zeros(len(qtf), dtype=int), list(qtf.keys()))),
                    shape=(1, W.shape[1]))
    return np.asarray((W @ qv.T).todense()).ravel()


def _csr_tfidf_index(doc_toks: list[list[str]], vocab: dict[str, int]):
    """第二稀疏视图：TF-IDF（L2 归一化）CSR 矩阵——供结果元数据与基准对比。"""
    n, V = len(doc_toks), len(vocab)
    if n == 0 or V == 0:
        return csr_matrix((n, V))
    # 文档频率一次统计
    df: dict[str, int] = {}
    per_doc_tf: list[dict[str, int]] = []
    for toks in doc_toks:
        tf: dict[str, int] = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1
        per_doc_tf.append(tf)
        for t in tf:
            df[t] = df.get(t, 0) + 1
    rows, cols, vals = [], [], []
    for i, tf in enumerate(per_doc_tf):
        dl = sum(tf.values()) or 1
        for t, f in tf.items():
            j = vocab[t]
            rows.append(i)
            cols.append(j)
            vals.append((f / dl) * math.log(1 + n / (1 + df[t])))
    M = csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, V))
    # L2 行归一化
    sq = np.asarray(M.multiply(M).sum(axis=1)).ravel()
    rn = np.sqrt(sq)
    rn[rn == 0] = 1.0
    return sparse.diags(1.0 / rn) @ M


# ---------- 混合检索器 ----------
class HybridRetriever:
    """对一组文本块建立三索引：密集(特征哈希) + 稀疏矩阵(CSR BM25/TF-IDF) + 可选远程嵌入。

    search() 返回 RRF 融合排序；通道数自适应——离线 2 通道（dense+bm25），
    接入硅基流动 Embedding 后自动升级 3 通道（+remote）。
    add_texts() 支持 LLM 生成知识回写索引（双向数据流的写入方向）。
    """

    def __init__(self, chunks: list[str]):
        self.chunks = list(chunks)
        self.doc_toks = [_tokenize(c) for c in self.chunks]
        self.dense_vecs = [dense_vec(t) for t in self.doc_toks]
        self.n_chunks = len(self.chunks)
        # 稀疏矩阵层：CSR BM25 权重矩阵 + CSR TF-IDF 视图（一次构建，查询 O(nnz) 向量化）
        self._W, self._vocab, self._idf, self._avgdl = _csr_bm25_index(self.doc_toks)
        self._tfidf = _csr_tfidf_index(self.doc_toks, self._vocab)
        # 远程嵌入通道（硅基流动）：由外部注入，缺省 None = 纯离线
        self.remote_vecs: "np.ndarray | None" = None   # shape (n_chunks, d), L2 归一化

    # ---- 索引回写（双向数据流：LLM 生成知识 → 检索索引）----
    def add_texts(self, texts: list[str]) -> int:
        """把 LLM 生成/新增的文本块追加进索引并重建稀疏矩阵（保证与全量重建等价）。
        空串与重复块自动去重。返回追加后的总块数。"""
        added = 0
        for t in texts:
            t = (t or "").strip()
            if t and t not in self.chunks:
                self.chunks.append(t)
                added += 1
        if added:
            self.doc_toks = [_tokenize(c) for c in self.chunks]
            self.dense_vecs = [dense_vec(t) for t in self.doc_toks]
            self.n_chunks = len(self.chunks)
            self._W, self._vocab, self._idf, self._avgdl = _csr_bm25_index(self.doc_toks)
            self._tfidf = _csr_tfidf_index(self.doc_toks, self._vocab)
            self.remote_vecs = None   # 远程向量与新块数不再对齐，交由上层重嵌入
        return self.n_chunks

    # ---- 远程嵌入注入 ----
    def set_remote_vecs(self, mat) -> bool:
        """注入硅基流动 Embedding 矩阵（行序须与 chunks 一致且已 L2 归一化）。"""
        m = np.asarray(mat, dtype=float)
        if m.ndim != 2 or m.shape[0] != self.n_chunks:
            return False
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.remote_vecs = m / norms
        return True

    def search(self, query: str, top_k: int = 5,
               remote_qv: "np.ndarray | None" = None) -> list[dict]:
        """混合检索。remote_qv 为查询的硅基流动嵌入向量（可选）：
        与 set_remote_vecs 注入的文档矩阵同维度时启用第三通道，否则静默降级 2 通道。"""
        q_toks = _tokenize(query)
        if not self.chunks or not q_toks:
            return []
        # 通道1 密集：特征哈希余弦
        qv = dense_vec(q_toks)
        dense_scores = [_dot(qv, dv) for dv in self.dense_vecs]
        # 通道2 稀疏：CSR BM25 —— 一次稀疏矩阵向量积
        csr_scores = _csr_query_scores(q_toks, self._W, self._vocab)
        sparse_scores = [round(float(x), 4) for x in csr_scores]
        # 稀疏第二视图：TF-IDF 余弦（元数据，展示稀疏矩阵层的第二个算法视角）
        qt = csr_matrix(
            (np.ones(len(set(q_toks) & set(self._vocab))),
             (np.zeros(len(set(q_toks) & set(self._vocab)), dtype=int),
              sorted(self._vocab[t] for t in set(q_toks) if t in self._vocab))),
            shape=(1, len(self._vocab)))
        tfidf_scores = np.asarray((self._tfidf @ qt.T).todense()).ravel()
        d_rank, s_rank = _ranks(dense_scores), _ranks(sparse_scores)
        # 通道3 远程嵌入（可选）：硅基流动语义向量余弦
        r_rank: dict[int, int] = {}
        remote_scores = [0.0] * self.n_chunks
        if self.remote_vecs is not None and remote_qv is not None:
            rq = np.asarray(remote_qv, dtype=float).ravel()
            if rq.size == self.remote_vecs.shape[1]:
                nq = np.linalg.norm(rq) or 1.0
                remote_scores = [round(float(x), 4)
                                 for x in self.remote_vecs @ (rq / nq)]
                r_rank = _ranks(remote_scores)
        # RRF 融合：2 或 3 通道排名互补（单路命中的块只累加该路贡献）
        rrf = {}
        for i in set(d_rank) | set(s_rank) | set(r_rank):
            s = 0.0
            if i in d_rank:
                s += 1.0 / (_RRF_K + d_rank[i])
            if i in s_rank:
                s += 1.0 / (_RRF_K + s_rank[i])
            if i in r_rank:
                s += 1.0 / (_RRF_K + r_rank[i])
            rrf[i] = s
        order = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        hits = []
        for rank, (i, fused) in enumerate(order, 1):
            hits.append({
                "rank": rank, "chunk_id": i, "rrf": round(fused, 5),
                "text": self.chunks[i][:160],
                "dense": {"score": dense_scores[i], "rank": d_rank.get(i)},
                "sparse": {"bm25": sparse_scores[i], "rank": s_rank.get(i),
                           "tfidf": round(float(tfidf_scores[i]), 4)},
                **({"remote": {"score": remote_scores[i], "rank": r_rank.get(i)}}
                   if r_rank else {}),
            })
        return hits


# ---------- 可读性质量门：过滤二进制/乱码 ----------
def _clean_ratio(s: str) -> float:
    # 只把控制字符/替换符视为乱码——中文标点（，：（）等全角字符）是正常文本，
    # 不能因标点占比高而把真实课件误判为二进制垃圾
    ok = sum(1 for ch in s
             if ch != '\ufffd' and (ch in '\t\n\r' or 31 < ord(ch) != 127))
    return ok / max(len(s), 1)


# ---------- GraphRAG：沿概念先修图扩展证据 ----------
def expand_with_graph(hits: list[dict], concepts: list, mastery: dict | None = None,
                      max_neighbors: int = 6) -> list[dict]:
    """识别命中片段涉及的概念，沿先修图扩展一跳邻居（上游先修 + 下游依赖），
    返回图谱上下文——LLM 看到的不止原文，还有知识结构与掌握度（GraphRAG：向量×图谱融合）。"""
    cmap = {c.cid: c for c in concepts}
    # 命中片段涉及哪些概念（概念名在片段文本中出现）
    involved: list[str] = []
    for h in hits:
        for c in concepts:
            if c.name and c.name in h["text"] and c.cid not in involved:
                involved.append(c.cid)
    # 下游依赖索引：谁的前置是我
    downstream: dict[str, list[str]] = {}
    for c in concepts:
        for pre in c.prerequisites or []:
            downstream.setdefault(pre, []).append(c.cid)
    out: list[dict] = []
    for cid in involved:
        for rel, nbs in (("上游先修", cmap[cid].prerequisites or []),
                         ("下游依赖", downstream.get(cid, []))):
            for n_cid in nbs[:3]:
                c = cmap.get(n_cid)
                if not c:
                    continue
                m = mastery.get(n_cid) if mastery else None
                out.append({"cid": n_cid, "name": c.name, "chapter": c.chapter,
                            "relation": f"{cmap[cid].name}的{rel}",
                            "mastery": round(m.p, 2) if m else None})
    seen, dedup = set(), []
    for g in out:
        if g["cid"] not in seen:
            seen.add(g["cid"])
            dedup.append(g)
    return dedup[:max_neighbors]


# ---------- 学生课件索引（缓存随源文件变化自动失效） ----------
_cache: dict = {}
# LLM 生成知识的内存补充索引（双向数据流·写方向）：sid -> [chunk, ...]
_gen_chunks: dict = {}


def index_llm_knowledge(sid: str, texts: list[str]) -> dict:
    """把 LLM 生成的知识（课程摘要/补弱建议/知识点讲解等）回写进该生的检索索引。

    双向数据流的写入方向：LLM 产出 → 分块 → 质量门 → 追加索引，
    下一次检索即可命中『LLM 自己生成过什么』，形成生成-检索闭环。
    返回 {indexed: n, n_chunks: 总块数}。"""
    clean = [t.strip() for t in texts if t and _clean_ratio(t.strip()) >= 0.9]
    if not clean:
        return {"indexed": 0, "n_chunks": len(_gen_chunks.get(sid, [])),
                "msg": "无可索引文本（空或未通过质量门）"}
    r, _meta = get_student_retriever(sid)
    store = _gen_chunks.setdefault(sid, [])
    for t in clean:
        for c in chunk_text(t):
            if c not in store:
                store.append(c)
    if r is not None:
        r.add_texts(store)
        r._gen_merged = True   # 标记：该检索器已合并当前补充索引
    return {"indexed": len(clean), "n_chunks": len(store)}


def get_student_retriever(sid: str) -> tuple[HybridRetriever | None, dict]:
    """加载学生课件（无则回退示例课件）并构建检索器；带 mtime+size 缓存。
    若存在 LLM 生成补充知识（_gen_chunks），自动合并进索引（双向数据流·写方向生效）。"""
    path = f"data/uploads/{sid}.txt"
    source = "student_upload"
    if not os.path.exists(path):
        path = "data/sample_lecture.txt"
        source = "sample_fallback"
        if not os.path.exists(path) and not _gen_chunks.get(sid):
            return None, {"msg": "无课件文本可索引"}
    r = None
    meta: dict = {}
    if os.path.exists(path):
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        hit = _cache.get(sid)
        if hit and hit[0] == key:
            r = hit[1]
            meta = {"source": source, "n_chunks": r.n_chunks, "cached": True}
        else:
            text = open(path, encoding="utf-8", errors="ignore").read()
            chunks = [c for c in chunk_text(text) if _clean_ratio(c) >= 0.9]
            if not chunks:
                meta = {"msg": "课件文本不可读（疑似二进制/乱码），请重新上传",
                        "source": source}
            else:
                r = HybridRetriever(chunks)
                _cache[sid] = (key, r)
                meta = {"source": source, "n_chunks": len(chunks), "cached": False}
    gen = _gen_chunks.get(sid)
    if gen:
        if r is None:
            r = HybridRetriever(gen)
            r._gen_merged = True
            meta = {"source": "llm_generated_only", "n_chunks": len(gen), "cached": False}
        elif not getattr(r, "_gen_merged", False):
            r.add_texts(gen)
            r._gen_merged = True
            meta = {**meta, "n_chunks": r.n_chunks, "llm_added": True}
    return r, meta
