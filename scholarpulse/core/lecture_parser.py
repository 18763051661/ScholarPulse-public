"""
多格式课件解析：TXT / PDF / DOCX。
根据文件扩展名自动分发，统一返回纯文本。
"""
from __future__ import annotations
import os


def parse_lecture(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return _parse_pdf(path)
        if ext in (".docx", ".doc"):
            return _parse_docx(path)
        return _parse_txt(path)
    except Exception as e:
        return f"[解析失败 {ext}: {e}]"


def _parse_txt(path: str) -> str:
    """严格 utf-8 → gbk 解码；latin-1 兜底前先做可读性检查，二进制文件直接拒收。"""
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="latin-1") as f:
        raw = f.read()
    ok = sum(1 for ch in raw if '\u4e00' <= ch <= '\u9fff'
             or (ch.isascii() and (ch.isprintable() or ch in '\t\n')))
    return raw if ok / max(len(raw), 1) >= 0.9 else ""


def _parse_pdf(path: str) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
    return "\n".join(parts)


def _parse_docx(path: str) -> str:
    import docx
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())
