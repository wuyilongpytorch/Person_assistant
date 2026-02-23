import re
from typing import List, Dict, Tuple

HEADING_RE = re.compile(r"^\s*(#+\s*)?([一二三四五六七八九十0-9]+[\.、])?\s*(教育|项目|项目经验|实习|工作|工作经历|经历|技能|获奖|论文|发表|证书|自我介绍|个人简介|Profile|Education|Projects|Experience|Skills)\s*$", re.I)

def split_sections(text: str) -> List[Tuple[str, str]]:
    """返回 [(section_name, section_text)]"""
    lines = text.splitlines()
    sections = []
    cur_name = "正文"
    cur = []
    for ln in lines:
        if HEADING_RE.match(ln.strip()):
            if cur:
                sections.append((cur_name, "\n".join(cur).strip()))
            cur_name = ln.strip()
            cur = []
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_name, "\n".join(cur).strip()))
    return sections

def sentence_window_chunks(section_text: str, window: int = 5, overlap: int = 2) -> List[str]:
    sents = [s.strip() for s in re.split(r"[。！？；\n]+", section_text) if s.strip()]
    if not sents:
        return []
    chunks = []
    i = 0
    while i < len(sents):
        j = min(i + window, len(sents))
        chunk = "。".join(sents[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j == len(sents):
            break
        i = max(0, j - overlap)
    return chunks

def chunk_document(text: str) -> List[Dict]:
    """每个 chunk 带 section 元信息"""
    out = []
    for sec, sec_text in split_sections(text):
        for ci, ch in enumerate(sentence_window_chunks(sec_text, window=6, overlap=2)):
            out.append({"text": ch, "meta": {"section": sec, "chunk_index": ci}})
    return out
