from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import time
from typing import List, Optional,Literal, TypedDict,Dict, Any,Tuple,Union
import re
from qdrant_client.http import models as qm
from qdrant_store import QdrantStore, ollama_embed
import json, re
import numpy as np
from pypdf import PdfReader

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
from reranker import rerank
from chunking import chunk_document
from bm25_store import BM25Store
from qdrant_client.http import models as qm
from pathlib import Path
from collections import OrderedDict

app = FastAPI(title="Persona Assistant API (Phase 2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

BM25_PATH = BASE_DIR / "data" / "bm25.pkl"
PROFILE_PATH = BASE_DIR / "data" / "profile.json"
PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

store = QdrantStore(url="http://localhost:6333")

# -----------------------------
# Chroma client + collection
# -----------------------------
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
print("Embedding model:", EMBED_MODEL)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="persona_docs",
    embedding_function=embedding_fn,
)

# -----------------------------
# Config knobs (Phase 2)
# -----------------------------
CHUNK_SIZE = 250        # 每个 chunk 字符数
CHUNK_OVERLAP = 60    # 重叠字符数
TOP_K = 6               # 检索返回段落数量
MIN_SCORE = 0.42       # “证据门禁”阈值：低于则拒答（你后面可以调）


# -----------------------------
# Schemas
# -----------------------------
class ChatRequest(BaseModel):
    message: str

class Citation(BaseModel):
    source: str
    excerpt: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []

class IndexStatus(BaseModel):
    files: int
    chunks: int
    updated_at: Optional[float] = None

class Cand(TypedDict, total=False):
    id: int
    text: str
    meta: Dict[str, Any]
    score: float          # 统一：越大越相关（0~1 或至少同方向）
    source: str           # dense/bm25/rrf/rerank 等

def to_cand(x: Any, source: str = "unknown") -> Cand:
    """
    把各种返回结构统一成 dict cand：
    - tuple(text, meta, score)
    - tuple(id, text, meta, score)
    - dict 已经是 cand
    """
    if isinstance(x, dict):
        c: Cand = {
            "text": x.get("text", ""),
            "meta": x.get("meta") or {},
            "score": float(x.get("score", 0.0)),
            "source": x.get("source", source),
        }
        if "id" in x and x["id"] is not None:
            c["id"] = int(x["id"])
        return c

    if isinstance(x, tuple):
        if len(x) == 3:
            text, meta, score = x
            return {"text": str(text), "meta": meta or {}, "score": float(score), "source": source}
        if len(x) == 4:
            _id, text, meta, score = x
            return {"id": int(_id), "text": str(text), "meta": meta or {}, "score": float(score), "source": source}

    # 最后兜底
    return {"text": str(x), "meta": {}, "score": 0.0, "source": source}

def normalize_cands(xs: List[Any], source: str = "unknown") -> List[Cand]:
    out: List[Cand] = []
    for x in xs or []:
        out.append(to_cand(x, source=source))
    # 统一按 score 降序
    out.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return out
# -----------------------------
# Utilities
# -----------------------------

META_PAT = re.compile(
    r"(你能(干什么|做什么|做些什么)|你是谁|你是做什么的|怎么用|如何使用|使用方法|怎么上传|上传文件|帮助|功能|说明|介绍一下)",
    re.I
)

def is_meta_question(q: str) -> bool:
    return bool(re.search(r"(你能做什么|你是谁|怎么用|如何使用|怎么上传|帮助|功能|说明)", q))


SMALLTALK_PAT = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|在吗|在不在|嗨|早上好|上午好|中午好|下午好|晚上好|晚安|谢谢|谢了|再见|拜拜)\s*[!！。．,.，]*\s*$",
    re.I
)

def _first_sentence_containing(text: str, needle: str) -> str:
    # 用最简单的“按行”取证据句：稳定且够用
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return ""


def extract_profile_from_texts(file_texts: list[tuple[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    file_texts: [(filename, full_text), ...]
    return:
      {
        "birth_year": {"value": "1999", "evidence": "...1999 年 5 月生...", "source": "简历.txt"},
        ...
      }
    """
    profile: Dict[str, Dict[str, str]] = {}

    def set_field(key: str, value: str, evidence: str, source: str):
        if not value:
            return
        # 已有就不覆盖（也可以按更长 evidence 或更靠前规则覆盖，这里先简单）
        if key not in profile:
            profile[key] = {"value": value, "evidence": evidence, "source": source}

    for fname, raw in file_texts:
        if not raw or not raw.strip():
            continue

        text = raw

        # 性别：优先 “性别：男/女”，其次 “，男，” “男，1999年”
        m = re.search(r"性别[:：]?\s*(男|女)", text)
        if m:
            ev = _first_sentence_containing(text, m.group(0)) or m.group(0)
            set_field("gender", m.group(1), ev, fname)
        else:
            m = re.search(r"(男|女)[，,]\s*\d{4}\s*年", text)
            if m:
                ev = _first_sentence_containing(text, m.group(0)) or m.group(0)
                set_field("gender", m.group(1), ev, fname)

        # 出生年/月：1999 年 5 月生 / 1999年生 / 出生于1999年
        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(生|出生)", text)
        if m:
            year, month = m.group(1), m.group(2)
            ev = _first_sentence_containing(text, m.group(0)) or m.group(0)
            set_field("birth_year", year, ev, fname)
            set_field("birth_month", month, ev, fname)
        else:
            m = re.search(r"(出生于|生于)?\s*(\d{4})\s*年\s*(生|出生)", text)
            if m:
                year = m.group(2)
                ev = _first_sentence_containing(text, m.group(0)) or m.group(0)
                set_field("birth_year", year, ev, fname)

        # 身高/体重
        m = re.search(r"身高[:：]?\s*(\d{2,3})\s*(cm|CM)?", text)
        if m:
            ev = _first_sentence_containing(text, "身高") or m.group(0)
            set_field("height_cm", m.group(1), ev, fname)

        m = re.search(r"体重[:：]?\s*(\d{2,3})\s*(kg|KG|公斤)?", text)
        if m:
            ev = _first_sentence_containing(text, "体重") or m.group(0)
            set_field("weight_kg", m.group(1), ev, fname)

        # 本科学校：本科毕业于XXX大学 / 毕业于XXX大学（本科）
        m = re.search(r"(本科毕业于|本科就读于|毕业于)\s*([^\n，,。；;（）()]{2,30}?(大学|学院))", text)
        if m:
            school = m.group(2)
            ev = _first_sentence_containing(text, school) or m.group(0)
            set_field("undergrad_school", school, ev, fname)

        # 本科专业：XXX专业 / 专业：XXX / 主修：XXX
        m = re.search(r"专业[:：]?\s*([^\n，,。；;（）()]{2,20})", text)
        if m:
            major = m.group(1).strip()
            ev = _first_sentence_containing(text, "专业") or m.group(0)
            set_field("undergrad_major", major, ev, fname)
        else:
            m = re.search(r"([^\n，,。；;（）()]{2,20})专业", text)
            if m:
                major = m.group(1).strip()
                # 避免把“软件工程专业（2017.09”抽成“软件工程专业（2017.09”
                major = re.sub(r"[（(].*$", "", major).strip()
                ev = _first_sentence_containing(text, "专业") or m.group(0)
                set_field("undergrad_major", major, ev, fname)

    return profile


def save_profile(profile: Dict[str, Dict[str, str]]) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile() -> Dict[str, Dict[str, str]]:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_smalltalk(q: str) -> bool:
    return bool(SMALLTALK_PAT.match(q.strip()))

def sanitize_text(text: str) -> str:
    # 邮箱
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    # 电话（简单处理：+xx 或一串数字，实际你可更精细）
    text = re.sub(r"(\+\d{1,3}[-\s]?)?\d{3,4}[-\s]?\d{4,8}", "[REDACTED_PHONE]", text)
    # 新加坡/中式地址（非常粗略：包含 Block/Road/Street/大道/路/街 的行）
    text = re.sub(r".*(Block|Road|Street|Avenue|大道|路|街|巷|弄).*", "[REDACTED_ADDRESS_LINE]", text)
    return text
def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts)

    # 其他格式先不支持
    return ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_name: Optional[str] = None,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # --- Markdown 结构切分：优先按标题拆（## / ###）---
    # 适用：你的 2025.md 项目经历
    looks_like_md = ("##" in text) or ("```" in text) or ("- " in text) or ("# " in text) \
                    or (source_name and source_name.lower().endswith(".md"))

    if looks_like_md:
        # 1) 找到所有二/三级标题作为 section 起点
        lines = text.splitlines()
        sections = []
        cur_title = None
        cur_buf = []

        header_pat = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")  # ## 标题 / ### 标题

        def flush():
            nonlocal cur_title, cur_buf
            body = "\n".join(cur_buf).strip()
            if body:
                sections.append((cur_title, body))
            cur_buf = []

        for ln in lines:
            m = header_pat.match(ln)
            if m:
                # 新 section
                flush()
                cur_title = m.group(2).strip()
                cur_buf.append(ln)  # 保留标题行本身
            else:
                cur_buf.append(ln)
        flush()

        # 2) 每个 section 作为一个 chunk；太长再二次切（按段落 > 滑窗）
        chunks: List[str] = []
        for title, body in sections:
            prefix = f"\n" if title else ""
            content = (prefix + body).strip()

            if len(content) <= chunk_size * 2:
                # section 不太长，直接用一个 chunk（更利于命中项目名）
                chunks.append(content)
                continue

            # 太长：先按空行拆段落，再拼到 chunk_size 左右
            paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            buf = ""
            for p in paras:
                if not buf:
                    buf = p
                    continue
                if len(buf) + 2 + len(p) <= chunk_size * 2:
                    buf = buf + "\n\n" + p
                else:
                    chunks.append(buf.strip())
                    buf = p
            if buf.strip():
                chunks.append(buf.strip())

        # 3) 如果没有拆出 section（比如 md 没有 ##），回退到滑窗
        if chunks:
            return chunks

    # --- 回退：字符滑窗（你原来的逻辑）---
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def route_query_local(q: str) -> str:
    prompt = f"""你是分类器。把用户问题分成两类：
- fact: 查询单一事实（身高/学校/联系方式/时间点等）
- summary: 需要总结/列举（项目/经历/技能/做过什么等）

只输出一个词：fact 或 summary。

问题：{q}
"""
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
                      timeout=60)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip().lower()
    return "summary" if "summary" in out else "fact"

def pick_diverse_by_section(cands: List[Cand], max_sections: int = 6, per_section: int = 2) -> List[Cand]:
    picked: List[Cand] = []
    section_count: Dict[str, int] = {}
    seen_sections: set[str] = set()

    for c in cands:
        meta = c.get("meta") or {}
        title = meta.get("section_title") or "unknown"

        # 每个 section 最多拿 per_section 个
        if section_count.get(title, 0) >= per_section:
            continue

        picked.append(c)
        section_count[title] = section_count.get(title, 0) + 1
        seen_sections.add(title)

        if len(seen_sections) >= max_sections and len(picked) >= max_sections * per_section:
            break

    return picked


def chunk_document(text: str, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    # 更宽松的“项目材料/markdown”检测：有 项目一/名称：/时间： 也算
    looks_like_structured = (
        (source_name and source_name.lower().endswith(".md"))
        or ("##" in text)
        or ("名称：" in text) or ("名称:" in text)
        or re.search(r"^\s*项目\s*[一二三四五六七八九十0-9]+\s*$", text, re.M)
    )

    out: List[Dict[str, Any]] = []

    # ---------- 按“标题/项目块”切 ----------
    if looks_like_structured:
        lines = text.splitlines()

        # 1) Markdown 标题（##/###）
        md_header_pat = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")

        # 2) 你的格式：项目一 / 项目二 / 项目1 / 项目2
        proj_header_pat = re.compile(r"^\s*项目\s*([一二三四五六七八九十0-9]+)\s*$")

        name_pat = re.compile(r"^\s*名称[:：]\s*(.+\S)\s*$", re.M)

        sections: List[tuple[str, str]] = []
        cur_title: Optional[str] = None
        cur_buf: List[str] = []

        def flush():
            nonlocal cur_title, cur_buf
            body = "\n".join(cur_buf).strip()
            if not body:
                cur_buf = []
                return

            # 如果正文里有“名称：XXX”，用它覆盖 title（更利于检索）
            m = name_pat.search(body)
            final_title = m.group(1).strip() if m else (cur_title or "未命名段落")

            sections.append((final_title, body))
            cur_buf = []

        for ln in lines:
            m1 = md_header_pat.match(ln)
            m2 = proj_header_pat.match(ln)

            if m1:
                flush()
                cur_title = m1.group(2).strip()
                cur_buf.append(ln)
                continue

            if m2:
                flush()
                cur_title = f"项目{m2.group(1)}"
                cur_buf.append(ln)
                continue

            cur_buf.append(ln)

        flush()

        # 如果成功切出 sections，就按 section 组 chunk
        if sections:
            chunk_id = 0
            for title, body in sections:
                prefix = f"【项目：{title}】\n"
                content = (prefix + body).strip()

                # section 不太长：一个 chunk（更稳）
                if len(content) <= CHUNK_SIZE * 3:
                    out.append({
                        "text": content,
                        "meta": {
                            "section_title": title,
                            "chunk_index": chunk_id,
                            "chunker": "section(project_or_md)",
                        }
                    })
                    chunk_id += 1
                    continue

                # 太长：按空行拆段落，再拼
                paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
                buf = ""
                for p in paras:
                    if not buf:
                        buf = p
                        continue
                    if len(buf) + 2 + len(p) <= CHUNK_SIZE * 3:
                        buf = buf + "\n\n" + p
                    else:
                        out.append({
                            "text": buf.strip(),
                            "meta": {
                                "section_title": title,
                                "chunk_index": chunk_id,
                                "chunker": "section_para(project_or_md)",
                            }
                        })
                        chunk_id += 1
                        buf = p
                if buf.strip():
                    out.append({
                        "text": buf.strip(),
                        "meta": {
                            "section_title": title,
                            "chunk_index": chunk_id,
                            "chunker": "section_para(project_or_md)",
                        }
                    })
                    chunk_id += 1

            return out

    # ---------- 回退：字符滑窗 ----------
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        ch = text[start:end].strip()
        if ch:
            chunks.append(ch)
        if end == n:
            break
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0

    for i, ch in enumerate(chunks):
        out.append({
            "text": ch,
            "meta": {
                "section_title": None,
                "chunk_index": i,
                "chunker": "sliding_window",
            }
        })
    return out
   

def rebuild_index() -> dict:
    files = sorted([p for p in UPLOAD_DIR.iterdir() if p.is_file()])
    file_texts = []
    for f in files:
        t = extract_text_from_file(f)
        if t and t.strip():
            # 注意：profile 建议用“未 sanitize 的原文”，否则邮箱会被 [REDACTED]
            file_texts.append((f.name, t))

    profile = extract_profile_from_texts(file_texts)
    save_profile(profile)

    chunks_all = []  # (id, text, meta)
    pid = 1
    for f in files:
        text = extract_text_from_file(f)
        if not text.strip():
            continue
        text = sanitize_text(text)

        for obj in chunk_document(text, source_name=f.name):
            ch = obj["text"]
            meta = obj["meta"]
            meta.update({"source": f.name, "char_len": len(ch)})
            chunks_all.append((pid, ch, meta))
            pid += 1

    if not chunks_all:
        return {"files": len(files), "chunks": 0, "updated_at": time.time()}

    # Qdrant reset
    dim = len(ollama_embed(chunks_all[0][1]))
    store.reset_collection(dim)

    points = []
    ids, texts, metas = [], [], []
    for _id, ch, meta in chunks_all:
        vec = ollama_embed(ch)
        points.append(qm.PointStruct(id=_id, vector=vec, payload={"text": ch, "meta": meta}))
        ids.append(_id); texts.append(ch); metas.append(meta)

    store.upsert_chunks(points)

    # BM25 build + save
    bm25 = BM25Store()
    bm25.build(ids, texts, metas)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(str(BM25_PATH))

    return {"files": len(files), "chunks": len(chunks_all), "updated_at": time.time()}

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for r, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return scores
def load_bm25():
    if BM25_PATH.exists():
        return BM25Store.load(str(BM25_PATH))
    return None
def hybrid_retrieve(query: str, dense_k: int = 60, bm25_k: int = 60, fused_k: int = 30):
    """
    return list of dict:
      {
        "id": int,
        "text": str,
        "meta": dict,
        "dense_score": float,
        "fused_score": float,
      }
    """
    bm25 = load_bm25()

    dense_hits = store.search(query, top_k=dense_k)  # (id, text, meta, score_dense)
    dense_ids = [h[0] for h in dense_hits]
    dense_map = {h[0]: h for h in dense_hits}

    bm25_ids = []
    if bm25:
        bm25_pairs = bm25.query(query, top_k=bm25_k)  # [(id, bm25_score)]
        bm25_ids = [p[0] for p in bm25_pairs]

    fused_scores = rrf_fuse([dense_ids, bm25_ids], k=60)
    fused_ids = sorted(fused_scores.keys(), key=lambda _id: fused_scores[_id], reverse=True)[:fused_k]

    items = []
    for _id in fused_ids:
        if _id in dense_map:
            _id, text, meta, dense_score = dense_map[_id]
            items.append({
                "id": int(_id),
                "text": text,
                "meta": meta,
                "dense_score": float(dense_score),
                "fused_score": float(fused_scores[_id]),
            })
        else:
            # 只从 bm25 来但不在 dense 里：先跳过（后面可做 qdrant retrieve by id）
            pass

    return items
# -----------------------------
# Sentence helpers
# -----------------------------
def split_sentences(text: str):
    sents = [s.strip() for s in re.split(r"[。！？；\n]+", text) if s.strip()]
    # 过滤残片
    sents = [s for s in sents if len(s) >= 8]
    return sents


# -----------------------------
# Router (local LLM)
# -----------------------------
def route_query_local(q: str) -> str:
    prompt = f"""你是分类器。把用户问题分成两类：
- fact: 查询单一事实（身高/学校/联系方式/时间点/某个具体值）
- summary: 需要总结/列举（项目/经历/技能/做过什么/优势/评价）

只输出一个词：fact 或 summary。

问题：{q}
"""
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    out = (r.json().get("response") or "").strip().lower()
    return "summary" if "summary" in out else "fact"

@app.get("/profile")
def get_profile():
    return {"profile": load_profile()}

@app.post("/profile/rebuild")
def rebuild_profile_only():
    files = sorted([p for p in UPLOAD_DIR.iterdir() if p.is_file()])
    file_texts = []
    for f in files:
        t = extract_text_from_file(f)
        if t and t.strip():
            file_texts.append((f.name, t))
    profile = extract_profile_from_texts(file_texts)
    save_profile(profile)
    return {"ok": True, "fields": list(profile.keys())}



def ollama_generate(prompt: str, model: str = "qwen2.5:7b", temperature: float = 0.0):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        },
        timeout=120,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def retrieve(query: str, top_k: int = TOP_K):
    hits = store.search(query, top_k=top_k)  # (id, text, meta, score)
    # 让后续代码保持 (doc, meta, score) 也行，或者全链路用带 id 的

    return [(text, meta, score) for (_id, text, meta, score) in hits]


def generate_answer_template(question: str, retrieved_items) -> str:
    if not retrieved_items:
        return "我在已上传资料中没有检索到相关信息，因此不确定答案。你可以补充材料或直接问我本人。"

    bullets = []
    for doc, meta, score in retrieved_items:
        preview = doc[:140].replace("\n", " ")
        bullets.append(f"-（来自 {meta.get('source')}）{preview}…")

    return (
        f"我会基于你上传的资料回答。\n\n"
        f"你的问题：{question}\n\n"
        f"我检索到的相关证据片段如下：\n" + "\n".join(bullets)
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "phase": 2}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name
    save_path = UPLOAD_DIR / safe_name

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size = save_path.stat().st_size
    return {"filename": safe_name, "size": size, "saved_to": str(save_path)}

@app.get("/files")
def list_files():
    files = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if p.is_file():
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return {"files": files}

# 索引状态：通过 collection count 来粗略统计
@app.get("/index/status", response_model=IndexStatus)
def index_status():
    file_count = len([p for p in UPLOAD_DIR.iterdir() if p.is_file()])
    try:
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0
    return IndexStatus(files=file_count, chunks=chunk_count, updated_at=None)

@app.post("/index/rebuild")
def index_rebuild():
    info = rebuild_index()
    return info

def expand_queries(q: str, n_rewrites: int = 3, n_keywords: int = 3) -> list[str]:
    prompt = f"""你是检索查询扩展器。给定用户问题，生成：
1) {n_rewrites} 条语义等价但表达不同的“改写问题”
2) {n_keywords} 条适合检索的“关键词短查询”（尽量短，保留核心名词）
只输出 JSON，格式：
{{"rewrites":[...], "keywords":[...]}}
问题：{q}
"""
    try:
        out = ollama_generate(prompt, model="qwen2.5:1.5b")
    except Exception:
        out = ""

    # 兜底：解析失败就只用原问题（不是关键词补丁，是鲁棒性）
    import json
    rewrites, keywords = [], []
    try:
        data = json.loads(out[out.find("{"): out.rfind("}") + 1])
        rewrites = [x.strip() for x in data.get("rewrites", []) if isinstance(x, str) and x.strip()]
        keywords = [x.strip() for x in data.get("keywords", []) if isinstance(x, str) and x.strip()]
    except Exception:
        pass

    # 去重 + 保留原 query
    qs = [q] + rewrites + keywords
    seen, dedup = set(), []
    for x in qs:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[: 1 + n_rewrites + n_keywords]

def decide_mode_by_evidence(q: str, top_sentences: list[str]) -> str:
    """
    返回: "fact" / "summary" / "refuse"
    不用问题关键词；用“证据形态”+ LLM 融合，避免误判。
    """
    # --- 证据形态信号（强泛化，不是问题补丁） ---
    s0 = (top_sentences[0] if top_sentences else "").strip()

    # 1) 单句很短 + 包含明确数值/日期/比例/邮箱/电话等 “原子事实” 信号 => 强推 fact
    has_number = bool(re.search(r"[0-9０-９]", s0))
    short_sentence = len(s0) <= 40
    atomic_like = bool(re.search(r"(身高|体重|毕业|本科|硕士|电话|邮箱|地址|现居|出生|年龄|学校|大学)", s0))  # 注意：这是证据形态词，不是问题词
    # ↑ 如果你觉得这里也像补丁，可以删掉 atomic_like，只靠 has_number+short_sentence 也能稳很多

    if has_number and short_sentence:
        # 典型：身高180，体重75kg / 2017.09-2021.06 / 电话+xx
        return "fact"

    # 2) 证据句子数很少，且每句都很短/很像单点事实 => fact
    if top_sentences:
        shorts = sum(1 for s in top_sentences[:4] if len(s.strip()) <= 45)
        nums = sum(1 for s in top_sentences[:4] if re.search(r"[0-9０-９]", s))
        if shorts >= 2 and nums >= 1:
            return "fact"

    # --- LLM 判定（作为软信号） ---
    evidence = "\n".join([f"- {s}" for s in top_sentences[:6]])
    prompt = f"""你是一个“可回答性判断器”。根据【问题】和【证据句子】判断：
- fact: 证据中有明确的单一事实，可用一句话回答
- summary: 证据包含多个要点，可做总结/列举回答
- refuse: 证据不足或不相关，无法可靠回答

只输出一个词：fact 或 summary 或 refuse

【问题】
{q}

【证据句子】
{evidence}
"""
    out = ollama_generate(prompt, model="qwen2.5:1.5b", temperature=0.0).strip().lower()
    if "summary" in out:
        return "summary"
    if "fact" in out:
        return "fact"
    return "refuse"

def normalize_cands(items: list) -> List[Dict[str, Any]]:
    """
    统一候选格式为 dict:
      {"id": int, "text": str, "meta": dict, "score": float}
    兼容：
      - (text, meta, score)
      - (id, text, meta, score)
      - dict（已是目标格式则补齐字段）
    """
    out: List[Dict[str, Any]] = []

    if not items:
        return out

    for i, it in enumerate(items):
        # dict
        if isinstance(it, dict):
            _id = int(it.get("id", i))
            text = it.get("text") or it.get("document") or ""
            meta = it.get("meta") or it.get("metadata") or {}
            score = it.get("score", it.get("fused_score", it.get("dense_score", 0.0)))
            out.append({"id": _id, "text": str(text), "meta": dict(meta), "score": float(score)})
            continue

        # tuple/list
        if isinstance(it, (tuple, list)):
            if len(it) == 3:
                text, meta, score = it
                _id = i
            elif len(it) >= 4:
                _id, text, meta, score = it[0], it[1], it[2], it[3]
            else:
                # 非法形状，跳过
                continue

            out.append({
                "id": int(_id),
                "text": str(text),
                "meta": dict(meta) if isinstance(meta, dict) else {},
                "score": float(score),
            })
            continue

        # 其他类型，跳过
        continue

    return out

def normalize_question(q: str) -> str:
    prompt = f"""把用户问题改写成更标准、更明确、便于检索与回答的形式。
要求：保持原意；如果是口语/省略表达，补全主语与属性名。
只输出改写后的问题文本。

用户问题：{q}
"""
    out = ollama_generate(prompt, model="qwen2.5:1.5b").strip()
    return out if out else q

def extract_substring_answer(q: str, evidence: str) -> str | None:
    prompt = f"""你是“证据子串抽取器”。
从【证据】中抽取能回答【问题】的最短连续片段作为答案。
强约束：答案必须是证据中的原文连续子串（一字不改）。
输出格式必须为：
ANSWER: <答案>
如果证据无法回答，输出：
ANSWER: NONE

【问题】{q}
【证据】{evidence}
"""
    out = ollama_generate(prompt, model="qwen2.5:7b", temperature=0.0)

    m = re.search(r"ANSWER:\s*(.*)", out)
    if not m:
        return None
    ans = m.group(1).strip()
    if not ans or ans.upper() == "NONE":
        return None

    # ✅ 程序强校验：必须真的是证据子串
    if ans in evidence and len(ans) <= 60:
        return ans
    return None

def fallback_short_numeric_span(evidence: str) -> str | None:
    """
    通用数字兜底：从证据里找最短的“数值+可选单位”片段。
    不针对身高/体重/年龄，只要证据有数字就能救场。
    """
    m = re.search(r"(\d{1,4}(?:\.\d+)?\s*(?:cm|厘米|米|m|kg|公斤|岁|year|years)?)", evidence, flags=re.I)
    if m:
        return m.group(1).strip()
    return None

def verify_answer(q: str, evidence: str, ans: str, answer_type: str) -> bool:
    if not ans or "未提及" in ans or "不确定" in ans:
        return False
    ans = ans.strip()
    if len(ans) > 30:
        return False
    if ans not in evidence:
        return False

    # ✅ entity 问题：拒绝纯数字答案
    if answer_type == "entity" and re.fullmatch(r"[0-9０-９.．\-–—]+", ans):
        return False

    return True

def probe_fact(q: str, answer_type: str, top_sents: list[str], top_sent_meta: list[tuple]) -> tuple[str, str, float] | None:

    """
    依次尝试从 top_sents 抽取一个“短答案”并通过 verify。
    返回 (answer, evidence_sentence, score)；失败返回 None
    """
    for idx, (sent, meta) in enumerate(zip(top_sents[:6], top_sent_meta[:6])):
        src, sec, sc = meta

        # 1) 子串抽取（更稳）
        ans = extract_substring_answer(q, sent)

        # 2) 数字兜底
        if not ans and answer_type == "number":
            ans = fallback_short_numeric_span(sent)

        ok = verify_answer(q, sent, ans, answer_type) if ans else False


        # DEBUG（先留着定位，稳定后可删）
        print(f"DEBUG probe idx={idx} src={src} sc={float(sc):.4f} ans={ans} verify={ok}")
        # print("DEBUG sent:", sent)

        if ans and ok:
            return ans, sent, float(sc)

    return None

def minmax_norm(xs):
    if not xs: return []
    mn, mx = min(xs), max(xs)
    if mx - mn < 1e-9:
        return [0.5 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


class Route(TypedDict):
    intent: Literal["fact", "summary", "refuse"]
    answer_type: Literal["number", "entity", "text"]
    evidence_need: Literal["strict", "normal"]
    output_style: Literal["one_line", "bullets"]

def classify_route(q: str) -> Route:
    prompt = f"""你是问题路由分类器。根据【问题】判断回答策略。
只输出 JSON，不能输出其他文字。

字段说明：
- intent: fact / summary / refuse
- answer_type: number / entity / text
- evidence_need: strict / normal
- output_style: one_line / bullets

判断准则：
- fact：一句话能回答的个人事实（学校/身高/时间/地点/联系方式等）
- summary：需要总结多个点（项目经历/能力/优势/自我介绍）
- refuse：资料不足或问题超出资料范围

- answer_type:
  - number：数字/日期/区间
  - entity：学校/公司/地点/人名/职位等实体
  - text：短文本描述

【问题】{q}
"""
    out = ollama_generate(prompt, model="qwen2.5:7b", temperature=0.0).strip()

    # 容错解析：只截取第一段 JSON
    try:
        j = json.loads(out[out.find("{"): out.rfind("}") + 1])
    except Exception:
        # 兜底默认：summary 更安全
        return {"intent": "summary", "answer_type": "text", "evidence_need": "normal", "output_style": "bullets"}

    # 规范化 + 兜底
    intent = j.get("intent", "summary")
    answer_type = j.get("answer_type", "text")
    evidence_need = j.get("evidence_need", "normal")
    output_style = j.get("output_style", "bullets")

    if intent not in ("fact", "summary", "refuse"):
        intent = "summary"
    if answer_type not in ("number", "entity", "text"):
        answer_type = "text"
    if evidence_need not in ("strict", "normal"):
        evidence_need = "normal"
    if output_style not in ("one_line", "bullets"):
        output_style = "bullets"

    return {
        "intent": intent,
        "answer_type": answer_type,
        "evidence_need": evidence_need,
        "output_style": output_style,
    }


def answer_from_profile(q: str) -> Tuple[str, Citation] | None:
    p = load_profile()
    q2 = q.strip()

    def hit(key: str, label: str):
        obj = p.get(key)
        if not obj:
            return None
        value = obj.get("value", "").strip()
        ev = obj.get("evidence", "").strip()
        src = obj.get("source", "unknown")
        if not value:
            return None
        # 只输出最终答案（更像你要的“一个结果”）
        return value, Citation(source=src, excerpt=ev[:260].replace("\n", " "), score=1.0)

    # 关键词映射（这是“路由”，不是补丁：稳定的事实层入口）
    if any(x in q2 for x in ["哪一年出生", "出生哪年", "出生年份", "出生年", "几几年出生"]):
        return hit("birth_year", "出生年份")

    if any(x in q2 for x in ["出生", "生日", "几月生"]):
        # 先年再月
        r = hit("birth_year", "出生年份")
        if r:
            return r
        return hit("birth_month", "出生月份")

    if any(x in q2 for x in ["性别", "男的女的", "男还是女", "你是男的", "你是女的"]):
        return hit("gender", "性别")

    if any(x in q2 for x in ["身高", "多高", "你多高"]):
        return hit("height_cm", "身高")

    if any(x in q2 for x in ["体重", "多重", "你多重"]):
        return hit("weight_kg", "体重")

    if any(x in q2 for x in ["专业", "学的什么", "主修"]):
        return hit("undergrad_major", "本科专业")

    if any(x in q2 for x in ["本科", "大学", "哪个学校", "毕业于", "毕业哪里", "哪所大学"]):
        return hit("undergrad_school", "本科院校")

    return None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q_raw = (req.message or "").strip()
    if not q_raw:
        return ChatResponse(answer="请输入问题。", citations=[])

    # 0) smalltalk / meta
    if is_smalltalk(q_raw):
        return ChatResponse(
            answer="你好！我在～你想了解我哪方面：教育背景 / 项目经历 / 实习工作 / 技能栈 / 个人亮点？",
            citations=[]
        )

    if is_meta_question(q_raw):
        return ChatResponse(
            answer=(
                "我主要能做这些：\n"
                "- 基于你上传的资料进行问答，并给出引用证据\n"
                "- 从资料中总结项目/经历亮点（3~6条要点）\n"
                "- 对事实类问题给出一句话答案（例如学校/身高/时间等），资料不足会明确说不确定\n\n"
                "你可以先上传简历/项目材料，然后试试问：\n"
                "1) 你做过什么项目？\n"
                "2) 你最擅长的技术栈是什么？\n"
                "3) 你在某个项目里做了什么、结果如何？"
            ),
            citations=[]
        )

    # 1) normalize
    q0 = q_raw
    q = normalize_question(q_raw)
    print(f"DEBUG normalize: {q0} => {q}", flush=True)

    # 2) profile quick hit（可选）
    profile_hit = answer_from_profile(q)
    if profile_hit:
        ans, cit = profile_hit
        return ChatResponse(answer=ans, citations=[cit])

    # 3) route（决定 fact / summary）
    route = classify_route(q)
    intent = route.get("intent", "summary")
    if intent == "summary":
        queries = [q]   # summary 不做扩展，保证稳定命中“项目标题块”
    else:
        queries = expand_queries(q)

    
    intent = (route or {}).get("intent", "summary")  # "fact" / "summary"

    # 4) 多查询扩展
    queries = expand_queries(q)

    # 5) Hybrid 召回 + merge 去重
    cand_map = {}  # id -> dict{id,text,meta,score}
    for qq in queries:
        raw = hybrid_retrieve(qq, dense_k=80, bm25_k=80, fused_k=40)
        raw = normalize_cands(raw)  # 统一成 dict list

        for it in raw:
            _id = it.get("id", -1)
            if _id == -1:
                continue
            key_score = float(it.get("score", 0.0))
            if (_id not in cand_map) or (key_score > float(cand_map[_id].get("score", 0.0))):
                cand_map[_id] = it

    cands = list(cand_map.values())
    cands = normalize_cands(cands)  # 再保险
    if not cands:
        return ChatResponse(
            answer="我在资料中没有检索到足够可靠的证据来回答这个问题（为避免编造，我选择不回答）。",
            citations=[]
        )

    # 初排（融合分）排序
    cands.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    

    # 6) chunk 级 rerank（更准）
    passages = [c["text"] for c in cands]
    rr_scores = rerank(q, passages)  # 越大越相关

    # 把 rerank 分数写回（用于 summary 的 section 多样性挑选）
    cands_rr = []
    for i, c in enumerate(cands):
        cc = dict(c)
        cc["retrieval_score"] = float(cc.get("score", 0.0))   # 保留召回分
        cc["rerank_score"] = float(rr_scores[i])             # 另存精排分
        cands_rr.append(cc)
    cands_rr.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)

    # 7) 句子池（用于 fact 抽取 / 可回答性）
    TOPN_CHUNKS = 8
    top_chunks = cands_rr[:TOPN_CHUNKS]

    sent_pool = []  # (sent, source, section_title, sent_score_seed)
    for it in top_chunks:
        meta = it.get("meta", {}) or {}
        source = meta.get("source", "unknown")
        section_title = meta.get("section_title") or meta.get("section") or ""
        sents = split_sentences(it["text"])
        if not sents:
            sents = [it["text"][:220].strip()]
        for s in sents[:12]:
            sent_pool.append((s, source, section_title, float(it.get("score", 0.0))))

    sent_texts = [s[0] for s in sent_pool]
    sent_scores = rerank(q, sent_texts)
    sent_order = sorted(range(len(sent_pool)), key=lambda i: sent_scores[i], reverse=True)

    top_sents = []
    top_sent_meta = []  # (source, section_title, score)
    for i in sent_order[:20]:
        s, source, section_title, _seed = sent_pool[i]
        sc = float(sent_scores[i])
        if sc < 0.02:
            continue
        top_sents.append(s)
        top_sent_meta.append((source, section_title, sc))
        if len(top_sents) >= 6:
            break

    if not top_sents:
        return ChatResponse(
            answer="我检索到了一些内容，但无法提取到足够可靠的证据来回答（为避免编造，我选择不回答）。",
            citations=[]
        )

    # ==========================
    # ✅ A) FACT 路由：才允许 probe_fact 抢答
    # ==========================
    if intent == "fact":
        # 先 probe_fact（短答案抽取 + verify）
        hit = probe_fact(q, route.get("answer_type", "text"), top_sents, top_sent_meta)
        if hit:
            ans, evidence, sc = hit

            # ⚠️ 这里不要用 top_sent_meta[0]，要找 evidence 对应的 source
            src = "unknown"
            for (s, (source, _sec, _sc)) in zip(top_sents, top_sent_meta):
                if s == evidence:
                    src = source
                    break

            citations = [Citation(source=src, excerpt=evidence, score=float(sc))]
            return ChatResponse(answer=ans, citations=citations)

        # probe 不命中，再走基于证据的模式判定
        mode = decide_mode_by_evidence(q, top_sents)
        print("DEBUG mode:", mode, flush=True)
        print("DEBUG top_sents[0]:", top_sents[0], flush=True)

        if mode == "refuse":
            return ChatResponse(
                answer="我在现有资料中没有找到足够明确且相关的证据来回答这个问题（为避免编造，我选择不回答）。你可以补充更相关的材料或换个更具体的问法。",
                citations=[]
            )

        # mode == "fact"
        evidence = top_sents[0]
        src, _sec, sc = top_sent_meta[0]

        ans = extract_substring_answer(q, evidence)
        if not ans:
            ans = fallback_short_numeric_span(evidence)
        if not ans:
            ans = "资料中未提及，我不确定。"

        print("DEBUG evidence:", evidence, flush=True)
        print("DEBUG span:", extract_substring_answer(q, evidence),
              "fallback:", fallback_short_numeric_span(evidence), flush=True)

        return ChatResponse(
            answer=ans,
            citations=[Citation(source=src, excerpt=evidence, score=float(sc))]
        )

    # ==========================
    # ✅ B) SUMMARY 路由：禁止 probe_fact 抢跑
    # ==========================
    # 多 section 取材（保证“你做过哪些项目”能拿到两个项目）
    diverse = pick_diverse_by_section(cands_rr, max_sections=6)  # 这里传 dict list

    # 证据块：只拼接最相关的几个 chunk（别太长）
    evidence_block = "\n\n".join([c["text"][:900] for c in diverse[:4]])

    prompt = f"""你是一个基于证据回答的助手。
只能使用【证据】中的信息回答【问题】，不要编造。

请输出“项目清单”，每个项目一条，格式固定：
- 项目名：一句话概括（你做了什么 + 技术/组件 + 目的/效果）

如果证据里有多个项目，必须全部列出；不要漏掉。

【问题】
{q}

【证据】
{evidence_block}
"""
    answer = ollama_generate(prompt, model="qwen2.5:7b").strip()
    if not answer:
        answer = "资料中未提及，我不确定。"
    scores = [float(c.get("rerank_score", 0.0)) for c in diverse]
    normed = minmax_norm(scores)
    for c, s01 in zip(diverse, normed):
        c["score"] = float(s01)   # 统一展示分

    # citations：最多给 2~3 条（每个项目/section 一条更清晰）
    citations = []
    used = set()
    for c in diverse:
        meta = c.get("meta", {}) or {}
        src = meta.get("source", "unknown")
        sec = meta.get("section_title") or meta.get("section") or ""
        key = (src, sec)
        if key in used:
            continue
        used.add(key)
        citations.append(Citation(
            source=src,
            excerpt=c["text"][:260].replace("\n", " ").strip(),
            score=float(c.get("score", 0.0))
        ))
        if len(citations) >= 3:
            break

    return ChatResponse(answer=answer, citations=citations)
