"""
PDF 統整 + 個人文件 RAG 資料庫（2026/9/30 新增，霽倫閣下的需求）。

流程：
  1. 使用者在 LINE 傳 PDF → 抽文字 → 先用本地 LLM 統整一份摘要回報
  2. 回報時附 Quick Reply 詢問「要不要加入資料庫」，選了才切塊、算
     embedding、存進 Chroma；不加入的話檔案跟紀錄都直接清掉
  3. 之後一般聊天時「自動判斷」要不要查這些文件，不需要特殊指令：
       - 先做語意檢索，最高分低於門檻就當作跟文件無關，走一般聊天
       - 過了門檻，再把檢索到的段落交給 LLM，請它判斷段落到底能不能
         回答這個問題，不能就回「無關」，一樣退回一般聊天
     兩段式把關，避免隨便一句閒聊也被硬套上文件內容。
  4. 可以刪除指定的 PDF：Chroma 裡對應的段落、原始檔、紀錄一起刪掉

跟 nutrition_lookup 共用同一個 bge-m3 embedding 模型，不另外載一份
（模型本身就佔好幾 GB 記憶體，兩個 bot 又在同一個 process 裡跑）。

文件紀錄存在 SQLite（pdf_documents 表），向量存在獨立的 Chroma
collection，metadata 帶 doc_id/user_id，刪除時用 where 條件一次清掉。
"""

import json
import os
import re
import sqlite3
import subprocess
import tempfile
import traceback
import urllib.request
from datetime import datetime

import chromadb
from pypdf import PdfReader

import nutrition_lookup

OLLAMA_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "llama3.2:latest"

PDF_DIR = r"D:\CalorieCalculation\data\pdf_docs\files"
DB_PATH = r"D:\CalorieCalculation\data\pdf_docs\pdf_docs.db"
CHROMA_PATH = r"D:\CalorieCalculation\data\pdf_docs\chroma_store"
COLLECTION_NAME = "pdf_docs"

MAX_PDF_BYTES = 20 * 1024 * 1024

# 文字抽取優先用 MinerU（OCR + 版面/表格辨識），裝在獨立 venv，避免它依賴
# 的 CUDA 版 torch 跟主環境（CLIP、bge-m3 用的 CPU 版）打架。實測
# source_manual_2023.pdf：pypdf 只抽得到約 2300 字，表格幾乎全漏；MinerU
# 抽出約 15900 字元，速查表、水果份量表都完整，數字跟衛福部代換表吻合。
# MinerU 不可用或失敗時退回 pypdf，功能不會整個掛掉。
# 一定只用本地解析：不要加 --remote（那會把文件上傳到 mineru.net）。
MINERU_EXE = r"D:\venvs\mineru\Scripts\mineru.exe"
MINERU_PARSE_TIMEOUT = 900
_PAGE_MARKER = re.compile(r"<!-- page (\d+) of \d+ -->")

# 統整用：每段餵給 LLM 的字數上限 / 最多分幾段（map-reduce）。
# llama3.2 開 8k context，中文大約一字一 token，單段抓 5000 字留空間給
# 提示詞跟輸出；超過 MAX_SUMMARY_SEGMENTS 段的超長文件只統整前面的部分，
# 並在摘要裡註明，不假裝整份都看完了。
SUMMARY_SEGMENT_CHARS = 5000
MAX_SUMMARY_SEGMENTS = 10

# RAG 切塊：以頁為單位切，跨頁不合併，方便回覆時標出頁碼。
CHUNK_CHARS = 600
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 30

# 自動判斷要不要用 RAG 的語意相似度門檻（cosine similarity）。
# 問題裡有明確提到文件/PDF 時門檻放寬，沒提到時要更相關才會啟用。
RAG_THRESHOLD = 0.55
RAG_THRESHOLD_EXPLICIT = 0.40
RAG_TOP_K = 4
DOC_HINT_KEYWORDS = ["pdf", "文件", "檔案", "報告", "講義", "論文", "資料裡", "裡面", "上面寫"]

IRRELEVANT_MARK = "無關"

_collection = None


# ---------- 紀錄層 ----------

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            page_count INTEGER,
            summary TEXT,
            status TEXT NOT NULL,        -- summarized（已統整、等決定）/ in_rag（已加入資料庫）
            chunk_count INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def _create_doc(user_id: str, filename: str, file_path: str) -> int:
    init_db()
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO pdf_documents (user_id, created_at, filename, file_path, status) VALUES (?, ?, ?, ?, 'summarized')",
        (user_id, datetime.now().isoformat(timespec="seconds"), filename, file_path),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def _update_doc(doc_id: int, **fields) -> None:
    columns = ", ".join(f"{k} = ?" for k in fields)
    conn = _connect()
    conn.execute(f"UPDATE pdf_documents SET {columns} WHERE id = ?", list(fields.values()) + [doc_id])
    conn.commit()
    conn.close()


def get_doc(doc_id: int) -> dict:
    init_db()
    conn = _connect()
    row = conn.execute("SELECT * FROM pdf_documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_docs(user_id: str, rag_only: bool = False) -> list:
    init_db()
    sql = "SELECT * FROM pdf_documents WHERE user_id = ?"
    if rag_only:
        sql += " AND status = 'in_rag'"
    conn = _connect()
    rows = conn.execute(sql + " ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def has_rag_docs(user_id: str) -> bool:
    return bool(list_docs(user_id, rag_only=True))


def find_docs(user_id: str, text: str) -> list:
    """從一句話裡找出使用者指的是哪份文件：#編號 或 檔名（含去掉 .pdf 的主檔名）。"""
    docs = list_docs(user_id)
    lowered = text.lower()
    matched = []
    for d in docs:
        stem = os.path.splitext(d["filename"])[0].lower()
        if f"#{d['id']}" in text or d["filename"].lower() in lowered or (len(stem) >= 2 and stem in lowered):
            matched.append(d)
    return matched


# ---------- LLM / embedding ----------

def _ollama(prompt: str, temperature: float = 0.2, num_predict: int = 800, timeout: int = 300) -> str:
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def _embed_model():
    nutrition_lookup._load()
    return nutrition_lookup._embed_model


def _get_collection():
    global _collection
    if _collection is None:
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection


# ---------- PDF 處理 ----------

def save_pdf(user_id: str, filename: str, content: bytes) -> int:
    """把 LINE 下載下來的 PDF 存檔並建立紀錄，回傳 doc_id。"""
    init_db()
    safe_name = "".join(c for c in filename if c not in '\\/:*?"<>|') or "document.pdf"
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(PDF_DIR, f"{stamp}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(content)
    return _create_doc(user_id, filename, file_path)


def _run_mineru(args: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [MINERU_EXE] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


def _clean_markdown(md: str) -> str:
    """把 MinerU 的 Markdown 整理成適合切塊/餵 LLM 的純文字：去掉圖片佔位、
    HTML 表格轉成「欄 | 欄」、移除全空白的表格列。"""
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", md)
    md = re.sub(r"</tr>", "\n", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"^\|[\s|]*\|\s*$", "", md, flags=re.M)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _extract_pages_mineru(file_path: str) -> list:
    if not os.path.exists(MINERU_EXE):
        raise RuntimeError(f"找不到 MinerU：{MINERU_EXE}")
    _run_mineru(["server", "start"], timeout=120)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "out.md")
        result = _run_mineru(
            ["parse", file_path, "--tier", "standard", "--pages", "all",
             "--wait", str(MINERU_PARSE_TIMEOUT), "-o", out_path],
            timeout=MINERU_PARSE_TIMEOUT + 60,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"MinerU 解析失敗：{(result.stderr or result.stdout)[-500:]}")
        with open(out_path, encoding="utf-8") as f:
            md = f.read()

    parts = _PAGE_MARKER.split(md)
    pages = []
    for i in range(1, len(parts), 2):
        text = _clean_markdown(parts[i + 1])
        if text:
            pages.append((int(parts[i]), text))
    return pages


def _extract_pages_pypdf(file_path: str) -> list:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _pages_cache_path(file_path: str) -> str:
    return file_path + ".pages.json"


def extract_pages(file_path: str) -> list:
    """回傳 [(頁碼, 文字), ...]，只保留有文字的頁面。

    結果快取在 PDF 旁邊的 .pages.json：統整跟加入資料庫都要抽一次文字，
    MinerU 一份文件要解析十幾秒以上，不要做兩次。"""
    cache = _pages_cache_path(file_path)
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return [tuple(p) for p in json.load(f)]

    pages = []
    try:
        pages = _extract_pages_mineru(file_path)
    except Exception:
        traceback.print_exc()
    if not pages:
        pages = _extract_pages_pypdf(file_path)

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)
    return pages


def _split_segments(full_text: str) -> tuple:
    segments = [full_text[i:i + SUMMARY_SEGMENT_CHARS] for i in range(0, len(full_text), SUMMARY_SEGMENT_CHARS)]
    truncated = len(segments) > MAX_SUMMARY_SEGMENTS
    return segments[:MAX_SUMMARY_SEGMENTS], truncated


def summarize_doc(doc_id: int) -> str:
    """抽文字 → 分段統整 → 合併成一份摘要，存回紀錄並回傳。抽不到文字時丟 ValueError。"""
    doc = get_doc(doc_id)
    pages = extract_pages(doc["file_path"])
    if not pages:
        raise ValueError("這份 PDF 抽不到任何文字（可能是掃描圖片檔），目前沒有 OCR 功能。")

    full_text = "\n".join(text for _, text in pages)
    segments, truncated = _split_segments(full_text)

    if len(segments) == 1:
        summary = _ollama(
            "請用繁體中文統整底下這份文件，先用一兩句話說明文件主旨，再條列 3~8 個重點，"
            "只根據文件內容，不要自己補充文件沒寫的資訊。\n\n"
            f"文件：{doc['filename']}\n內容：\n{segments[0]}\n\n統整："
        )
    else:
        partials = []
        for i, seg in enumerate(segments, start=1):
            partials.append(
                _ollama(
                    f"底下是文件「{doc['filename']}」的第 {i}/{len(segments)} 段，請用繁體中文條列這段的重點"
                    f"（最多 5 點），只根據內容，不要補充。\n\n內容：\n{seg}\n\n重點：",
                    num_predict=400,
                )
            )
        joined = "\n\n".join(f"[第 {i} 段]\n{p}" for i, p in enumerate(partials, start=1))
        summary = _ollama(
            "底下是同一份文件各段的重點整理，請用繁體中文合併成一份完整統整：先用一兩句話說明"
            "文件主旨，再條列 5~10 個最重要的重點，去掉重複，不要補充原文沒有的資訊。\n\n"
            f"文件：{doc['filename']}\n{joined}\n\n統整："
        )

    if truncated:
        summary += f"\n\n（文件太長，只統整了前 {MAX_SUMMARY_SEGMENTS * SUMMARY_SEGMENT_CHARS} 字左右的內容）"

    _update_doc(doc_id, summary=summary, page_count=len(PdfReader(doc["file_path"]).pages))
    return summary


def _chunk_pages(pages: list) -> list:
    chunks = []
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for page_no, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_CHARS].strip()
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append((page_no, piece))
            if start + CHUNK_CHARS >= len(text):
                break
    return chunks


def add_to_rag(doc_id: int) -> int:
    """把文件切塊存進 Chroma，回傳段落數。已經加入過的會先清掉舊段落再重建。"""
    doc = get_doc(doc_id)
    chunks = _chunk_pages(extract_pages(doc["file_path"]))
    if not chunks:
        raise ValueError("這份 PDF 沒有可以切塊的文字內容。")

    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})

    model = _embed_model()
    batch_size = 32
    for b in range(0, len(chunks), batch_size):
        batch = chunks[b:b + batch_size]
        texts = [t for _, t in batch]
        collection.add(
            ids=[f"{doc_id}-{b + i}" for i in range(len(batch))],
            documents=texts,
            embeddings=model.get_text_embedding_batch(texts),
            metadatas=[
                {"doc_id": doc_id, "user_id": doc["user_id"], "filename": doc["filename"], "page": page_no}
                for page_no, _ in batch
            ],
        )

    _update_doc(doc_id, status="in_rag", chunk_count=len(chunks))
    return len(chunks)


def delete_doc(doc_id: int) -> bool:
    """從 RAG 資料庫刪掉這份文件的所有段落，連同原始檔跟紀錄一起刪。"""
    doc = get_doc(doc_id)
    if doc is None:
        return False
    if doc["status"] == "in_rag":
        _get_collection().delete(where={"doc_id": doc_id})
    # MinerU 自己的文件庫也留了一份解析快取，一起清掉才算真的刪乾淨。
    if os.path.exists(MINERU_EXE):
        try:
            _run_mineru(["forget", doc["file_path"], "--no-dry-run"], timeout=60)
        except Exception:
            traceback.print_exc()
    for path in (doc["file_path"], _pages_cache_path(doc["file_path"])):
        if os.path.exists(path):
            os.remove(path)
    conn = _connect()
    conn.execute("DELETE FROM pdf_documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True


# ---------- 聊天時的自動 RAG ----------

def retrieve(user_id: str, question: str, top_k: int = RAG_TOP_K) -> list:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    embedding = _embed_model().get_query_embedding(question)
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "score": round(1 - dist, 4), **meta})
    return hits


GROUNDED_ANSWER_PROMPT = (
    "底下是從使用者自己上傳的文件裡檢索到的段落。請嚴格遵守規則：\n"
    "1. 問題問到的那個品項或主題，必須在段落裡「原文直接出現」，而且段落直接寫了答案，才可以回答。\n"
    "2. 回答時照抄段落裡的數字與單位，不可以推算、加總、換算，也不可以拿別的品項來類比。\n"
    f"3. 段落裡沒有出現問題問的品項、或沒有直接寫答案、或問題只是閒聊，只回覆「{IRRELEVANT_MARK}」兩個字，不要多說。\n"
    "4. 用繁體中文回答。\n\n"
)


def answer_with_rag(user_id: str, question: str) -> str:
    """判斷這句話需不需要查文件；需要就回傳根據文件的回答，不需要回傳 None（交給一般聊天）。

    刻意不套角色人設、溫度 0：2026/10/2 評測（source_manual_2023.pdf，6 題 × 3 次）
    - 原本的寬鬆提示詞 + 溫度 0.2：文件裡沒有的題目會偶發編造（牛肉麵 50~60 大卡）
    - 前面加上 Ryuzu 人設：文件裡沒有的題目 6/6 編造，還掛上手冊當出處
      （「根據《食物份量代換手冊》，一碗牛肉麵 120 大卡」）
    - 現在這組（無人設 + 嚴格規則 + 溫度 0）：18 次零編造，沒有的題目每次都拒答
    """
    if not has_rag_docs(user_id):
        return None

    hits = retrieve(user_id, question)
    if not hits:
        return None

    explicit = any(k in question.lower() for k in DOC_HINT_KEYWORDS)
    threshold = RAG_THRESHOLD_EXPLICIT if explicit else RAG_THRESHOLD
    hits = [h for h in hits if h["score"] >= threshold]
    if not hits:
        return None

    context = "\n\n".join(f"[{h['filename']} 第{h['page']}頁]\n{h['text']}" for h in hits)
    answer = _ollama(
        f"{GROUNDED_ANSWER_PROMPT}段落：\n{context}\n\n問題：{question}\n回答：",
        temperature=0.0,
        num_predict=600,
        timeout=120,
    )
    cleaned = (answer or "").strip().strip("。.「」\"'")
    if not cleaned or (cleaned.startswith(IRRELEVANT_MARK) and len(cleaned) <= 8):
        return None

    sources = []
    for h in hits:
        label = f"{h['filename']} p.{h['page']}"
        if label not in sources:
            sources.append(label)
    return f"{answer}\n\n📄 參考：{'、'.join(sources)}"


if __name__ == "__main__":
    init_db()
    print(f"initialized {DB_PATH}")
