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
import sqlite3
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


def extract_pages(file_path: str) -> list:
    """回傳 [(頁碼, 文字), ...]，只保留有文字的頁面。掃描檔（純圖片）會是空的。"""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
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
    if os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])
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


def answer_with_rag(user_id: str, question: str, persona_prompt: str = "") -> str:
    """判斷這句話需不需要查文件；需要就回傳根據文件的回答，不需要回傳 None（交給一般聊天）。"""
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
        f"{persona_prompt}\n\n"
        "底下是從使用者自己上傳的文件裡檢索到的段落。請先判斷這些段落跟問題有沒有關係：\n"
        f"- 如果段落跟問題無關、或問題只是閒聊，只回覆「{IRRELEVANT_MARK}」兩個字，不要多說\n"
        "- 如果有關，只根據段落內容用繁體中文回答，段落沒寫的不要自己編，資訊不夠就老實說\n\n"
        f"段落：\n{context}\n\n問題：{question}\n回答：",
        temperature=0.2,
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
