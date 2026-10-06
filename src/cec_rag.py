"""
CEC 建築 Revit API 查詢助手的 RAG（規格：data/CEC_Revit API/AIRAGUse.md）。

流程（照規格書第 5 節，判斷與比對都用程式規則，不讓小模型猜）：
  ① 問題類型：非本庫範圍（機電/土木/Autodesk）直接轉介；通用類（授權、灰色、
     安裝、Excel…）檢索範圍加入 _通用問題；「有沒有功能可以…」查 _功能目錄
  ② 目錄比對：使用者的話比對每張卡片的中文名/英文名/別名（容錯：梁/樑、驅/軀、版/板）
     1 個 → 只搜那支按鈕 + _通用問題；多個 → 回問；0 個 → ③
  ③ 錯誤訊息：去掉空白標點後，跟所有卡片「」裡的訊息原文做字串比對，找到就直接用那幾段
  ④ 向量檢索（bge-m3，跟熱量估算共用同一份模型）
  ⑤ LLM 只根據檢索段落回答，說明頁網址由程式附上（不讓模型抄網址，避免抄錯）

規格書寫的是 docs\ 子資料夾 + catalog.csv；實際交付的資料是卡片直接放在
data/CEC_Revit API/、沒有 catalog.csv，所以目錄改由各卡片開頭的「中文名／英文名／
別名」欄位建立（規格書第 3.3 節也說別名可以拿來比對）。
"""

import hashlib
import json
import os
import re
import unicodedata
import urllib.request
from datetime import datetime

import chromadb
import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from opencc import OpenCC

import nutrition_lookup

# qwen3 偶爾會混出簡體字（「没有」）。逐字轉換（s2tw）也會把「台」轉成「臺」、「里」
# 轉成「裡」，改寫原文用字（實測「這台電腦」→「這臺電腦」）。所以只轉換知識庫原文
# 從來沒用過的字：原文出現過的字一定是正確的繁體用字，保持不動。
_s2tw = OpenCC("s2tw").convert
_kb_chars = None


def _to_traditional(text: str) -> str:
    global _kb_chars
    if _kb_chars is None:
        _kb_chars = set()
        for path in _kb_files():
            with open(path, encoding="utf-8-sig") as f:
                _kb_chars.update(f.read())
    converted = _s2tw(text)
    if len(converted) != len(text):  # s2tw 理論上逐字對應；萬一長度變了就整段採用轉換結果
        return converted
    return "".join(o if o in _kb_chars else c for o, c in zip(text, converted))

KB_DIR = r"D:\CalorieCalculation\data\CEC_Revit API"
STORE_DIR = r"D:\CalorieCalculation\data\cec_rag"
CHROMA_PATH = os.path.join(STORE_DIR, "chroma_store")
# 2026/10/6 常見問題、錯誤訊息對照改成一條一塊，換狀態檔讓所有卡片重新切段匯入
STATE_PATH = os.path.join(STORE_DIR, "ingest_state_llamaindex_v2.json")
LOG_PATH = os.path.join(STORE_DIR, "qa_log.jsonl")
COLLECTION_NAME = "cec_docs_llamaindex"  # 2026/10/6 改由 LlamaIndex 寫入，換新 collection 重建
EXCLUDE_FILES = {"AIRAGUse.md"}  # 給工程師看的實作說明，不是知識內容

GENERAL = "_通用問題"
CATALOG = "_功能目錄"

OLLAMA_URL = "http://localhost:11434/api/generate"
ANSWER_MODEL = "qwen3:8b"
TOP_K = 4
MAX_CHUNK_CHARS = 1200

# 通用問題：關鍵字直接對應到 _通用問題 的段落（程式規則決定，不靠向量檢索排序——
# 實測「為什麼按鈕都是灰的」向量檢索把「授權與註冊」排在「按鈕是灰色的不能按」前面，
# 回答就漏掉了視圖類型限制）。
GENERAL_SECTION_RULES = [
    (["灰色", "灰的", "反灰", "按不了", "不能按", "按不下"], "按鈕是灰色的不能按"),
    (["授權", "註冊", "密鑰", "金鑰", "本機id"], "授權與註冊"),
    (["安裝", "版本"], "安裝與支援版本"),
    (["excel"], "匯出 Excel 或讀取 Excel 的功能"),
    (["找誰", "聯絡", "窗口"], "找誰問（聯絡窗口）"),
    (["回報"], "遇到錯誤怎麼回報"),
    (["說明頁"], "說明頁怎麼看"),
    (["按鈕在哪", "找不到按鈕", "在哪個頁籤"], "按鈕在哪裡"),
]
GENERAL_KEYWORDS = [k for words, _ in GENERAL_SECTION_RULES for k in words]
FEATURE_SEARCH_KEYWORDS = ["有沒有", "哪個按鈕", "哪個功能", "什麼功能", "有什麼", "可以做", "能不能做", "有哪些"]
OUT_OF_SCOPE = [
    (["機電", "mep", "套管"], "機電"),
    (["土木", "civil", "lebr"], "土木"),
    (["autodesk", "登不進", "登入不了", "無法登入", "授權過期", "雲端"], "Autodesk"),
]
ERROR_HINT_WORDS = ["「", "請確認", "無法", "失敗", "錯誤", "error", "!"]

_cards = None          # api -> card dict
_error_index = None    # [(normalized 訊息, api, section 名稱)]
_collection = None
_vector_store = None
_index = None


# ---------- 文字正規化 ----------

def _normalize(s: str, for_error: bool = False) -> str:
    """全形轉半形、去空白標點、常見錯字統一（梁→樑、驅→軀、版→板）。"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("梁", "樑").replace("驅", "軀").replace("版", "板")
    if for_error:
        s = s.replace("xxx", "")
        s = re.sub(r"\d", "", s)
    return "".join(ch for ch in s if ch.isalnum())


# ---------- 讀卡片 ----------

HEADER_FIELD = re.compile(r"^- (中文名|英文名|別名|位置|可用視圖|說明頁)：(.*)$")


def _parse_card(path: str) -> dict:
    api = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    card = {"api": api, "title": "", "zh": "", "en": "", "aliases": [], "url": "", "views": "", "sections": []}
    current = None
    for line in lines:
        if line.startswith("# ") and not card["title"]:
            card["title"] = line[2:].strip()
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            current = {"heading": heading, "name": heading.split("｜", 1)[-1], "lines": []}
            card["sections"].append(current)
            continue
        if current is None:
            m = HEADER_FIELD.match(line.strip())
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key == "中文名":
                    card["zh"] = val
                elif key == "英文名":
                    card["en"] = val
                elif key == "別名":
                    card["aliases"] = [a.strip() for a in val.split("、") if a.strip()]
                elif key == "說明頁":
                    card["url"] = val
                elif key == "可用視圖":
                    card["views"] = val
            continue
        current["lines"].append(line)
    for sec in card["sections"]:
        sec["text"] = "\n".join(sec.pop("lines")).strip()
    card["items"] = [it for sec in card["sections"] if sec["name"] in ITEM_SECTIONS for it in _split_items(sec)]
    return card


ITEM_SECTIONS = ("常見問題", "錯誤訊息對照")


def _split_items(sec: dict) -> list:
    """常見問題 / 錯誤訊息對照 拆成一條一條：每條從行首的「- 」開始（「- 問：…」「- 「訊息」」），
    後面縮排的「答：」「原因：」「解法：」都算同一條。"""
    blocks, buf = [], []
    for line in sec["text"].splitlines():
        if line.startswith("- ") and buf:
            blocks.append("\n".join(buf).strip())
            buf = []
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    items = []
    for block in blocks:
        first = block.splitlines()[0] if block else ""
        if not first.startswith("- "):  # 段落開頭的說明文字，不是一條問答
            continue
        title = re.sub(r"^- (問：)?", "", first).strip()
        items.append({"section": sec["name"], "heading": sec["heading"], "title": title, "text": block,
                      "quoted": title.startswith("「")})
    return items


def _kb_files() -> list:
    return sorted(
        os.path.join(KB_DIR, f) for f in os.listdir(KB_DIR) if f.endswith(".md") and f not in EXCLUDE_FILES
    )


def _load_cards():
    global _cards, _error_index
    if _cards is not None:
        return _cards
    cards = {}
    for path in _kb_files():
        card = _parse_card(path)
        cards[card["api"]] = card
    # 常見問題裡用「」框起來的大多是按鈕名稱（「建立切割樓板」），不是錯誤訊息；
    # 只收錯誤訊息對照段落，常見問題裡的「」只有不是任何按鈕名稱時才收。
    button_names = {_normalize(n) for c in cards.values() for n in [c["zh"], c["en"]] + c["aliases"] if n}
    errors = []
    for api, card in cards.items():
        for item in card["items"]:
            for quoted in re.findall(r"「([^」]+)」", item["text"]):
                norm = _normalize(quoted, for_error=True)
                if len(norm) < 6:
                    continue
                if item["section"] == "常見問題" and _normalize(quoted) in button_names:
                    continue
                errors.append((norm, api, item))
    _cards, _error_index = cards, errors
    return cards


# ---------- 切段、匯入 ----------

def _split_long(heading_line: str, text: str) -> list:
    """超過 MAX_CHUNK_CHARS 的段落（主要是 _功能目錄 的大表格）依行切開，表格段每塊都補上表頭。"""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    lines = text.splitlines()
    table_header = [l for l in lines[:3] if l.startswith("|")][:2]
    pieces, buf = [], []
    for line in lines:
        if buf and len("\n".join(buf + [line])) > MAX_CHUNK_CHARS:
            pieces.append("\n".join(buf))
            buf = list(table_header) if line.startswith("|") and table_header else []
        buf.append(line)
    if buf:
        pieces.append("\n".join(buf))
    return pieces


def _chunks_for(card: dict) -> list:
    """每個 ## 段落一塊，前面加上卡片第一行標題，讓每段都知道自己屬於哪個按鈕。
    常見問題、錯誤訊息對照則是一條一塊：整段當一塊的話，同一個按鈕的不同問題
    （「欄杆扶手沒列出」vs「報告存在哪」）檢索到的都是同一大段，回答也就都一樣。"""
    chunks = []
    for sec in card["sections"]:
        if sec["name"] in ITEM_SECTIONS:
            pieces = [it["text"] for it in card["items"] if it["heading"] == sec["heading"]] or [sec["text"]]
        else:
            pieces = _split_long(sec["heading"], sec["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{card['api']}::{sec['heading']}::{i}",
                "text": f"# {card['title']}\n## {sec['heading']}\n{piece}",
                "meta": {"api": card["api"], "section": sec["name"], "notion_url": card["url"], "zh": card["zh"] or card["title"]},
            })
    return chunks


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _get_index():
    """LlamaIndex 的 VectorStoreIndex，底層存在 Chroma（跟 Belfast 的 nutrition_lookup 同一套寫法），
    向量模型共用 nutrition_lookup 已載入的 bge-m3。"""
    global _collection, _vector_store, _index
    if _index is None:
        os.makedirs(STORE_DIR, exist_ok=True)
        nutrition_lookup._load()
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        _vector_store = ChromaVectorStore(chroma_collection=_collection)
        _index = VectorStoreIndex.from_vector_store(_vector_store, embed_model=nutrition_lookup._embed_model)
    return _index


def _to_node(chunk: dict) -> TextNode:
    # 段落文字本身已經帶「# 按鈕標題 / ## 段落名稱」，metadata 不再併進向量和給 LLM 的文字，
    # 不然 LlamaIndex 預設會把 api、notion_url 這些欄位也塞進去算向量。
    keys = list(chunk["meta"].keys())
    return TextNode(
        id_=chunk["id"],
        text=chunk["text"],
        metadata=chunk["meta"],
        excluded_embed_metadata_keys=keys,
        excluded_llm_metadata_keys=keys,
        # 以卡片（檔名）當來源文件，更新時用 delete(ref_doc_id=檔名) 一次刪掉整張卡片的舊段落
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk["meta"]["api"])},
    )


def ensure_ingested() -> dict:
    """只重新匯入有變動的檔案（規格書第 9 節：刪掉該卡片的舊段落再重建）。"""
    cards = _load_cards()
    index = _get_index()
    state = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)

    current = {os.path.splitext(os.path.basename(p))[0]: _file_hash(p) for p in _kb_files()}
    changed = [api for api, h in current.items() if state.get(api) != h]
    removed = [api for api in state if api not in current]

    for api in changed + removed:
        _vector_store.delete(ref_doc_id=api)
    for api in changed:
        index.insert_nodes([_to_node(c) for c in _chunks_for(cards[api])])

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=1)
    return {"changed": len(changed), "removed": len(removed), "total_chunks": _collection.count()}


# ---------- 查詢流程 ----------

def _card_names(card: dict) -> list:
    names = [card["zh"], card["en"], card["en"].split(".")[-1]] + card["aliases"]
    return [n for n in names if n and len(_normalize(n)) >= 2]


def match_catalog(question: str) -> list:
    """回傳 [(api, 比對到的名稱)]。先試完全包含，沒有才試模糊比對；
    同一句話命中多個時，被別的命中「包含」的較短命中會被拿掉
    （問「建立Deck切割樓板」時，不要再跳出 RC 的「切割樓板」）。"""
    cards = _load_cards()
    qn = _normalize(question)
    hits = {}
    for api, card in cards.items():
        if api.startswith("_"):
            continue
        for name in _card_names(card):
            nn = _normalize(name)
            if nn and nn in qn and len(nn) > len(hits.get(api, "")):
                hits[api] = nn
    if not hits:
        # 模糊比對只容許「同長度、恰好錯一個字」的打錯字（單線轉粱）。用相似度比例的話，
        # 「干涉檢查的」跟「門干涉檢查」只是錯開一格也有 0.8，會亂對。
        for api, card in cards.items():
            if api.startswith("_"):
                continue
            for name in _card_names(card):
                nn = _normalize(name)
                if len(nn) < 4 or len(qn) < len(nn):
                    continue
                for i in range(len(qn) - len(nn) + 1):
                    window = qn[i:i + len(nn)]
                    if sum(a != b for a, b in zip(nn, window)) == 1 and len(nn) > len(hits.get(api, "")):
                        hits[api] = nn
                        break
    # 比對到的詞如果也是別的按鈕名稱的一部分（「切割樓板」⊂「建立Deck切割樓板」），
    # 使用者可能指的是任一個，一併列為候選、回問（規格書第 7 節範例）。
    for api, nn in list(hits.items()):
        if len(nn) < 3:
            continue
        for other, card in cards.items():
            if other in hits or other.startswith("_"):
                continue
            if any(nn in _normalize(n) for n in _card_names(card)):
                hits[other] = nn
    longest = {api: nn for api, nn in hits.items() if not any(nn != o and nn in o for o in hits.values())}
    return sorted(longest.items(), key=lambda kv: -len(kv[1]))


def match_errors(question: str) -> list:
    """回傳命中的 [(api, 那一條)]，同一句訊息可能出現在多張卡片。"""
    _load_cards()
    qn = _normalize(question, for_error=True)
    if len(qn) < 6:
        return []
    found = []
    for err, api, item in _error_index:
        if err in qn or (len(qn) >= 8 and qn in err):
            if not any(a == api and it is item for a, it in found):
                found.append((api, item))
    return found


def _item_hit(api: str, item: dict) -> dict:
    card = _load_cards()[api]
    return {"text": f"# {card['title']}\n## {item['heading']}\n{item['text']}", "api": api,
            "section": item["section"], "notion_url": card["url"], "zh": card["zh"] or card["title"]}


def _contacts() -> list:
    """從 _通用問題 的「找誰問」表格讀出 [(問題類型, 聯絡人)]。"""
    card = _load_cards().get(GENERAL, {})
    for sec in card.get("sections", []):
        if sec["name"].startswith("找誰問"):
            rows = []
            for line in sec["text"].splitlines():
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cols) == 2 and cols[0] not in ("問題類型", "---") and not set(cols[0]) <= {"-"}:
                    rows.append((cols[0], cols[1]))
            return rows
    return []


def _contacts_text() -> str:
    return "\n".join(f"- {kind}：{who}" for kind, who in _contacts())


def _out_of_scope(question: str):
    q = question.lower()
    for words, key in OUT_OF_SCOPE:
        if any(w in q for w in words):
            for kind, who in _contacts():
                if key.lower() in kind.lower():
                    return kind, who
    return None


def _node_hit(node, score=None) -> dict:
    hit = {"text": node.get_content(metadata_mode=MetadataMode.NONE), **node.metadata}
    if score is not None:
        hit["score"] = round(score, 4)
    return hit


def _retrieve(question: str, apis: list = None, k: int = TOP_K) -> list:
    """向量檢索（LlamaIndex retriever），apis 有給時用 metadata 篩選只搜這幾張卡片。"""
    filters = None
    if apis:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="api", value=apis, operator=FilterOperator.IN) if len(apis) > 1
            else MetadataFilter(key="api", value=apis[0], operator=FilterOperator.EQ)
        ])
    retriever = _get_index().as_retriever(similarity_top_k=k, filters=filters)
    return [_node_hit(n.node, n.score) for n in retriever.retrieve(question)]


def _sections_text(pairs: list) -> list:
    """直接取指定 (api, section) 的整段——錯誤訊息比對命中、通用問題指定段落時用。
    用 get_nodes 依條件取出，刻意不經過相似度排序（見 match_errors 的說明）。"""
    _get_index()
    hits = []
    for api, section in pairs:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="api", value=api), MetadataFilter(key="section", value=section)],
            condition=FilterCondition.AND,
        )
        hits += [_node_hit(n) for n in _vector_store.get_nodes(node_ids=None, filters=filters)]
    return hits


# 規則刻意放在參考資料「後面」：8B 模型對越靠近結尾的指示越聽話，規則放前面時
# 「不可以拼湊建議」被忽略（實測把「不支援斜板」跟下一行「Deck 樓板請改用…」
# 連起來，回答成「斜板請改用建立Deck切割樓板」）。
SYSTEM_PROMPT = """你是「CEC 建築 Revit API」的使用助手，回答公司同仁在 Line 上的問題。

【參考資料】
{context}

【找誰問】
{contacts}

【使用者問題】
{question}

回答前請遵守：
1. 只能根據【參考資料】回答，不可以自己編造功能、步驟、參數名稱或錯誤原因。
2. 只回答使用者問的事。參考資料的每一條都是獨立的，不可以把不同條拼湊成資料沒有寫的建議或替代做法。
   例如資料寫「4. 不支援斜板。5. Deck 樓板請改用「建立Deck切割樓板」」，被問到斜板時只能回答「不支援斜板」，
   第 5 條講的是 Deck 樓板、跟斜板無關，不可以推薦。
3. 參考資料沒有答案時，直接說「這個問題我沒有資料」，並依【找誰問】建議聯絡對象。有答案時不用附聯絡人。
4. 一律使用繁體中文，口氣簡潔、條列式，適合在手機上閱讀，長度大約手機一個畫面（20 行內）。
   只挑跟問題有關的內容：問「怎麼用」就講執行前要準備和操作步驟，不用把常見問題全部列出來。
5. 錯誤訊息、參數名稱、按鈕名稱要照參考資料原文寫，用「」標示。
6. 回答步驟時依參考資料的順序列出，不要省略「執行前要準備」的重點。
7. 不要寫網址。
8. 問題不是 CEC 建築 API 時（機電、土木、Autodesk 帳號、雲端、Revit 原生操作），不要回答內容，直接依【找誰問】轉介。

回答："""


def _llm(prompt: str) -> str:
    payload = {
        "model": ANSWER_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1500},
    }
    if "qwen3" in ANSWER_MODEL:
        payload["think"] = False
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = json.loads(resp.read().decode("utf-8")).get("response", "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _dedupe(hits: list) -> list:
    seen, out = set(), []
    for h in hits:
        if h["text"] not in seen:
            seen.add(h["text"])
            out.append(h)
    return out


def _general_hits(question: str) -> list:
    q = question.lower()
    sections = [sec for words, sec in GENERAL_SECTION_RULES if any(w in q for w in words)]
    return _sections_text([(GENERAL, s) for s in sections])


def _compose(question: str, hits: list, history: list = None) -> str:
    hits = _dedupe(hits)
    context = "\n\n".join(h["text"] for h in hits)
    prompt = SYSTEM_PROMPT.format(contacts=_contacts_text(), context=context, question=question)
    if history:
        # 追問常用「清單」「那個」之類的指代，附上前幾輪問答讓模型知道在講什麼
        turns = "\n\n".join(f"問：{q}\n答：{a[:400]}" for q, a in history[-HISTORY_TURNS:])
        prompt = prompt.replace(
            "【使用者問題】", f"【先前對話】（同一位同仁剛才問過的，供理解追問用）\n{turns}\n\n【使用者問題】", 1
        )
    answer = _to_traditional(_llm(prompt))
    # 網址一律由程式附上（模型抄網址容易抄錯），模型自己寫的網址行拿掉
    answer = "\n".join(l for l in answer.splitlines() if "http" not in l and "網址" not in l).strip()
    answer = re.sub(r"^回答[:：]\s*", "", answer)
    urls = []
    for h in hits:
        if h.get("notion_url") and h["notion_url"] not in urls:
            urls.append(h["notion_url"])
    if urls:
        answer += "\n\n📖 說明頁（有圖文與影片）：\n" + "\n".join(urls[:2])
    return answer


def _log(entry: dict):
    os.makedirs(STORE_DIR, exist_ok=True)
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def candidate_choices(question: str, n: int = 3) -> list:
    """比對不到按鈕時，用向量檢索找幾個可能的按鈕給使用者選。"""
    seen = []
    for h in _retrieve(question, k=12):
        if not h["api"].startswith("_") and h["api"] not in [a for a, _ in seen]:
            seen.append((h["api"], h["zh"]))
        if len(seen) >= n:
            break
    return seen


# ---------- 對話狀態：記住目前在聊哪個按鈕 ----------
#
# 2026/10/5 實測：同仁選了「建立切割樓板」之後接著追問「我已經建立好柱樑了，但是切割失敗」
# 「執行後的清單是什麼」，這些追問沒有按鈕名稱，每句都被當成全新的問題——一句跑去全庫
# 檢索混進 Deck 版的資料，一句直接回問「是哪個功能」。改成記住目前的話題（按鈕），
# 沒提到其他按鈕的追問就延續同一個話題；提到別的按鈕或錯誤訊息就換話題；閒置太久就忘掉。

SESSION_TTL_SECONDS = 15 * 60
HISTORY_TURNS = 2
RESET_WORDS = ["新問題", "換個問題", "換問題", "重新開始", "問別的"]

_sessions = {}  # user_id -> {"api": str|None, "history": [(問, 答)], "time": datetime}


def _session(user_id: str) -> dict:
    s = _sessions.get(user_id)
    if s and (datetime.now() - s["time"]).total_seconds() > SESSION_TTL_SECONDS:
        s = None
    if s is None:
        s = {"api": None, "history": [], "time": datetime.now()}
        _sessions[user_id] = s
    return s


def reset_session(user_id: str):
    _sessions.pop(user_id, None)


# ---------- 先釐清狀況，再回答 ----------
#
# 2026/10/6 實測：「救命樓梯干涉壞掉了」→ 選了按鈕之後，直接回一份用途＋操作步驟的
# 制式說明。「壞掉」沒說是什麼狀況，檢索只能抓到最泛的段落；同一個按鈕不管是
# 「報告找不到」還是「扶手沒列出」都會拿到同一份回答。改成：問題含「壞掉／不能用」
# 這類籠統說法、又對不到這張卡片任何一條常見問題或錯誤訊息時，先回問發生什麼狀況，
# 選項直接用卡片自己的常見問題（不是我們編的分類）。
#
# 門檻 0.80 是實測校準的：具體描述對到正確那條的相似度 0.77~0.98（多數 ≥0.89），
# 籠統說法最高 0.62；唯一例外「切割樓板壞掉了」0.773（對到不相干的「高度跑掉」），
# 所以規則是「含籠統詞」且「最高相似度 < 0.80」兩個條件都成立才回問。

VAGUE_WORDS = ["壞", "不能用", "不能跑", "不行", "有問題", "出問題", "怪", "失敗", "沒反應", "出錯",
               "跑不出", "跑不動", "沒用", "異常", "不正常", "當掉", "閃退", "卡住", "不對"]
USAGE_WORDS = ["怎麼用", "如何", "步驟", "教學", "要準備", "怎麼操作", "用法", "做什麼", "是什麼", "在哪"]
ITEM_MATCH_THRESHOLD = 0.80
MAX_CLARIFY_OPTIONS = 11  # LINE quick reply 最多 13 個，留 2 個給「跳出錯誤訊息」「其他狀況」

_item_vectors = {}  # api -> 正規化後的常見問題/錯誤訊息標題向量


def _best_item_score(api: str, question: str) -> float:
    items = _load_cards()[api]["items"]
    if not items:
        return 0.0
    if api not in _item_vectors:
        m = np.array(nutrition_lookup._embed_model.get_text_embedding_batch([it["title"] for it in items]))
        _item_vectors[api] = m / np.linalg.norm(m, axis=1, keepdims=True)
    q = np.array(nutrition_lookup._embed_model.get_query_embedding(question))
    return float((_item_vectors[api] @ (q / np.linalg.norm(q))).max())


def _needs_clarify(api: str, question: str) -> bool:
    q = question.lower()
    if not any(w in q for w in VAGUE_WORDS) or any(w in q for w in USAGE_WORDS):
        return False
    if any(k.lower() in q for k in GENERAL_KEYWORDS) or match_errors(question):
        return False  # 「按不了／灰色」「授權」這類有通用答案，貼了錯誤訊息的也已經很具體
    return _best_item_score(api, question) < ITEM_MATCH_THRESHOLD


def _clarify_options(api: str) -> list:
    """回問的選項：常見問題全收；錯誤訊息對照只收「不是訊息原文」的那幾條（像「跳出 Revit 警告…」
    「報告沒有…」這種現象描述）——有訊息原文的請使用者直接貼上，比列一長串好選。"""
    items = _load_cards()[api]["items"]
    faq = [it for it in items if it["section"] == "常見問題"]
    symptoms = [it for it in items if it["section"] == "錯誤訊息對照" and not it["quoted"]]
    return (faq + symptoms)[:MAX_CLARIFY_OPTIONS]


def answer(question: str, user_id: str = "", forced_api: str = None, search_all: bool = False,
           item_index: int = None) -> dict:
    """回傳 {"type": "answer"|"ask"|"clarify"|"refer"|"reset", "text", "route",
    "choices": [(api, 中文名)]（ask 用）, "options": [(編號, 狀況)]（clarify 用）}。
    item_index：使用者在「發生什麼狀況」的回問裡點了第幾個選項。"""
    cards = _load_cards()
    q_lower = question.lower()
    general = any(k.lower() in q_lower for k in GENERAL_KEYWORDS)
    session = _session(user_id)
    session["time"] = datetime.now()

    def switch_topic(api):
        """換了話題就先清掉舊話題的問答紀錄，再組提示詞——不然新問題的回答會被上一個
        按鈕的對話干擾（例：剛聊完切割樓板就問單線轉樑，模型還看得到切割樓板的問答）。"""
        if api != session["api"]:
            session["history"] = []
        session["api"] = api

    def done(result: dict) -> dict:
        if result["type"] == "answer":
            session["history"] = (session["history"] + [(question, result["text"])])[-HISTORY_TURNS:]
            session["clarify"] = None
        _log({"user": user_id, "question": question, "route": result.get("route"), "type": result["type"],
              "topic": session["api"], "choices": result.get("choices"), "answer": result.get("text", "")[:2000]})
        return result

    def compose(hits: list) -> str:
        return _compose(question, hits, session["history"])

    def ask_symptom(api: str) -> dict:
        options = _clarify_options(api)
        session["clarify"] = {"api": api, "question": question, "options": options}
        lines = [f"了解，是「{cards[api]['zh']}」出了狀況。想先確認發生什麼事，才能給對的解法：", "",
                 "・有跳出錯誤訊息 → 直接把訊息文字複製貼上來"]
        if options:
            lines.append("・是下面其中一種 → 點下方對應的按鈕")
            lines += [f"  {i + 1}. {it['title']}" for i, it in enumerate(options)]
        lines.append("・都不是 → 描述一下做了哪一步、畫面出現什麼")
        return done({"type": "clarify", "route": f"釐清狀況:{api}", "text": "\n".join(lines),
                     "options": [(i, it["title"]) for i, it in enumerate(options)]})

    def answer_about(api: str, route: str, query: str = None, note: str = "") -> dict:
        """回答某一個按鈕的問題；太籠統（「壞掉了」）就先回問狀況。同一個按鈕問過一次就不再問，
        使用者接著描述的狀況就算還是講得不清楚，也直接用他的描述去檢索。"""
        switch_topic(api)
        already_asked = (session.get("clarify") or {}).get("api") == api
        if not already_asked and _needs_clarify(api, question):
            return ask_symptom(api)
        hits = _retrieve(query or question, apis=[api, GENERAL]) + _general_hits(question)
        return done({"type": "answer", "text": compose(hits) + note, "route": route})

    if any(w in question for w in RESET_WORDS) and len(question) <= 10:
        reset_session(user_id)
        return {"type": "reset", "route": "重設話題", "text": "好的，請問新的問題是？可以直接講按鈕名稱，或貼上錯誤訊息。"}

    if item_index is not None:
        c = session.get("clarify")
        if not c or not 0 <= item_index < len(c["options"]):
            return {"type": "answer", "route": "釐清選項過期", "text": "剛才的選項已經過期了，麻煩再描述一次遇到的狀況。"}
        api, item = c["api"], c["options"][item_index]
        switch_topic(api)
        # 只給選到的那一條：回答就只會針對這個狀況，不會又附上整套操作步驟
        question = f"{c['question']}（狀況：{item['title']}）"
        return done({"type": "answer", "text": compose([_item_hit(api, item)]), "route": f"釐清後:{api}#{item['title']}"})

    if forced_api:
        return answer_about(forced_api, f"使用者選擇:{forced_api}")
    if search_all:
        switch_topic(None)
        return done({"type": "answer", "text": compose(_retrieve(question)), "route": "全庫檢索"})

    catalog_hits = match_catalog(question)

    if not catalog_hits:
        scope = _out_of_scope(question)
        if scope:
            kind, who = scope
            return done({"type": "refer", "route": "非本庫範圍",
                         "text": f"這個問題不在 CEC 建築 API 的範圍（{kind}），請直接聯絡：{who}。"})

    if len(catalog_hits) == 1:
        api = catalog_hits[0][0]
        return answer_about(api, f"目錄比對:{api}")

    if len(catalog_hits) > 1:
        # 正在聊的按鈕剛好是候選之一（例：選過 RC 版後又說「切割樓板…」），就不再回問
        if session["api"] in [api for api, _ in catalog_hits]:
            api = session["api"]
            return answer_about(api, f"延續話題:{api}", note=_topic_note(api))
        choices = [(api, cards[api]["zh"]) for api, _ in catalog_hits]
        return done({"type": "ask", "route": "目錄比對多個", "choices": choices,
                     "text": "找到好幾個相關的功能，請問是哪一個？"})

    errors = match_errors(question)
    if errors:
        if session["api"] in {a for a, _ in errors}:
            errors = [(a, sec) for a, sec in errors if a == session["api"]]
        apis = sorted({a for a, _ in errors})
        switch_topic(apis[0] if len(apis) == 1 else None)
        hits = [_item_hit(a, it) for a, it in errors] + _retrieve(question, apis=[GENERAL], k=1)
        return done({"type": "answer", "text": compose(hits), "route": "錯誤訊息比對:" + ",".join(apis)})

    if session["api"] and not (any(k in question for k in FEATURE_SEARCH_KEYWORDS) and ("功能" in question or "按鈕" in question)):
        # 沒提到任何按鈕、也不是在找新功能 → 當成正在聊的按鈕的追問。
        # 檢索時把前一個問題也帶上，「清單是什麼」這種短追問才查得到對的段落。
        api = session["api"]
        prev = session["history"][-1][0] if session["history"] else ""
        return answer_about(api, f"延續話題:{api}", query=f"{prev} {question}", note=_topic_note(api))

    if general:
        hits = _general_hits(question) + _retrieve(question, apis=[GENERAL], k=2)
        return done({"type": "answer", "text": compose(hits), "route": "通用問題"})

    if any(k in question for k in FEATURE_SEARCH_KEYWORDS):
        hits = _retrieve(question, apis=[CATALOG], k=2) + _retrieve(question, k=3)
        return done({"type": "answer", "text": compose(hits), "route": "功能目錄"})

    if any(w in q_lower for w in ERROR_HINT_WORDS) and len(question) >= 15:
        return done({"type": "answer", "text": compose(_retrieve(question)), "route": "錯誤訊息未命中→全庫檢索"})

    return done({"type": "ask", "route": "比對不到→回問", "choices": candidate_choices(question),
                 "text": "請問是哪個功能的問題呢？下面是幾個可能的功能，或直接告訴我按鈕名稱。"})


def _topic_note(api: str) -> str:
    zh = _load_cards()[api]["zh"]
    return f"\n\n（延續「{zh}」的問題。要問別的功能，直接講按鈕名稱；或說「新問題」重新開始）"


if __name__ == "__main__":
    print(ensure_ingested())
