"""
C1：食物名稱檢索與比對邏輯。

輸入：B2 辨識出的食物名稱字串（可能不準確，見 B2_REPORT.md）。
輸出：對應的 nutrition_db 品項與三級信心標示：
  - exact   精確對應：品項名稱或俗名完全相符
  - similar 相近品項：沒有完全相符，但語意檢索找到夠接近的品項
  - none    查無品項：找不到任何合理對應，誠實回覆「無法確定」

設計原則（呼應 TODO.md「避免幻覺」）：查無品項時絕不硬湊一個最接近的
結果充當答案——語意檢索分數再高，只要低於 SIMILAR_THRESHOLD 就一率
視為查無品項。
"""

import csv

import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CSV_PATH = r"D:\CalorieCalculation\data\nutrition_db\tfda_nutrition_db.csv"
CHROMA_PATH = r"D:\CalorieCalculation\data\nutrition_db\chroma_store"
COLLECTION_NAME = "nutrition_db"

# 補充知識庫：台大膳食協調委員會的自助餐/盤菜熱量表，只有在 TFDA
# 完全查無品項時才會退而求其次查這裡（見 build_composite_index.py
# 開頭的說明），信心度標成 exact_secondary/similar_secondary，跟
# TFDA 的 exact/similar 分開，因為這份資料的可信度天生比較低
# （來源頁面自己就寫「僅供參考」，沒有標示檢測方法或更新日期）。
COMPOSITE_CSV_PATH = r"D:\CalorieCalculation\data\composite_dishes\ntu_buffet_kcal.csv"
COMPOSITE_CHROMA_PATH = r"D:\CalorieCalculation\data\composite_dishes\chroma_store"
COMPOSITE_COLLECTION_NAME = "composite_dishes"
# 這份資料另外校準過，不是沿用 TFDA 的 0.70：測了幾個「應該查無品項」
# 的雜訊詞(腳踏車/亂打字/珍珠奶茶等)，分數上限約 0.59；而「炒青菜」這種
# 常見但沒有完全對應品項的詞，實際分數是 0.6965，中間有約 0.10 的安全
# 間隔，比 TFDA 當初"土豆"跟"地瓜葉"只差 0.013 的情況健康很多。加上
# 這份資料裡的蔬菜熱量本來就集中在 57~89 kcal/100g 窄區間，就算比對到
# 「不是最精確」的那道菜，熱量估計也不會差太多，門檻可以不用像 TFDA
# 那麼保守。
COMPOSITE_SIMILAR_THRESHOLD = 0.65

# 使用者自己回報的熱量資料（2026/9/27 新增，見霽倫閣下的需求：使用者
# 說「這個是XX大卡」時，除了存進個人紀錄，也要讓之後的估算查得到）。
# 這是一個會「動態成長」的知識庫，不是像 A1/A4 那樣一次建好的靜態檔案，
# 用 add_user_food() 隨時插入新資料。信心度是三者中最高的——不是查表
# 查來的，是使用者本人親口講的（通常是看包裝標示或自己查過的），比
# 語意檢索猜出來的相近品項可信，所以查詢順序排在 TFDA 之後、
# composite_dishes(公用參考表)之前。
USER_FOODS_CHROMA_PATH = r"D:\CalorieCalculation\data\user_provided\chroma_store"
USER_FOODS_COLLECTION_NAME = "user_provided_foods"

# 語意相似度門檻，依 C1_REPORT.md 的混淆測試集校準，不是憑感覺決定的數字。
# 校準時發現 bge-m3 對這些短食材詞常被「字面相似」誤導而非真的語意相近
# （例："小黃瓜片"最高分是"小黃魚"、"炸雞腿"最高分是"炸雞粉"、"腳踏車"
# 這種完全無關的詞也能生出 0.59 分），門檻設太低會把錯誤類別的食物包裝成
# 「相近品項」端出去，比直接說「查無品項」更危險，所以刻意設得保守。
SIMILAR_THRESHOLD = 0.70

_embed_model = None
_index = None
_alias_lookup = None  # 品項名稱/俗名 -> row dict，用於精確比對
_composite_index = None
_composite_alias_lookup = None  # food_item -> row dict
_user_index = None
_user_alias_lookup = None  # food_name -> {"fixed_kcal", "source"}

# 已知會造成誤判的模糊俗名，不能直接當「精確對應」使用。
# 例："土豆"在 TFDA 資料裡被列為花生的俗名，但這是中國用語習慣
# （台灣語境的"土豆"多半指馬鈴薯或單純的花生口語，容易跟真正的馬鈴薯
# 混淆）。實測發現辨識結果"土豆"若直接精確比對，會被導向"黑金剛花生"
# (553 kcal/100g)，熱量比實際的馬鈴薯(77 kcal/100g)高將近 7 倍，
# 是會嚴重誤導使用者的錯誤。發現一個就加進這裡，而非假設俗名都可信。
AMBIGUOUS_ALIASES = {"土豆"}


def _normalize(s: str) -> str:
    return (s or "").strip()


def _load():
    global _embed_model, _index, _alias_lookup, _composite_index, _composite_alias_lookup
    global _user_index, _user_alias_lookup
    if _index is not None:
        return

    _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)
    _index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_embed_model)

    alias_lookup = {}
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys = [row["品項名稱"]] + [a.strip() for a in (row.get("俗名") or "").split(",") if a.strip()]
            keys = [k for k in keys if k not in AMBIGUOUS_ALIASES]
            for k in keys:
                # 同一個別名可能對到多筆(不同取樣批次)，全部留著，之後取熱量最接近中位數的那筆
                alias_lookup.setdefault(k, []).append(row)
    _alias_lookup = alias_lookup

    composite_db = chromadb.PersistentClient(path=COMPOSITE_CHROMA_PATH)
    composite_collection = composite_db.get_or_create_collection(
        COMPOSITE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    composite_vector_store = ChromaVectorStore(chroma_collection=composite_collection)
    _composite_index = VectorStoreIndex.from_vector_store(composite_vector_store, embed_model=_embed_model)

    composite_alias_lookup = {}
    with open(COMPOSITE_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            composite_alias_lookup.setdefault(row["food_item"], []).append(row)
    _composite_alias_lookup = composite_alias_lookup

    user_db = chromadb.PersistentClient(path=USER_FOODS_CHROMA_PATH)
    user_collection = user_db.get_or_create_collection(
        USER_FOODS_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    user_vector_store = ChromaVectorStore(chroma_collection=user_collection)
    _user_index = VectorStoreIndex.from_vector_store(user_vector_store, embed_model=_embed_model)

    # 這個知識庫是動態成長的，沒有背後的 CSV 可以重讀，重啟服務後要從
    # Chroma 自己存的 metadata 重建 alias 表，不然新增過的資料下次查
    # 精確比對會找不到（只剩語意檢索，退化成跟其他品項一樣的路徑）。
    user_alias_lookup = {}
    existing = user_collection.get(include=["metadatas"])
    for meta in existing.get("metadatas", []) or []:
        user_alias_lookup[meta["food_name"]] = meta
    _user_alias_lookup = user_alias_lookup


def _pick_representative(rows: list[dict]) -> dict:
    """多筆同名資料時，優先取「平均值」版本，否則取熱量中位數那筆。"""
    for r in rows:
        if "平均值" in r["整合編號"] or "平均值" in (r.get("資料來源") or ""):
            return r
    sorted_rows = sorted(rows, key=lambda r: float(r["熱量_kcal_per_100g"]))
    return sorted_rows[len(sorted_rows) // 2]


def _normalize_composite_row(row: dict) -> dict:
    """把補充知識庫的欄位名稱轉成跟 nutrition_db 一樣的 schema
    （品項名稱／熱量_kcal_per_100g），downstream(C2/C3)才不用另外判斷
    來源、寫兩套邏輯。"""
    return {
        "品項名稱": row["food_item"],
        "熱量_kcal_per_100g": row["kcal_per_100g"],
        "食品分類": None,  # 這份資料沒有跟 TFDA 一樣的分類系統，C2 的分類規則對不上就會誠實回報不確定
        "資料來源": row.get("source", ""),
    }


# 乾貨、粉類的熱量密度跟新鮮/煮熟的同種食物差好幾倍（花椰菜乾 291 vs 新鮮
# 花椰菜約 30 kcal/100g），語意檢索分不出這個差別。2026/10/5 實測「炒花椰菜」
# 被比對成「花椰菜乾」，一格配菜算成 291 大卡。查詢名稱本身沒提到這些字時，
# 就跳過帶有這些字的候選（「炒麵」也就不會比對成「麵粉」）。
_PRESERVED_FORM_MARKERS = ("乾", "粉")


def _form_compatible(query: str, candidate_name: str) -> bool:
    return not any(m in candidate_name and m not in query for m in _PRESERVED_FORM_MARKERS)


def _first_acceptable(candidates: list, query: str, threshold: float, name_key: str):
    for c in candidates:
        if c["score"] < threshold:
            return None  # 依分數排序，後面只會更低
        if _form_compatible(query, c.get(name_key) or ""):
            return c
    return None


def _lookup_composite(name: str, top_k: int, allow_similar: bool = True) -> dict:
    """TFDA 查無品項時的第二層查詢，回傳的 confidence 一律帶 _secondary
    後綴，讓下游知道這不是官方送檢數據。查不到就回傳 None。"""
    if name in _composite_alias_lookup:
        # 這份資料的欄位跟 TFDA 不一樣，_pick_representative() 是為 TFDA
        # 設計的(依賴"整合編號"/"資料來源"欄位)，不能直接沿用；同名品項
        # 目前資料裡沒有真的重複，取第一筆即可。
        row = _normalize_composite_row(_composite_alias_lookup[name][0])
        return {"confidence": "exact_secondary", "matched": row, "candidates": []}
    if not allow_similar:
        return None

    retriever = _composite_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(name)
    candidates = [{"score": round(n.score, 4), **n.node.metadata} for n in nodes]
    best = _first_acceptable(candidates, name, COMPOSITE_SIMILAR_THRESHOLD, "food_item")
    if best is not None:
        return {
            "confidence": "similar_secondary",
            "matched": _normalize_composite_row(best),
            "candidates": candidates,
        }
    return None


def add_user_food(food_name: str, fixed_kcal: float, source: str) -> None:
    """把使用者回報的「這個是XX大卡」存進可查詢的知識庫。

    fixed_kcal 是「這份/這個品項」的總熱量，不是每 100 公克密度——使用者
    講的通常是包裝標示或整份餐點的總量，跟 TFDA 那種密度資料是不同的
    量綱，所以用 fixed_kcal 這個獨立欄位，C3 算熱量時看到這個欄位就直接
    採用，不會拿去乘份量公克數（見 calorie_estimator.py 的特殊處理）。
    """
    _load()
    metadata = {"food_name": food_name, "fixed_kcal": fixed_kcal, "source": source}
    _user_index.insert(Document(text=food_name, metadata=metadata))
    _user_alias_lookup[food_name] = metadata


def _lookup_user_food(name: str, top_k: int, allow_similar: bool = True) -> dict:
    if name in _user_alias_lookup:
        return {"confidence": "exact_user", "matched": _user_alias_lookup[name], "candidates": []}

    if not allow_similar or not _user_alias_lookup:
        return None  # 索引是空的，語意檢索也不用跑了

    retriever = _user_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(name)
    candidates = [{"score": round(n.score, 4), **n.node.metadata} for n in nodes]
    # 沿用 TFDA 校準出的保守門檻——使用者自建資料筆數少，還沒辦法像
    # composite_dishes 那樣另外做雜訊校準，寧可保守一點。
    best = _first_acceptable(candidates, name, SIMILAR_THRESHOLD, "food_name")
    if best is not None:
        return {"confidence": "similar_user", "matched": best, "candidates": candidates}
    return None


def lookup(food_name: str, top_k: int = 3) -> dict:
    """查詢單一食物名稱，回傳 {confidence, matched, candidates}。"""
    _load()
    name = _normalize(food_name)

    # 1) 精確比對(品項名稱或俗名完全相符)
    #
    # 曾嘗試用語意分數做通用版的「合理性檢查」取代逐案例黑名單
    # （構想：精確比對到的品項如果語意分數太低，代表俗名可能不可靠）。
    # 實測發現行不通——已知有問題的"土豆"語意分數是 0.6477，但另一個
    # 完全合法的精確比對"地瓜葉"只有 0.6607，兩者只差 0.013，找不到
    # 一個安全的門檻能同時抓到壞案例又不誤傷好案例。這不是隨便調參數
    # 就能解決的，是這批短食材詞的語意分數本身區分度不夠。放棄這個
    # 通用化方向，繼續用 AMBIGUOUS_ALIASES 逐案例處理——這裡誠實記錄
    # 失敗的嘗試，避免以後又走一次同樣的死路。
    if name in _alias_lookup:
        row = _pick_representative(_alias_lookup[name])
        return {
            "query": name,
            "confidence": "exact",
            "matched": row,
            "candidates": [],
        }

    # 2) 語意檢索
    retriever = _index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(name)
    candidates = [
        {"score": round(n.score, 4), **n.node.metadata} for n in nodes
    ]

    best = _first_acceptable(candidates, name, SIMILAR_THRESHOLD, "品項名稱")
    if best is not None:
        return {
            "query": name,
            "confidence": "similar",
            "matched": best,
            "candidates": candidates,
        }

    # 3) TFDA 查無品項，接著查使用者自己回報過的資料跟補充知識庫（台大自助餐
    # 熱量表）。兩邊都先只看「菜名完全相同」，都沒有才看「相近菜名」：
    # 2026/10/5 實測「炸豬排」被相近比對到使用者回報的「炸豬排便當」（整個
    # 便當的熱量），排在台大表裡完全相同的「炸豬排」前面，等於把白飯、配菜
    # 又算一次。使用者資料本身可信度仍排在台大表前面（本人講的），只是
    # 「完全相同」一律優先於「相近」。
    for allow_similar in (False, True):
        user_result = _lookup_user_food(name, top_k, allow_similar)
        if user_result is not None:
            return {"query": name, "candidates": candidates, **user_result}
        composite_result = _lookup_composite(name, top_k, allow_similar)
        if composite_result is not None:
            return {"query": name, "candidates": candidates, **composite_result}

    # 5) 三個知識庫都查無品項
    return {
        "query": name,
        "confidence": "none",
        "matched": None,
        "candidates": candidates,
    }


if __name__ == "__main__":
    import json
    import sys

    result = lookup(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
