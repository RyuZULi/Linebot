"""
補充知識庫：台大膳食協調委員會的自助餐/盤菜熱量表，用來補 TFDA 原始
食材資料庫查不到的複合菜餚（糖醋排骨、炸豆腐、麻婆豆腐…）。

刻意跟 nutrition_db 分開成獨立 collection，因為這份資料的可信度跟
TFDA 官方送檢數據不是同一個等級——來源頁面自己就寫「僅供參考，實際
以現場供應為主」，沒有標示更新日期或檢測方法。lookup() 只有在 TFDA
完全查無品項時才會退而求其次查這裡，而且信心度會標示為比 TFDA 的
「精確對應」更低的等級，不能混為一談。
"""

import csv

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CSV_PATH = r"D:\CalorieCalculation\data\composite_dishes\ntu_buffet_kcal.csv"
CHROMA_PATH = r"D:\CalorieCalculation\data\composite_dishes\chroma_store"
COLLECTION_NAME = "composite_dishes"

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def build():
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    documents = [
        Document(
            text=row["food_item"],
            metadata={
                "category": row["category"],
                "food_item": row["food_item"],
                "kcal_per_100g": row["kcal_per_100g"],
                "source": row["source"],
            },
        )
        for row in rows
    ]
    print(f"Loaded {len(documents)} composite dish documents")

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context, embed_model=EMBED_MODEL
    )
    print(f"Indexed into Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    build()
