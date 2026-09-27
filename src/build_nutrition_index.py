"""
C1 前置作業：把 nutrition_db 建成 Chroma 向量索引。

檢索文字用「品項名稱 + 俗名」，讓「玉米」這種口語詞能透過俗名連到
「甜玉米」「雙色水果玉米」等 TFDA 正式品項名稱。
"""

import csv
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CSV_PATH = r"D:\CalorieCalculation\data\nutrition_db\tfda_nutrition_db.csv"
CHROMA_PATH = r"D:\CalorieCalculation\data\nutrition_db\chroma_store"
COLLECTION_NAME = "nutrition_db"

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def load_documents():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    documents = []
    for row in rows:
        aliases = (row.get("俗名") or "").strip()
        text = row["品項名稱"]
        if aliases:
            text += "，別名：" + aliases
        documents.append(
            Document(
                text=text,
                metadata={
                    "食品分類": row["食品分類"],
                    "品項名稱": row["品項名稱"],
                    "俗名": aliases,
                    "整合編號": row["整合編號"],
                    "熱量_kcal_per_100g": row["熱量_kcal_per_100g"],
                    "蛋白質_g": row["蛋白質_g"],
                    "脂肪_g": row["脂肪_g"],
                    "碳水化合物_g": row["碳水化合物_g"],
                },
            )
        )
    return documents


def build():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
    )
    print(f"Indexed into Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    build()
