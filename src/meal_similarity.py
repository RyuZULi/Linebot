"""
A3：整餐視覺校準錨點。

用 CLIP 圖片 embedding，把 LINE 使用者傳來的餐點照片跟 Nutrition5k 的
俯視照片做視覺相似度比對，取最相近的參考餐點的「已知真實總熱量／總重量」
當校準錨點——錨點是查表得來的真實數字，不是模型看照片自己猜的。

注意：Nutrition5k 是美式自助餐廳的菜色，跟台灣便當長得不一樣，這個
校準錨點的準確度天生受限於資料集跟目標場景的視覺落差，只能當一個
「合理區間」的參考，不是精確答案——這件事要在回覆文字裡對使用者講清楚，
不能包裝成看起來很精確的數字。
"""

import csv
import json
import os

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

N5K_DIR = r"D:\CalorieCalculation\data\nutrition5k"
IMG_DIR = os.path.join(N5K_DIR, "images")
INDEX_PATH = os.path.join(N5K_DIR, "clip_index.npz")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None


def _load_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _model, _processor


def embed_image(image_path: str) -> np.ndarray:
    from PIL import Image

    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    # 這個 transformers 版本的 get_image_features() 回傳一個
    # BaseModelOutputWithPooling，真正投影過的 512 維圖片向量在
    # .pooler_output（不是直接回傳扁平張量，跟舊版 API 不一樣）。
    vec = outputs.pooler_output[0].detach().numpy()
    return vec / np.linalg.norm(vec)


def load_dish_metadata() -> dict:
    """回傳 {dish_id: {"total_calories": float, "total_mass": float}}。"""
    meta = {}
    for fname in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]:
        path = os.path.join(N5K_DIR, fname)
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                dish_id, total_calories, total_mass = row[0], row[1], row[2]
                meta[dish_id] = {
                    "total_calories": float(total_calories),
                    "total_mass": float(total_mass),
                }
    return meta


def build_index():
    metadata = load_dish_metadata()
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    print(f"found {len(files)} downloaded images")

    vectors = []
    dish_ids = []
    for i, fname in enumerate(files):
        dish_id = fname[:-4]
        if dish_id not in metadata:
            continue
        try:
            vec = embed_image(os.path.join(IMG_DIR, fname))
        except Exception as e:
            print(f"skip {dish_id}: {e}")
            continue
        vectors.append(vec)
        dish_ids.append(dish_id)
        if (i + 1) % 200 == 0:
            print(f"embedded {i + 1}/{len(files)}")

    matrix = np.stack(vectors)
    np.savez(
        INDEX_PATH,
        matrix=matrix,
        dish_ids=np.array(dish_ids),
        calories=np.array([metadata[d]["total_calories"] for d in dish_ids]),
        masses=np.array([metadata[d]["total_mass"] for d in dish_ids]),
    )
    print(f"saved index with {len(dish_ids)} dishes to {INDEX_PATH}")


_index_cache = None


def _load_index():
    global _index_cache
    if _index_cache is None:
        data = np.load(INDEX_PATH, allow_pickle=True)
        _index_cache = {
            "matrix": data["matrix"],
            "dish_ids": data["dish_ids"],
            "calories": data["calories"],
            "masses": data["masses"],
        }
    return _index_cache


def find_similar_meals(image_path: str, top_k: int = 5) -> list:
    """回傳最相近的 top_k 個 Nutrition5k 參考餐點，含相似度分數與已知總熱量/總重量。"""
    idx = _load_index()
    query_vec = embed_image(image_path)
    scores = idx["matrix"] @ query_vec  # 都已正規化，內積 = cosine similarity
    top_idx = np.argsort(-scores)[:top_k]
    return [
        {
            "dish_id": str(idx["dish_ids"][i]),
            "similarity": round(float(scores[i]), 4),
            "total_calories": float(idx["calories"][i]),
            "total_mass": float(idx["masses"][i]),
        }
        for i in top_idx
    ]


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "build":
        build_index()
    else:
        result = find_similar_meals(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
