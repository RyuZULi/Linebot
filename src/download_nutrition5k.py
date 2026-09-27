"""
A3 前置作業：從 Nutrition5k 的公開 GCS bucket 下載俯視 RGB 照片跟 metadata。

完整資料集(含影片、深度資料)有 181GB，我們只需要：
  - imagery/realsense_overhead/dish_*/rgb.png（俯視 RGB 照片）
  - metadata/dish_metadata_cafe{1,2}.csv（每道菜的總重量/總熱量）

bucket 是公開的，直接用 HTTPS + JSON API 列表，不需要安裝 gsutil/gcloud SDK。
"""

import concurrent.futures
import csv
import json
import os
import urllib.request

BUCKET = "nutrition5k_dataset"
PREFIX = "nutrition5k_dataset/imagery/realsense_overhead/"
API_LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OUT_DIR = r"D:\CalorieCalculation\data\nutrition5k"
IMG_DIR = os.path.join(OUT_DIR, "images")
METADATA_URLS = [
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv",
]

os.makedirs(IMG_DIR, exist_ok=True)


def download_metadata():
    for url in METADATA_URLS:
        fname = os.path.join(OUT_DIR, os.path.basename(url))
        urllib.request.urlretrieve(url, fname)
        print(f"downloaded {fname}")


def list_rgb_objects():
    """列出所有 dish_*/rgb.png 的物件，回傳 [(dish_id, media_url), ...]。"""
    items = []
    page_token = None
    while True:
        url = API_LIST_URL.format(bucket=BUCKET) + f"?prefix={PREFIX}&maxResults=1000"
        if page_token:
            url += f"&pageToken={page_token}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        for obj in body.get("items", []):
            name = obj["name"]
            if name.endswith("/rgb.png"):
                dish_id = name.split("/")[-2]
                items.append((dish_id, obj["mediaLink"]))
        page_token = body.get("nextPageToken")
        print(f"listed {len(items)} rgb.png so far...")
        if not page_token:
            break
    return items


def download_one(dish_id, url):
    out_path = os.path.join(IMG_DIR, f"{dish_id}.png")
    if os.path.exists(out_path):
        return dish_id, True
    try:
        urllib.request.urlretrieve(url, out_path)
        return dish_id, True
    except Exception as e:
        return dish_id, str(e)


def download_all_images(items, max_workers=16):
    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_one, dish_id, url) for dish_id, url in items]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            dish_id, result = fut.result()
            if result is True:
                ok += 1
            else:
                fail += 1
                print(f"FAILED {dish_id}: {result}")
            if (i + 1) % 200 == 0:
                print(f"progress: {i + 1}/{len(items)} (ok={ok}, fail={fail})")
    print(f"done. ok={ok} fail={fail}")


if __name__ == "__main__":
    print("downloading metadata...")
    download_metadata()
    print("listing rgb.png objects...")
    items = list_rgb_objects()
    print(f"total dishes with overhead rgb photo: {len(items)}")
    with open(os.path.join(OUT_DIR, "dish_image_list.json"), "w", encoding="utf-8") as f:
        json.dump(items, f)
    print("downloading images...")
    download_all_images(items)
