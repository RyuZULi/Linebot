"""3 模型 × 2 提示詞 × 10 張便當照片：比較每個模型的辨識結果與信心分數。
只收集原始輸出，評分另外做（品項名稱比對要人工判斷，同義詞太多）。"""
import base64
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\CalorieCalculation\src")
from food_recognition import extract_json_object, to_traditional

IMG_DIR = r"D:\CalorieCalculation\refer\便當"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b5_three_model_results.json")
MODELS = ["minicpm-v", "qwen3-vl:8b", "gemma3:12b"]

COMMON_TAIL = """
4. 每個品項附上 confidence（0 到 1）：你有多確定照片裡真的有這個東西。看得很清楚給 0.8 以上；
   看不太清楚、只是覺得可能有的給 0.5 以下。不要每個都給一樣的分數。
5. 品項名稱用繁體中文。
6. 回覆最後附上一段 JSON，格式與範例完全一致：

```json
{"items": [{"name": "白飯", "portion_size": "半碗", "confidence": 0.95}, {"name": "炸雞腿", "portion_size": "一份", "confidence": 0.9}]}
```
"""

PROMPTS = {
    # A：沿用現有規則（盡量具體），只多加信心分數
    "A_specific": """這是一張便當或餐點照片。請仔細觀察後，列出照片中「實際看得到」的食物品項與粗略份量估計。

規則：
1. 只寫照片中真的看得到的東西，不要猜測或想像沒看到的食物。
2. 份量只用粗略描述（例如：一份、半碗、一片、兩顆），不要給出精確公克數。
3. 品項名稱請盡量具體（例如「糖醋排骨」優於「肉」，「炒青菜」優於「蔬菜」）。""" + COMMON_TAIL,
    # B：明確規定「一道菜一個品項」，混在一起的料不拆開
    "B_dish_level": """這是一張便當或餐點照片。請仔細觀察後，列出照片中「實際看得到」的每一道菜與粗略份量估計。

規則：
1. 只寫照片中真的看得到的東西，不要猜測或想像沒看到的食物。
2. 一道菜列成一個品項：分開盛裝或分開擺放的菜分開列（例如便當裡的主菜、每一格配菜、白飯各一項）；
   同一道菜裡混在一起的料不要拆開（例如炒麵就寫「炒麵」，不要另外列麵、青菜、肉絲；滷肉飯就寫「滷肉飯」）。
   份量用粗略描述（例如：一份、半碗、一片、兩顆），不要給精確公克數。
3. 菜名請用常見的台灣菜名，盡量具體（例如「炸豬排」優於「肉」，「炒高麗菜」優於「蔬菜」）。""" + COMMON_TAIL,
}


def call(model, prompt, b64):
    if "qwen3" in model:
        prompt = "/no_think\n" + prompt
    payload = {"model": model, "prompt": prompt, "images": [b64], "stream": False,
               "options": {"temperature": 0.1, "num_ctx": 8192}}
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
images = sorted(os.listdir(IMG_DIR))
for model in MODELS:  # 依模型分組跑，減少 Ollama 來回切換模型的載入時間
    for pname, prompt in PROMPTS.items():
        for img in images:
            key = f"{model}|{pname}|{img}"
            if key in results and "error" not in results[key]:
                continue
            b64 = base64.b64encode(open(os.path.join(IMG_DIR, img), "rb").read()).decode("utf-8")
            t = time.time()
            try:
                raw = call(model, prompt, b64)
                parsed = extract_json_object(raw)
                items = [{"name": to_traditional(str(i.get("name", ""))), "portion": to_traditional(str(i.get("portion_size", ""))),
                          "conf": i.get("confidence")} for i in (parsed or {}).get("items", [])]
                results[key] = {"ok": parsed is not None, "items": items, "sec": round(time.time() - t, 1),
                                "raw_tail": raw[-300:] if parsed is None else ""}
            except Exception as e:
                results[key] = {"ok": False, "items": [], "sec": round(time.time() - t, 1), "error": str(e)}
            json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(key, results[key]["ok"], len(results[key]["items"]), results[key]["sec"], flush=True)
print("ALL DONE")
