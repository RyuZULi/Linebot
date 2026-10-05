"""
B2：食物辨識模組。
呼叫本機 Ollama 的 minicpm-v 模型辨識照片中的食物品項與粗略份量。

依 B1 測試結果，這裡刻意不用 Ollama 的強制 JSON schema 模式──
schema 模式下 minicpm-v 會直接放棄辨識、回傳空陣列，比亂猜更危險。
改用自由輸出 + 平衡括號擷取 JSON 區塊，並把辨識結果統一轉成繁體中文
（minicpm-v 即使被要求繁體中文，仍經常回覆簡體）。
"""

import base64
import json
import urllib.request

from opencc import OpenCC

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "minicpm-v"

_cc = OpenCC("s2twp")

PROMPT = """這是一張便當或餐點照片。請仔細觀察後，列出照片中「實際看得到」的食物品項與粗略份量估計。

規則：
1. 只寫照片中真的看得到的東西，不要猜測或想像沒看到的食物。
2. 份量只用粗略描述（例如：一份、半碗、一片、兩顆），不要給出精確公克數。
3. 品項名稱請盡量具體（例如「糖醋排骨」優於「肉」，「炒青菜」優於「蔬菜」）。
4. 每個品項附上 confidence（0 到 1）：你有多確定照片裡真的有這個東西。看得很清楚給 0.8 以上；
   看不太清楚、只是覺得可能有的給 0.5 以下。不要每個都給一樣的分數。
5. 品項名稱用繁體中文。
6. 回覆最後附上一段 JSON，格式與範例完全一致：

```json
{"items": [{"name": "白飯", "portion_size": "半碗", "confidence": 0.95}, {"name": "炸雞腿", "portion_size": "一份", "confidence": 0.9}]}
```
"""

# 模型沒給 confidence（或給了非數字）時的預設值
DEFAULT_CONFIDENCE = 0.7


def extract_json_object(text: str):
    """在文字中找第一個平衡的 {...} 區塊並解析成 JSON。

    minicpm-v 常在 JSON 前後夾雜中文說明或 code fence，不能假設
    整段輸出從頭到尾都是純 JSON，所以用括號配對去抓出真正的物件邊界。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def to_traditional(s: str) -> str:
    return _cc.convert(s) if s else s


def recognize_food(image_path: str, temperature: float = 0.1, model: str = None) -> dict:
    """辨識一張照片，回傳 {"ok", "raw", "items"}。

    items 內的 name / portion_size 都已轉成繁體中文，可以直接拿去對
    nutrition_db / portion_reference（兩者皆為繁體）做檢索。

    model：指定要用哪個 Ollama 模型，預設用 B1 選定的 minicpm-v。
    B4(集合辨識)會用這個參數輪流呼叫不同模型。
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    effective_model = model or MODEL
    prompt = PROMPT
    if "qwen3" in effective_model:
        # qwen3-vl 預設會開「思考」模式，實測發現它會把整個 context(就算
        # 加到 8192)全部拿去做內部推理、永遠生不出最終 JSON 答案。加
        # /no_think 關掉思考模式後，19 秒內就能正常回覆。其他模型沒有
        # 這個機制，不受影響，所以只在 qwen3 系列才加這個前綴。
        prompt = "/no_think\n" + PROMPT

    payload = {
        "model": effective_model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw = body.get("response", "")
    parsed = extract_json_object(raw)
    if parsed is None or "items" not in parsed:
        return {"ok": False, "raw": raw, "items": []}

    items = []
    for it in parsed.get("items", []):
        name = to_traditional(str(it.get("name", "")).strip())
        portion = to_traditional(str(it.get("portion_size", "")).strip())
        conf = it.get("confidence")
        conf = min(max(float(conf), 0.0), 1.0) if isinstance(conf, (int, float)) else DEFAULT_CONFIDENCE
        if name:
            items.append({"name": name, "portion_size": portion, "confidence": conf})

    return {"ok": True, "raw": raw, "items": items}


if __name__ == "__main__":
    import sys

    result = recognize_food(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
