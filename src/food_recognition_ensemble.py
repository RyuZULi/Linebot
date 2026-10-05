"""
B4/B5：多模型集合辨識與信心分數投票。

同一張照片用 3 個不同家族的模型分別辨識（單數才能分出多數），每個模型
對每個品項給 0~1 的信心分數，跨模型把同一個品項歸成一群後算：

    共識分數 = 各模型對這個品項的信心加總 ÷ 模型總數（3）

分數 ≥ CONSENSUS_THRESHOLD 才算「共識品項」計入熱量；不到的標成「低共識」，
交給 C3 標示成疑似誤判、不計入。除數固定用設定的模型總數，不是「這次有
成功回覆的模型數」：某個模型失敗時寧可少抓，也不要讓單一模型的意見因為
分母變小而過關。

2026/10/5 改版前是 2 個模型、「兩個都講到才算」，實測（10 張便當照片、
61 道人工標註的菜，見 data/vision_benchmark/B5_REPORT.md）只抓到 44% 的菜，
青菜、肉片這種真的有的配菜大量被當成幻覺丟掉；改成 3 模型 + 門檻 0.4 後
抓到 70.5%，正確率維持 85.7%（舊方法 86.7%）。門檻 0.3 會讓單一模型的高
信心意見過關，正確率掉到 56%，所以不要再往下調。

模型家族刻意錯開（MiniCPM、Qwen、Gemma），同家族的模型容易錯在同一個
地方，投票就失去意義。

品項名稱的比對用字串相似度（difflib）+ 包含關係（「青菜」vs「炒青菜」），
不用語意 embedding：bge-m3 對短食材詞容易被字面誤導（見 C1_REPORT.md）。
"""

import difflib

from food_recognition import recognize_food

ENSEMBLE_MODELS = ["minicpm-v", "qwen3-vl:8b", "gemma3:12b"]

CONSENSUS_THRESHOLD = 0.4

# 份量描述明顯很小、低共識時可以直接不列的關鍵字
SMALL_PORTION_WORDS = ["幾片", "少許", "一片", "一小塊", "一點", "少量", "一小片"]

NAME_SIMILARITY_THRESHOLD = 0.5


def _similar(a: str, b: str) -> float:
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_small_portion(portion_size: str) -> bool:
    return any(w in (portion_size or "") for w in SMALL_PORTION_WORDS)


def recognize_food_ensemble(image_path: str, models: list = None) -> dict:
    """回傳 {"items", "dropped_items", "per_model_raw", "per_model_ok"}。

    items 每筆：name, portion_size, ensemble_confidence（"共識品項"/"低共識"）,
    ensemble_score（0~1）, agree_count, seen_names。
    """
    models = models or ENSEMBLE_MODELS

    per_model_results = {}
    for model in models:
        try:
            per_model_results[model] = recognize_food(image_path, model=model)
        except Exception as e:
            per_model_results[model] = {"ok": False, "raw": str(e), "items": []}

    clusters = []  # {"names": [...], "portions": [...], "conf": {model: 最高信心}}
    for model, result in per_model_results.items():
        if not result.get("ok"):
            continue
        for item in result["items"]:
            name = item["name"]
            target = next(
                (c for c in clusters if any(_similar(name, n) >= NAME_SIMILARITY_THRESHOLD for n in c["names"])),
                None,
            )
            if target is None:
                target = {"names": [], "portions": [], "conf": {}}
                clusters.append(target)
            target["names"].append(name)
            target["portions"].append(item["portion_size"])
            # 同一個模型對同一群講了兩次（例如「青菜」「炒青菜」），只算一次、取較高信心
            target["conf"][model] = max(item["confidence"], target["conf"].get(model, 0.0))

    final_items = []
    dropped_items = []
    for c in clusters:
        name = max(set(c["names"]), key=c["names"].count)
        portion = max(set(c["portions"]), key=c["portions"].count)
        score = round(sum(c["conf"].values()) / len(models), 3)
        consensus = score >= CONSENSUS_THRESHOLD

        if not consensus and _is_small_portion(portion):
            dropped_items.append(
                {"name": name, "portion_size": portion, "ensemble_score": score, "reason": "低共識且份量描述很小，予以捨去"}
            )
            continue

        final_items.append(
            {
                "name": name,
                "portion_size": portion,
                "ensemble_confidence": "共識品項" if consensus else "低共識",
                "ensemble_score": score,
                "agree_count": len(c["conf"]),
                "seen_names": sorted(set(c["names"])),
            }
        )

    return {
        "items": final_items,
        "dropped_items": dropped_items,
        "per_model_raw": {m: r.get("items", []) for m, r in per_model_results.items()},
        "per_model_ok": {m: r.get("ok", False) for m, r in per_model_results.items()},
    }


if __name__ == "__main__":
    import json
    import sys

    result = recognize_food_ensemble(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
