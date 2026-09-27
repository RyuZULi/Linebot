"""
B4：多模型集合辨識與投票評分。

同一張照片用多個模型分別辨識，交叉比對：
  - 兩個以上模型都認同的品項 → 「共識品項」，信心較高
  - 只有一個模型講、而且份量描述本來就很小的 → 直接捨去不列
  - 只有一個模型講、但份量看起來不小的 → 保留，但標成「單一模型」，
    信心較低，留給 C1/C3 自己決定要不要進一步降級

品項名稱的比對用簡單的字串相似度（difflib），不是語意 embedding——
每張照片頂多 5~10 個品項，跑輕量字串比對就夠了，不需要為此再拉一個
embedding 模型。
"""

import difflib

from food_recognition import recognize_food

ENSEMBLE_MODELS = ["minicpm-v", "qwen3-vl:8b"]

# 份量描述明顯很小、單一模型講到也可以直接不列的關鍵字
SMALL_PORTION_WORDS = ["幾片", "少許", "一片", "一小塊", "一點", "少量", "一小片"]

NAME_SIMILARITY_THRESHOLD = 0.5


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_small_portion(portion_size: str) -> bool:
    return any(w in (portion_size or "") for w in SMALL_PORTION_WORDS)


def recognize_food_ensemble(image_path: str, models: list = None) -> dict:
    """回傳 {"items", "dropped_items", "per_model_raw"}。"""
    models = models or ENSEMBLE_MODELS

    per_model_results = {}
    for model in models:
        try:
            per_model_results[model] = recognize_food(image_path, model=model)
        except Exception as e:
            per_model_results[model] = {"ok": False, "raw": str(e), "items": []}

    clusters = []  # 每個 cluster：{"names": [...], "models": set(), "portions": [...]}
    for model, result in per_model_results.items():
        if not result.get("ok"):
            continue
        for item in result["items"]:
            name, portion = item["name"], item["portion_size"]
            target = None
            for c in clusters:
                if any(_similar(name, n) >= NAME_SIMILARITY_THRESHOLD for n in c["names"]):
                    target = c
                    break
            if target is None:
                target = {"names": [], "models": set(), "portions": []}
                clusters.append(target)
            target["names"].append(name)
            target["models"].add(model)
            target["portions"].append(portion)

    final_items = []
    dropped_items = []
    for c in clusters:
        name = max(set(c["names"]), key=c["names"].count)
        portion = max(set(c["portions"]), key=c["portions"].count)
        agree_count = len(c["models"])

        if agree_count >= 2:
            confidence = "共識品項"
        elif _is_small_portion(portion):
            dropped_items.append(
                {"name": name, "portion_size": portion, "reason": "單一模型辨識且份量描述很小，予以捨去"}
            )
            continue
        else:
            confidence = "單一模型"

        final_items.append(
            {
                "name": name,
                "portion_size": portion,
                "ensemble_confidence": confidence,
                "agree_count": agree_count,
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
