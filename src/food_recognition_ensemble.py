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

# 順序 = 可信度優先順序（B5 單模型正確率：gemma3 71.4% > qwen3-vl 61.7% > minicpm-v 53.8%），
# 越前面的模型先建立分群、菜名平手時也採用它的說法。minicpm-v 墊底：它有「套模板」
# 的毛病，B5 測試 10 張裡有 8 張都說有「糖醋排骨」。
ENSEMBLE_MODELS = ["gemma3:12b", "qwen3-vl:8b", "minicpm-v"]

# 主菜用「肉的種類」做第二層分群：三個模型常常都看到同一道肉類主菜，但菜名各說各話
# （B5 images (3) 白斬雞：糖醋排骨 / 烤鴨 / 滷雞肉），字面比對完全歸不在一起，每道都只有
# 1 票被當成幻覺丟掉——「有一道主菜」這件事本身是共識，反而被丟了。
MEAT_TYPES = [
    ("禽肉", ["雞", "鴨", "鵝"]),
    ("牛羊肉", ["牛", "羊"]),
    ("魚海鮮", ["魚", "鮭", "鯖", "鯛", "鱈", "蝦", "花枝", "魷", "透抽", "蚵", "蟹"]),
    ("豬肉", ["豬", "排骨", "控肉", "爌肉", "焢肉", "五花", "里肌", "叉燒", "香腸", "培根", "火腿", "肉"]),
]
# 這些雖然有肉類字眼，但不是主菜（蛋、肉燥這種配料），不做肉類分群
NOT_MAIN_WORDS = ["蛋", "肉燥", "肉鬆", "肉絲炒", "高湯"]

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


def _meat_type(name: str):
    if any(w in name for w in NOT_MAIN_WORDS):
        return None
    for meat, words in MEAT_TYPES:
        if any(w in name for w in words):
            return meat
    return None


def _find_cluster(clusters: list, name: str, model: str):
    # 第一層：菜名字面相似（同一個模型講兩次相近的名字，例如「青菜」「炒青菜」，也歸同一群）
    for c in clusters:
        if any(_similar(name, n) >= NAME_SIMILARITY_THRESHOLD for n in c["names"]):
            return c
    # 第二層：肉類主菜看肉的種類；但同一個模型講的兩道菜不合併（「炸豬排」+「香腸」是兩道）
    meat = _meat_type(name)
    if meat is None:
        return None
    for c in clusters:
        if c["meat"] == meat and model not in c["conf"]:
            return c
    return None


def _pick_name(cluster: dict, models: list) -> str:
    """信心加總最高的菜名；平手時採用可信度順序較前面的模型的說法。"""
    support = {}
    for name, model, conf in cluster["votes"]:
        s, rank = support.get(name, (0.0, len(models)))
        support[name] = (s + conf, min(rank, models.index(model)))
    return max(support, key=lambda n: (round(support[n][0], 6), -support[n][1]))


UNCERTAIN_MAIN_NAME = "主菜（種類未定）"


def _merge_uncertain_main(items: list, clusters: list, models: list) -> list:
    """沒有任何肉類主菜達到共識、但至少兩個模型都說有肉類主菜（只是連肉的
    種類都不同）時，把這些意見合併成一道「種類未定」的主菜。

    「有主菜」本身是共識，只是不知道是什麼——整道丟掉的話總熱量會嚴重偏低。
    熱量由 C3 取候選菜名熱量密度的中位數估算（見 calorie_estimator）。
    已經有共識主菜時不做這件事：剩下的零散主菜意見多半是 minicpm-v 套模板
    的「糖醋排骨」，再多算一道會重複計算。
    """
    if any(it["ensemble_confidence"] == "共識品項" and _meat_type(it["name"]) for it in items):
        return items

    best_per_model = {}  # model -> (信心, 菜名)：每個模型只取它最有把握的那道主菜
    for c in clusters:
        if c["meat"] is None:
            continue
        for name, model, conf in c["votes"]:
            if _meat_type(name) and conf > best_per_model.get(model, (0.0, ""))[0]:
                best_per_model[model] = (conf, name)
    if len(best_per_model) < 2:
        return items

    candidates = [name for _, name in sorted(best_per_model.values(), reverse=True)]
    score = round(sum(conf for conf, _ in best_per_model.values()) / len(models), 3)
    kept = [it for it in items if not (_meat_type(it["name"]) and it["name"] in candidates)]
    kept.append({
        "name": UNCERTAIN_MAIN_NAME,
        "portion_size": "一份",
        "ensemble_confidence": "共識品項",
        "ensemble_score": score,
        "agree_count": len(best_per_model),
        "seen_names": sorted(set(candidates)),
        "main_candidates": list(dict.fromkeys(candidates)),
    })
    return kept


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

    # {"names", "portions", "conf": {model: 最高信心}, "votes": [(name, model, conf)], "meat"}
    clusters = []
    for model in models:  # 依可信度順序，讓較可信的模型先建立分群
        result = per_model_results[model]
        if not result.get("ok"):
            continue
        for item in result["items"]:
            name = item["name"]
            target = _find_cluster(clusters, name, model)
            if target is None:
                target = {"names": [], "portions": [], "conf": {}, "votes": [], "meat": _meat_type(name)}
                clusters.append(target)
            target["names"].append(name)
            target["portions"].append(item["portion_size"])
            target["votes"].append((name, model, item["confidence"]))
            # 同一個模型對同一群講了兩次（例如「青菜」「炒青菜」），只算一次、取較高信心
            target["conf"][model] = max(item["confidence"], target["conf"].get(model, 0.0))

    final_items = []
    dropped_items = []
    for c in clusters:
        name = _pick_name(c, models)
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

    final_items = _merge_uncertain_main(final_items, clusters, models)

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
