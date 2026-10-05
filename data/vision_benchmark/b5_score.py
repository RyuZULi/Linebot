import difflib
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
results = json.load(open(os.path.join(D, "b5_three_model_results.json"), encoding="utf-8"))
labels = {k: v for k, v in json.load(open(os.path.join(D, "b5_labels.json"), encoding="utf-8")).items() if not k.startswith("_")}
MODELS = ["minicpm-v", "qwen3-vl:8b", "gemma3:12b"]


def sim(a, b):
    if not a or not b:
        return 0
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def label_match(name, label):
    return max(sim(name, syn) for syn in label.split("/")) >= 0.5


def cluster(per_model):
    """per_model: {model: [items]} → [{"names", "models": {model: conf}, "portions"}]"""
    clusters = []
    for model, items in per_model.items():
        for it in items:
            conf = it["conf"] if isinstance(it["conf"], (int, float)) else 0.7
            target = next((c for c in clusters if any(sim(it["name"], n) >= 0.5 for n in c["names"])), None)
            if target is None:
                target = {"names": [], "models": {}}
                clusters.append(target)
            target["names"].append(it["name"])
            target["models"][model] = max(conf, target["models"].get(model, 0))
    return clusters


def evaluate(prompt, models, mode, threshold=None):
    tp_labels = total_labels = kept = kept_correct = 0
    detail = {}
    for img, labs in labels.items():
        per_model = {}
        for m in models:
            r = results.get(f"{m}|{prompt}|{img}")
            if r and r["ok"]:
                per_model[m] = r["items"]
        cl = cluster(per_model)
        n = len(models)
        accepted = []
        for c in cl:
            if mode == "vote2":
                ok = len(c["models"]) >= 2
                score = len(c["models"])
            else:
                score = sum(c["models"].values()) / n
                ok = score >= threshold
            name = max(set(c["names"]), key=c["names"].count)
            correct = any(label_match(nm, lab) for nm in c["names"] for lab in labs)
            if ok:
                accepted.append(name)
                kept += 1
                kept_correct += correct
        for lab in labs:
            total_labels += 1
            if any(label_match(a, lab) for a in accepted):
                tp_labels += 1
        detail[img] = accepted
    return {"recall": round(tp_labels / total_labels, 3), "precision": round(kept_correct / kept, 3) if kept else 0,
            "kept": kept}, detail


rows = []
ok_counts = {f"{m}|{p}": sum(1 for k, v in results.items() if k.startswith(f"{m}|{p}|") and v["ok"]) for m in MODELS for p in ["A_specific", "B_dish_level"]}
for p in ["A_specific", "B_dish_level"]:
    s, _ = evaluate(p, ["minicpm-v", "qwen3-vl:8b"], "vote2")
    rows.append((p, "舊方法：2模型都講到", s))
    for th in [0.3, 0.4, 0.5]:
        s, det = evaluate(p, MODELS, "conf", th)
        rows.append((p, f"3模型信心分數 ≥{th}", s))

# 單一模型自己的表現（不投票，信心 ≥0.5 就收）
single = {}
for p in ["A_specific", "B_dish_level"]:
    for m in MODELS:
        s, _ = evaluate(p, [m], "conf", 0.5)
        single[f"{p}|{m}"] = s

conf_stats = {}
for m in MODELS:
    vals = [it["conf"] for k, v in results.items() if k.startswith(m + "|") for it in v["items"] if isinstance(it["conf"], (int, float))]
    conf_stats[m] = {"n": len(vals), "distinct": sorted(set(round(x, 2) for x in vals))[:15], "mean": round(sum(vals) / len(vals), 2) if vals else None}

times = {m: round(sum(v["sec"] for k, v in results.items() if k.startswith(m + "|")) / max(1, sum(1 for k in results if k.startswith(m + "|"))), 1) for m in MODELS}

_, best_detail = evaluate("B_dish_level", MODELS, "conf", 0.4)
_, a_detail = evaluate("A_specific", MODELS, "conf", 0.4)
out = {"ok_counts": ok_counts, "methods": rows, "single_model": single, "confidence_stats": conf_stats,
       "avg_seconds": times, "accepted_B_0.4": best_detail, "accepted_A_0.4": a_detail}
json.dump(out, open(os.path.join(D, "b5_scores.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("scored")
