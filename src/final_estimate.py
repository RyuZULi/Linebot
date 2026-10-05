"""
C4：整合估算——已知品項加總 + 整餐校準錨點。

最終呈現給使用者的是兩個都有真實資料依據的數字，刻意不合併成一個
假裝精確的單一總熱量：
  1. 「已知品項加總」：C1~C3 算出來的下限（資料庫查得到的品項才算進去）
  2. 「整餐校準參考」：從 Nutrition5k 裡視覺最相近的參考餐點的已知真實
     總熱量區間——這是查表得來的錨點，不是模型看照片自己生成的數字

Nutrition5k 是美式自助餐廳菜色，跟台灣便當外觀有落差，這個錨點的準確度
天生受限，回覆文字必須誠實講清楚這是「大概量級參考」，不是精確答案。
"""

from calorie_estimator import assemble_meal_estimate, format_reply
from meal_similarity import find_similar_meals


def estimate_with_anchor(image_path: str, items: list) -> dict:
    meal = assemble_meal_estimate(items)

    try:
        similar = find_similar_meals(image_path, top_k=5)
    except FileNotFoundError:
        similar = []
    except Exception:
        similar = []

    anchor = None
    if similar:
        cals = sorted(s["total_calories"] for s in similar)
        anchor = {
            "min": round(cals[0], 1),
            "max": round(cals[-1], 1),
            "median": round(cals[len(cals) // 2], 1),
            "top_match_similarity": similar[0]["similarity"],
            "candidates": similar,
        }

    return {"meal": meal, "anchor": anchor}


def format_final_reply(result: dict) -> str:
    lines = [format_reply(result["meal"])]
    lines.append("")
    lines.append("【整餐校準參考】")

    anchor = result["anchor"]
    if anchor is None:
        lines.append("目前沒有找到夠相似的參考餐點，這部分暫時略過。")
    else:
        lines.append(
            f"從視覺相似的參考資料庫中，外觀最接近的幾道參考餐點，"
            f"已知真實總熱量約在 {anchor['min']}～{anchor['max']} 大卡之間"
            f"（中位數 {anchor['median']} 大卡，最相近視覺相似度 {anchor['top_match_similarity']}）。"
        )
        lines.append(
            "提醒：這個參考資料庫主要是西式自助餐廳菜色，跟台式餐點外觀有落差，"
            "這個區間只能當作「大概量級」參考，不是這一餐的精確答案，"
            "跟上面「已知品項加總」的下限一起看，兩個數字之間的落差正好反映"
            "目前資料庫還查不到的品項可能貢獻了多少熱量。"
        )

    return "\n".join(lines)


def estimate_options(result: dict) -> dict:
    """兩種估計各自的數字，算不出來的是 None。

    刻意不替使用者挑、也不把兩個數字平均：兩者的誤差來源完全不同（逐項
    加總是漏算品項，外觀參考是西式參考餐點跟台菜的外觀落差），由使用者
    選比較準的那個（或自己輸入），選擇結果連同兩個數字一起存起來，累積
    夠多筆之後再分析哪種估計在什麼情況下比較準。"""
    meal = result["meal"]
    anchor = result["anchor"]
    return {
        "items": meal["total_calories"] if meal["total_calories"] > 0 else None,
        "anchor": anchor["median"] if anchor else None,
    }


def format_summary_reply(result: dict) -> str:
    """同時列出兩種估計，細節（逐項明細）要使用者另外叫出來。
    這裡是數據本身，不帶任何角色語氣，人設文字由各個 bot 自己包。"""
    meal = result["meal"]
    anchor = result["anchor"]
    opts = estimate_options(result)
    n_items = len(meal["items"])
    uncounted = meal["uncounted_count"]

    lines = ["這餐的熱量估算："]
    if opts["items"] is not None:
        note = f"，有 {uncounted}/{n_items} 項算不到，可能偏低" if uncounted else ""
        lines.append(f"① 逐項加總：約 {opts['items']:g} 大卡（{meal['overall_tier']}{note}）")
    else:
        lines.append("① 逐項加總：算不出來（辨識出的品項都查不到熱量或疑似誤判）")
    if opts["anchor"] is not None:
        lines.append(f"② 外觀參考：約 {opts['anchor']:g} 大卡（外觀相似的參考餐點，範圍 {anchor['min']:g}～{anchor['max']:g}）")
    else:
        lines.append("② 外觀參考：找不到相似的參考餐點")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    items = json.loads(sys.argv[2])
    result = estimate_with_anchor(sys.argv[1], items)
    print(format_final_reply(result))
