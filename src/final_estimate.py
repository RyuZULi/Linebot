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


def format_summary_reply(result: dict) -> str:
    """2026/9/27 新增：預設只回總熱量，不逐項列出——霽倫閣下的要求，
    使用者不需要每次都看到品項明細，想看再打「細節」叫出來
    （見 format_final_reply，那個存進資料庫的 detail 欄位）。"""
    meal = result["meal"]
    lines = [f"總熱量估算：約 {meal['total_calories']} 大卡（信心程度：{meal['overall_tier']}）"]
    if meal["uncounted_count"] > 0:
        lines.append(f"（有 {meal['uncounted_count']} 項無法確定或疑似誤判，未計入，實際熱量可能更高）")

    anchor = result["anchor"]
    if anchor:
        lines.append(f"整餐視覺校準參考：約 {anchor['min']}～{anchor['max']} 大卡（僅供大概量級參考）")

    lines.append("")
    lines.append("想看每一項的詳細估算，跟本大小姐說「細節」就會給你看。")
    return "\n".join(lines)


PORTION_ADJUST_MULTIPLIERS = {"less": 0.7, "same": 1.0, "more": 1.3}
PORTION_ADJUST_LABELS = {"less": "偏少", "same": "差不多", "more": "偏多"}


def format_adjusted_reply(result: dict, adjust_key: str) -> str:
    """D3：使用者用 Quick Reply 回報「跟估計比起來份量偏少/差不多/偏多」後，
    對「已知品項加總」套一個粗略的整體倍率重新估算。

    這是刻意簡化過的設計：不重新逐項計算份量，而是誠實地說「這是一個
    概略調整」，不假裝變得更精確——真正要精確，需要使用者重新拍照或
    指名哪個品項份量不對，那是更大的功能，這裡先用最小可行的方式讓
    使用者能夠回饋修正。
    """
    multiplier = PORTION_ADJUST_MULTIPLIERS[adjust_key]
    label = PORTION_ADJUST_LABELS[adjust_key]
    base = result["meal"]["total_calories"]
    adjusted = round(base * multiplier, 1)

    if adjust_key == "same":
        return f"了解，維持原本估計：已知品項加總約 {base} 大卡（信心程度：{result['meal']['overall_tier']}）。"

    return (
        f"明白了，份量比原先估計的{label}。已知品項加總概略調整為約 {adjusted} 大卡"
        f"（原估計 {base} 大卡 × {multiplier}，這是粗略整體調整，不是重新逐項計算，僅供參考）。"
    )


if __name__ == "__main__":
    import json
    import sys

    items = json.loads(sys.argv[2])
    result = estimate_with_anchor(sys.argv[1], items)
    print(format_final_reply(result))
