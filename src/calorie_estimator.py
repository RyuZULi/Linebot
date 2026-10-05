"""
C3：熱量估算與信心標示組裝。

把 C1(食物名稱檢索)跟 C2(份量轉公克數)兜起來，算出每個品項的熱量，
再依兩邊的信心度組合出最終的三級標示，組裝成給使用者看的回覆文字。

三級信心標示（呼應 TODO.md 的設計原則）：
  - 精確對應：C1 精確比對到品項，且 C2 份量單位換算可信
  - 相近估算：C1/C2 任一邊是「相近/單位不match」，數字僅供參考
  - 查無資料：C1 完全找不到品項，或雖然找到品項但完全沒有份量依據
    可以推算重量 —— 這種情況誠實列出「無法確定」，不硬湊熱量，也
    不會被悄悄排除在外不讓使用者知道。
"""

from nutrition_lookup import lookup as nutrition_lookup
from portion_lookup import estimate_grams
from typical_portion import typical_grams


def _estimate_item(name: str, portion_size: str, ensemble_confidence: str = None, ensemble_score: float = None,
                   agree_count: int = None) -> dict:
    """ensemble_confidence：B4 集合辨識的投票結果（"共識品項"/"單一模型"），
    只有一個模型講、另一個模型沒有附和的品項最可能是幻覺（B2/B4 報告記錄
    的套模板問題），這裡要把這個警訊帶進最終信心標示，不能算完 B4 的投票
    卻沒讓它影響到使用者最後看到的結果——這正是 2026/9/27 實機測試抓到
    的漏洞：B4 白算了，資訊在接回 D2 時被捨棄掉。"""
    c1 = nutrition_lookup(name)
    category = (c1["matched"] or {}).get("食品分類") if c1["matched"] else None
    c2 = estimate_grams(name, portion_size, category)
    # 2026/10/5：投票改成 3 模型 + 信心分數，低於門檻的標「低共識」
    # （舊版 2 模型時叫「單一模型」，保留相容）。
    single_model = ensemble_confidence in ("低共識", "單一模型")

    # B4 集合投票的安全網放在最前面、優先於資料庫查得到查不到——就算
    # 資料庫查得到品項、份量也換算得出來，只要「只有一個辨識模型講到
    # 這個品項、另一個模型完全沒附和」，就代表這個品項本身有不小的機率
    # 是幻覺（B2/B4 報告記錄的套模板問題，2026/9/27 實機測試證實「糖醋
    # 排骨」正是這種案例）。這段一定要放在其他分支的 return 之前，不然
    # 像「份量換算不出來」這種分支會搶先 return，這個防呆永遠輪不到。
    if single_model:
        # 使用者回報的資料欄位叫 food_name，不是 TFDA 的「品項名稱」
        matched_name = (c1["matched"] or {}).get("品項名稱") or (c1["matched"] or {}).get("food_name")
        matched_note = f"（資料庫查得到「{matched_name}」的熱量，但不採用）" if matched_name else ""
        vote_note = (
            f"3 個辨識模型中只有 {agree_count} 個講到「{name}」（共識分數 {ensemble_score}，未達門檻）"
            if agree_count is not None and ensemble_score is not None
            else f"辨識模型對「{name}」的意見不一致"
        )
        return {
            "name": name,
            "portion_size": portion_size,
            "tier": "疑似誤判",
            "calories": None,
            "detail": f"{vote_note}，可能是辨識錯誤{matched_note}，為求保守不計入總熱量。",
        }

    if c1["confidence"] == "none":
        return {
            "name": name,
            "portion_size": portion_size,
            "tier": "查無資料",
            "calories": None,
            "detail": f"資料庫查無「{name}」對應品項，無法確定熱量，不予估算。",
        }

    # 使用者自己回報過的資料：存的是「這份/這個品項」的總熱量，不是
    # 每100公克密度，不能拿去乘份量公克數（不同量綱），直接採用。
    if c1["confidence"] in ("exact_user", "similar_user"):
        fixed_kcal = float(c1["matched"]["fixed_kcal"])
        tier = "精確對應" if c1["confidence"] == "exact_user" else "相近估算"
        reason = "" if c1["confidence"] == "exact_user" else f"「{name}」沒有完全相符的回報紀錄，使用相近的「{c1['matched']['food_name']}」數值；"
        return {
            "name": name,
            "portion_size": portion_size,
            "tier": tier,
            "calories": fixed_kcal,
            "matched_item": c1["matched"]["food_name"],
            "kcal_per_100g": None,
            "detail": f"{reason}使用者回報的數值（{c1['matched'].get('source', '')}），不隨份量調整。",
        }

    matched_item = c1["matched"]["品項名稱"]
    kcal_per_100g = float(c1["matched"]["熱量_kcal_per_100g"])

    # C2 單位對得上（matched）就用手冊換算，最可靠；對不上或查不到時，
    # C2 只能給一個明知不對的基準值（例如「半碗飯」被算成 1 湯匙 50 公克），
    # 改用有出處的典型便當份量（見 typical_portion.py），結果標相近估算。
    if c2["confidence"] != "matched":
        typical = typical_grams(name, portion_size)
        if typical is not None:
            c2 = {"confidence": "typical", "grams": typical["grams"], "basis": typical["basis"], "note": ""}

    if c2["grams"] is None:
        return {
            "name": name,
            "portion_size": portion_size,
            "tier": "查無資料",
            "calories": None,
            "detail": (
                f"對應資料庫品項「{matched_item}」（{kcal_per_100g} kcal/100g），"
                f"但無法確定「{portion_size}」對應的公克數，故不計入總熱量。"
            ),
        }

    calories = round(kcal_per_100g * c2["grams"] / 100, 1)

    if c1["confidence"] == "exact" and c2["confidence"] == "matched":
        tier = "精確對應"
    else:
        tier = "相近估算"

    reasons = []
    if c1["confidence"] == "similar":
        reasons.append(f"「{name}」沒有精確對應品項，使用相近品項「{matched_item}」的數值估算")
    if c1["confidence"] in ("exact_secondary", "similar_secondary"):
        reasons.append(
            f"官方食品成分資料庫(TFDA)查無「{name}」，改用補充知識庫「{matched_item}」的數值"
            f"（來源：{c1['matched'].get('資料來源', '未標明')}，非官方送檢數據，僅供參考）"
        )
    if c2["confidence"] == "unit_mismatch":
        reasons.append(c2["note"])
    if c2["confidence"] == "typical":
        reasons.append(f"約 {c2['grams']:g} 公克，{c2['basis']}")

    return {
        "name": name,
        "portion_size": portion_size,
        "tier": tier,
        "calories": calories,
        "matched_item": matched_item,
        "grams": c2["grams"],
        "kcal_per_100g": kcal_per_100g,
        "detail": "；".join(reasons) if reasons else f"對應資料庫品項「{matched_item}」，{c2['basis']}",
    }


def assemble_meal_estimate(items: list) -> dict:
    """items: [{"name": str, "portion_size": str, "ensemble_confidence": str(optional)}, ...]

    ensemble_confidence 是 B4 集合辨識的投票結果，有帶的話會影響信心
    標示（見 _estimate_item 的說明），沒帶（例如單模型辨識的舊路徑）
    就不套用這層防呆，行為跟以前一樣。
    """
    results = [
        _estimate_item(
            it["name"], it["portion_size"], it.get("ensemble_confidence"),
            it.get("ensemble_score"), it.get("agree_count"),
        )
        for it in items
    ]

    countable = [r for r in results if r["calories"] is not None]
    uncountable = [r for r in results if r["calories"] is None]
    total_calories = round(sum(r["calories"] for r in countable), 1) if countable else 0

    if not countable:
        overall_tier = "查無資料"
    elif uncountable or any(r["tier"] == "相近估算" for r in countable):
        overall_tier = "相近估算"
    else:
        overall_tier = "精確對應"

    return {
        "items": results,
        "total_calories": total_calories,
        "overall_tier": overall_tier,
        "uncounted_count": len(uncountable),
    }


def format_reply(meal: dict) -> str:
    lines = ["【這餐估算結果】"]
    for r in meal["items"]:
        if r["calories"] is not None:
            lines.append(f"・{r['name']}（{r['portion_size']}）→ 約 {r['calories']} 大卡　[{r['tier']}]")
        else:
            lines.append(f"・{r['name']}（{r['portion_size']}）→ 無法確定　[{r['tier']}]")

    lines.append("")
    if meal["uncounted_count"] > 0:
        lines.append(
            f"總熱量估算：約 {meal['total_calories']} 大卡"
            f"（有 {meal['uncounted_count']} 項無法確定，未計入，實際熱量可能更高）"
        )
    else:
        lines.append(f"總熱量估算：約 {meal['total_calories']} 大卡")

    lines.append(f"整體信心程度：{meal['overall_tier']}")

    lines.append("")
    lines.append("【估算依據】")
    for r in meal["items"]:
        lines.append(f"・{r['name']}：{r['detail']}")

    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    items = json.loads(sys.argv[1])
    meal = assemble_meal_estimate(items)
    print(format_reply(meal))
