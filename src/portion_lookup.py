"""
C2：份量描述轉換公克數邏輯。

關鍵陷阱（務必先讀）：B2 辨識出的份量詞（半碗、一片、兩顆…）是日常說法，
但 `portion_reference.csv` 的「1份」是衛教用的交換單位——例如飯 1份=兩湯匙
=50公克，遠比一般人認知的「半碗飯」小很多。如果不管單位種類直接把數字
相乘，會嚴重低估／高估熱量。所以這裡的邏輯是：先比對「單位種類」是否
相容，單位對得上才做數字換算；單位對不上（例如辨識說「半碗」、參照表
卻是「湯匙」），寧可承認「份量無法可靠估算」，不要硬湊。
"""

import csv
import re

CSV_PATH = r"D:\CalorieCalculation\data\portion_reference\food_portion_reference.csv"

# 中文數量詞 -> 倍率。"幾"/"多"無法精確對應數字，先給保守估計並標記為模糊。
QUANTITY_WORDS = [
    ("半", 0.5, False),
    ("一", 1.0, False), ("1", 1.0, False), ("兩", 2.0, False), ("二", 2.0, False),
    ("三", 3.0, False), ("四", 4.0, False), ("五", 5.0, False),
    ("幾", 2.0, True), ("多", 2.0, True), ("少", 0.5, True),
]

UNIT_WORDS = ["湯匙", "茶匙", "碗", "杯", "份", "顆", "根", "片", "粒", "塊", "條", "個"]

# 少數幾個單位在特定食物類別下可視為近似相容（僅限這裡明確列出的組合，
# 不做通用换算，因为不同食物的"1碗"重量差异太大，无法一概而论）。
CATEGORY_UNIT_ALIASES = {
    "蔬菜類": {"碗": "份"},  # 手冊原文：半碗~2/3碗 ≈ 1份，粗略視為同一數量級
}

_rows_by_food = None


def _load():
    global _rows_by_food
    if _rows_by_food is not None:
        return
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["unit"] != "N/A"]
    _rows_by_food = rows


def parse_quantity(portion_size: str):
    """從「半碗」「兩顆」「幾片」解析出 (單位, 倍率, 是否模糊)。"""
    s = (portion_size or "").strip()
    unit = next((u for u in UNIT_WORDS if u in s), None)
    for word, mult, vague in QUANTITY_WORDS:
        if word in s:
            return unit, mult, vague
    # 沒有明確數量詞（例如只寫「一份」但用詞是"份"本身），預設倍率 1
    return unit, 1.0, False


def _find_reference_row(food_name: str, category: str = None):
    """在 30 筆份量對照表裡找最相關的一筆，只用簡單子字串比對——
    資料量小，不需要動用向量檢索。"""
    _load()
    candidates = _rows_by_food
    if category:
        same_cat = [r for r in candidates if r["category"] == category]
        if same_cat:
            candidates = same_cat
    for r in candidates:
        if food_name in r["food_item"] or r["food_item"] in food_name:
            return r
    return None


# 只有這兩類有「不分品項都成立」的固定規則，其餘類別份量差異太大，
# 沒有查到對應品項時就該誠實說不知道，而不是套一個平均值充數。
CATEGORY_FLAT_RULES = {
    "蔬菜類": {"grams": 100, "unit": "份"},
    "乳品類": {"grams": 240, "unit": "杯"},
}


def estimate_grams(food_name: str, portion_size: str, category: str = None) -> dict:
    """回傳 {confidence, grams, basis, note}。

    confidence 三種：
      - matched       找到對應品項，且辨識出的單位跟參照表單位相容，換算可信
      - unit_mismatch 找到對應品項，但單位種類對不上，只能給參照表的基準公克數當粗略錨點
      - unknown       完全沒有可用的份量依據，誠實回覆無法估算
    """
    unit, multiplier, vague = parse_quantity(portion_size)
    row = _find_reference_row(food_name, category)

    if row is None:
        flat = CATEGORY_FLAT_RULES.get(category)
        if flat is None:
            return {
                "confidence": "unknown",
                "grams": None,
                "basis": None,
                "note": f"份量對照表沒有「{food_name}」，且「{category}」不是有固定份量規則的類別，無法可靠估算公克數。",
            }
        ref_unit = flat["unit"]
        alias = CATEGORY_UNIT_ALIASES.get(category, {})
        compatible = unit == ref_unit or alias.get(unit) == ref_unit
        if compatible:
            grams = round(flat["grams"] * multiplier, 1)
            return {
                "confidence": "matched" if not vague else "unit_mismatch",
                "grams": grams,
                "basis": f"{category}固定規則：1{ref_unit}={flat['grams']}公克",
                "note": "份量詞模糊（幾/多），僅供粗估" if vague else "",
            }
        return {
            "confidence": "unit_mismatch",
            "grams": flat["grams"],
            "basis": f"{category}固定規則：1{ref_unit}={flat['grams']}公克",
            "note": f"辨識出的單位「{unit or portion_size}」跟規則單位「{ref_unit}」不同類型，無法直接換算，僅提供 1{ref_unit} 的基準值作參考，建議由使用者確認實際份量。",
        }

    ref_unit = row["unit"]
    ref_grams = float(row["grams"])
    alias = CATEGORY_UNIT_ALIASES.get(row["category"], {})
    compatible = unit == ref_unit or alias.get(unit) == ref_unit

    if compatible:
        grams = round(ref_grams * multiplier, 1)
        return {
            "confidence": "unit_mismatch" if vague else "matched",
            "grams": grams,
            "basis": f"{row['food_item']}：1{ref_unit}={ref_grams}公克（{row['source_page']}）",
            "note": "份量詞模糊（幾/多），僅供粗估" if vague else "",
        }

    return {
        "confidence": "unit_mismatch",
        "grams": ref_grams,
        "basis": f"{row['food_item']}：1{ref_unit}={ref_grams}公克（{row['source_page']}）",
        "note": f"辨識出的單位「{unit or portion_size}」跟手冊單位「{ref_unit}」不同類型（例如「碗」不等於「{ref_unit}」），無法直接換算，僅提供 1{ref_unit} 的基準值，建議透過 Quick Reply 讓使用者確認實際份量。",
    }


if __name__ == "__main__":
    import json
    import sys

    result = estimate_grams(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
