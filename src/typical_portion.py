"""
C2 備援：典型便當份量（熟重）。

C2（portion_lookup）用的是衛教「代換份」，例如飯 1 份 = 兩湯匙 = 50 公克，
辨識模型講的「半碗」「一份」「一顆」常常換算不過去，結果整個便當只算得到
白飯、而且白飯還只算了 1/4 碗（2026/10/5 實測：炸豬排便當只算出 91.5 大卡）。

這裡補一層「一個便當裡這種菜通常有多少公克」：只有在 C2 換算不了時才用，
算出來的結果一律標「相近估算」。每個數字都有出處（見
data/portion_reference/typical_bento_portions.csv 的 source 欄），沒有出處的
菜就不套用、照舊回報無法確定，不自己編份量。

資料庫的熱量是「煮熟的菜」每 100 公克，所以這裡的份量也都是熟重。
"""

import csv

from portion_lookup import parse_quantity

CSV_PATH = r"D:\CalorieCalculation\data\portion_reference\typical_bento_portions.csv"

_rows = None


def _load():
    global _rows
    if _rows is None:
        with open(CSV_PATH, encoding="utf-8") as f:
            _rows = [
                {**r, "keywords": r["keywords"].split("|"), "base_grams": float(r["base_grams"]),
                 "bowl_grams": float(r["bowl_grams"]) if r["bowl_grams"] else None}
                for r in csv.DictReader(f)
            ]
    return _rows


def _find_row(food_name: str):
    """依 CSV 順序比對（越具體的放越前面，例如「滷雞腿」要排在「雞腿」前）。
    名稱以「飯」結尾的一律當主食，避免「滷肉飯」被「滷肉」比對成主菜。"""
    rows = _load()
    if food_name.endswith("飯"):
        return next(r for r in rows if r["kind"] == "主食")
    for r in rows:
        if any(k in food_name for k in r["keywords"]):
            return r
    return None


def _multiplier(kind: str, portion_size: str, mult: float, vague: bool) -> float:
    s = portion_size or ""
    if kind == "主菜":
        # 模型常把同一片排骨/雞排切開的條狀描述成「幾片」「兩片」，便當主菜
        # 通常就是一份，所以只認「半」，其他一律當一份，避免份量被灌水。
        return 0.5 if "半" in s else 1.0
    if kind == "配菜" and vague:
        # 「少許/一點」→ 半格；「幾片/多」不確定多少，當一格。
        return 0.5 if "少" in s or "點" in s else 1.0
    return mult


def typical_grams(food_name: str, portion_size: str) -> dict:
    """回傳 {"grams", "basis", "kind"}；這道菜沒有可靠的典型份量時回傳 None。"""
    row = _find_row(food_name)
    if row is None:
        return None
    unit, mult, vague = parse_quantity(portion_size)
    base = row["bowl_grams"] if unit == "碗" and row["bowl_grams"] else row["base_grams"]
    grams = round(base * _multiplier(row["kind"], portion_size, mult, vague), 1)
    unit_label = "1碗" if unit == "碗" and row["bowl_grams"] else "1份"
    return {
        "grams": grams,
        "kind": row["kind"],
        "basis": f"典型便當份量：{row['kind']}{unit_label}約 {base:g} 公克（{row['source']}）",
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(typical_grams(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
