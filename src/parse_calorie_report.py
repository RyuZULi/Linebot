"""
解析使用者回報「XX大概是YY大卡」這類自由格式文字。故意做得很簡單
（正規表示式抓數字+單位，其餘文字當品名），這不是要取代辨識模型，
只是接住使用者主動講的內容，抓不到品名就用「這餐」當通用標籤。
"""

import re

CALORIE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:大卡|kcal|卡路里|卡)")
FILLER_WORDS = ["大概", "大約", "差不多", "整份", "一份", "這個", "是", "約", "，", "。", "、"]


def parse(text: str) -> dict:
    """回傳 {"food_name": str, "kcal": float} 或 None（完全抓不到數字時）。"""
    match = CALORIE_PATTERN.search(text)
    if not match:
        return None

    kcal = float(match.group(1))
    remainder = text[: match.start()] + text[match.end() :]
    for w in FILLER_WORDS:
        remainder = remainder.replace(w, "")
    food_name = remainder.strip() or "這餐"

    return {"food_name": food_name, "kcal": kcal}


if __name__ == "__main__":
    import sys

    print(parse(sys.argv[1]))
