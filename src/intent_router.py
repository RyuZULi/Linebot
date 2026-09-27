"""
文字訊息的意圖判斷：先用固定關鍵字（保證能動作，不受 AI 誤判影響），
比對不到才交給本地 LLM 做語意判斷，抓不到固定關鍵字沒講到的自然說法
（例如「上次吃的是什麼」「幫我看一下之前的」）。

「算熱量」不在這裡判斷——那是收到照片當下用 Quick Reply 讓使用者
明確選的，不會靠文字猜測，比較不會出錯。
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "llama3.2:latest"

HISTORY_KEYWORDS = ["紀錄", "記錄", "歷史", "上次", "之前吃"]
DETAIL_KEYWORDS = ["細節", "明細", "詳細"]
DELETE_KEYWORDS = ["刪除", "刪掉", "不要記", "不用記"]
HELP_KEYWORDS = ["幫助", "說明", "help", "指令", "怎麼用"]

INTENTS = ("HELP", "HISTORY", "DETAIL", "DELETE", "CHAT")

CLASSIFY_PROMPT = """判斷底下這句話的意圖，只回傳一個英文單字，不要解釋：
- HELP：使用者想知道這個機器人怎麼用、有哪些功能
- HISTORY：使用者想看過去的飲食紀錄
- DETAIL：使用者想看上一次估算的詳細品項列表
- DELETE：使用者想刪除某筆紀錄
- CHAT：以上皆非，只是想聊天或問其他問題

句子：{text}
意圖："""


def _keyword_intent(text: str) -> str:
    if any(k in text for k in DELETE_KEYWORDS):
        return "DELETE"
    if any(k in text for k in DETAIL_KEYWORDS):
        return "DETAIL"
    if any(k in text for k in HISTORY_KEYWORDS):
        return "HISTORY"
    if any(k.lower() in text.lower() for k in HELP_KEYWORDS):
        return "HELP"
    return None


def _llm_intent(text: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "prompt": CLASSIFY_PROMPT.format(text=text),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 10},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        answer = body.get("response", "").strip().upper()
        for intent in INTENTS:
            if intent in answer:
                return intent
    except Exception:
        pass
    return "CHAT"  # 分類本身失敗，保守當作聊天處理，不要誤觸發資料操作


def classify_intent(text: str) -> str:
    """關鍵字比對優先（保證動作，不受 AI 影響）；比對不到才交給 LLM。"""
    keyword_result = _keyword_intent(text)
    if keyword_result:
        return keyword_result
    return _llm_intent(text)


TIME_OF_DAY_KEYWORDS = {"morning": ["早上", "早餐", "上午"], "afternoon": ["中午", "下午", "午餐"], "evening": ["晚上", "晚餐", "傍晚"]}


def parse_delete_filter(text: str):
    """回傳 (date_filter, time_of_day)，抓不到就用預設值(今天/整天)。"""
    date_filter = "yesterday" if "昨天" in text else "today"
    time_of_day = None
    for key, words in TIME_OF_DAY_KEYWORDS.items():
        if any(w in text for w in words):
            time_of_day = key
            break
    return date_filter, time_of_day
