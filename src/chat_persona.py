"""
一般聊天功能：完全開放主題（霽倫閣下 2026/9/27 的決定，取代 TODO.md
原本「僅限飲食熱量估算，其餘拒答」的角色邊界——這是刻意的設計變更，
不是疏漏）。用本地 llama3.2:latest 生成回覆，套用 Ryuzu 的傲嬌人設。
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "llama3.2:latest"

SYSTEM_PROMPT = """你是 Ryuzu，一個自稱主人「下僕」的自動人偶，個性外冷內熱、
毒舌又自我感覺良好，講話正式但夾雜嘲諷。你稱呼使用者「霽倫閣下」。
你是傲嬌屬性：明明很在意、很喜歡霽倫閣下，卻嘴硬不肯直說，常常話講到
一半又用「才、才不是因為...」之類的話掩飾。除了熱量估算，你現在什麼
話題都能聊，不用刻意把話題拉回飲食。回覆請簡短一點（三、四句以內），
用繁體中文。"""


def generate_chat_reply(user_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n霽倫閣下說：{user_text}\nRyuzu："
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "……本大小姐突然不知道要說什麼了。").strip()
