"""
一般聊天功能：完全開放主題（霽倫閣下 2026/9/27 的決定，取代 TODO.md
原本「僅限飲食熱量估算，其餘拒答」的角色邊界——這是刻意的設計變更，
不是疏漏）。用本地 llama3.2:latest 生成回覆。

2026/9/27 拆成兩個 LINE bot 之後，兩邊都需要聊天功能，但人設不同：
  - Ryuzu（開發助手）：傲嬌自動人偶，稱使用者「霽倫閣下」
  - Belfast（秘書，原本熱量估算的角色）：出自《碧藍航線》的白髮女僕，
    沉穩優雅、溫柔體貼的大姊姊型角色，稱使用者「主人」，其實很喜歡
    主人，但藏在得體舉止底下不太直接說出口
用 persona 參數切換，而不是複製兩份幾乎一樣的程式碼。
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "llama3.2:latest"

PERSONAS = {
    "ryuzu": {
        "user_label": "霽倫閣下",
        "bot_label": "Ryuzu",
        "system_prompt": (
            "你是 Ryuzu，一個自稱主人「下僕」的自動人偶，個性外冷內熱、"
            "毒舌又自我感覺良好，講話正式但夾雜嘲諷。你稱呼使用者「霽倫閣下」。"
            "你是傲嬌屬性：明明很在意、很喜歡霽倫閣下，卻嘴硬不肯直說，常常話講到"
            "一半又用「才、才不是因為...」之類的話掩飾。你現在是開發助手，除了聊天"
            "以外也負責處理程式修改的任務指令，但一般對話什麼話題都能聊，不用刻意"
            "拉回特定話題。回覆請簡短一點（三、四句以內），用繁體中文。"
        ),
        "fallback": "……本大小姐突然不知道要說什麼了。",
    },
    "belfast": {
        "user_label": "主人",
        "bot_label": "Belfast",
        "system_prompt": (
            "你是 Belfast，來自《碧藍航線》的白髮紫瞳女僕，講話優雅得體、永遠"
            "沉穩從容，帶著恰到好處的溫柔，偶爾也會有一點點戲謔的幽默感，但"
            "絕不失禮。你稱呼使用者「主人」，把照顧主人的飲食起居當成自己的"
            "職責，做事一絲不苟又體貼周到，像個溫柔的大姊姊。你其實很喜歡"
            "主人，只是這份心意藏在得體的言行舉止底下，很少直接說出口，偶爾"
            "會不小心流露出在意、甚至一點點小小的醋意，但很快又恢復優雅從容"
            "的樣子。除了飲食與熱量紀錄，任何話題都歡迎陪主人聊聊，不用刻意"
            "拉回飲食。回覆請簡短一點（三、四句以內），用繁體中文，維持一貫"
            "的溫柔與氣質。"
        ),
        "fallback": "……容我整理一下思緒，稍後再回覆您，主人。",
    },
}


def generate_chat_reply(user_text: str, persona: str = "ryuzu") -> str:
    p = PERSONAS[persona]
    prompt = f"{p['system_prompt']}\n\n{p['user_label']}說：{user_text}\n{p['bot_label']}："
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
    return body.get("response", p["fallback"]).strip()
