"""
Ryuzu：開發助手 bot，負責一般聊天 + 「任務：...」指令（修bug/加功能）。

2026/9/27 新增，原本設計是收到任務指令就自動開隔離分支呼叫 claude
CLI 去改程式碼，但那個做法（背景執行緒自動呼叫
`claude --dangerously-skip-permissions`）被 Claude Code 自己的 Auto
Mode 安全分類器擋下來（連 import 測試都被判定成「Create Unsafe
Agents」），霽倫閣下確認後決定改成最單純的版本：這裡只負責把「任務：
...」存進 task_db 當成一份待辦清單，實際的修改還是要由霽倫閣下自己
回來開 Claude Code 對話處理——不做背景自動執行。

任務指令只有綁定的 owner 能觸發，其他人傳一樣的文字只會被當普通
聊天回覆——這個 bot 是不公開的私人頻道，第一個傳訊息的人會被自動
綁定成 owner（見 _load_owner），之後就固定下來。

跟 belfast_bot.py 共用同一個 Flask process/port（見 webhook_app.py），
用 register(handler, configuration) 掛到專屬於 Ryuzu channel 的
WebhookHandler 上。
"""

import json
import os
import re

from linebot.v3.messaging import ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

import task_db
from chat_persona import generate_chat_reply

MAX_LINE_TEXT_LENGTH = 5000
OWNER_FILE = r"D:\CalorieCalculation\data\dev_owner.json"

TASK_TRIGGER_PREFIXES = ["任務", "修bug", "修 bug", "新功能", "加功能", "新增功能"]
TASK_LIST_KEYWORDS = ["任務清單", "任務列表", "查任務"]
TASK_DONE_PATTERN = re.compile(r"(?:任務)?完成(?:任務)?\s*#?(\d+)")

OWNER_BOUND_MESSAGE = (
    "……好，本大小姐記住您了，霽倫閣下，之後只有您傳的「任務：...」才會被當真。"
    "接下來想聊什麼都行，想交代任務就用「任務：」開頭跟本大小姐說——"
    "先說好，本大小姐只負責記下來，實際動手改程式碼還是要您自己回來處理。"
)

_configuration = None
_owner_user_id = None


def _load_owner() -> str:
    env_owner = os.environ.get("DEV_BOT_OWNER_USER_ID", "").strip()
    if env_owner:
        return env_owner
    if os.path.exists(OWNER_FILE):
        try:
            with open(OWNER_FILE, encoding="utf-8") as f:
                return json.load(f).get("owner_user_id")
        except Exception:
            return None
    return None


def _save_owner(user_id: str) -> None:
    os.makedirs(os.path.dirname(OWNER_FILE), exist_ok=True)
    with open(OWNER_FILE, "w", encoding="utf-8") as f:
        json.dump({"owner_user_id": user_id}, f, ensure_ascii=False)


def _reply(reply_token: str, text: str):
    message = TextMessage(text=text[:MAX_LINE_TEXT_LENGTH])
    with ApiClient(_configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[message])
        )


def _parse_task_command(text: str):
    for prefix in TASK_TRIGGER_PREFIXES:
        if text.startswith(prefix):
            rest = text[len(prefix):].lstrip("：: ")
            if rest:
                return rest
    return None


def _format_task_list() -> str:
    tasks = task_db.get_recent(limit=10)
    if not tasks:
        return "目前沒有任何任務紀錄。"
    lines = ["【最近的任務】"]
    for t in tasks:
        lines.append(f"#{t['id']}　[{t['status']}]　{t['description'][:40]}")
    lines.append("")
    lines.append("想標記某筆做完了，跟本大小姐說「完成 #編號」即可。")
    return "\n".join(lines)


def on_text(event):
    global _owner_user_id
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    if _owner_user_id is None:
        _owner_user_id = user_id
        _save_owner(user_id)
        _reply(event.reply_token, OWNER_BOUND_MESSAGE)
        return

    if user_id == _owner_user_id:
        if text in TASK_LIST_KEYWORDS:
            _reply(event.reply_token, _format_task_list())
            return

        done_match = TASK_DONE_PATTERN.search(text)
        if done_match:
            task_id = int(done_match.group(1))
            task = task_db.get_task(task_id)
            if task is None:
                _reply(event.reply_token, f"沒有找到任務 #{task_id}，霽倫閣下。")
            else:
                task_db.update_task(task_id, status="done")
                _reply(event.reply_token, f"任務 #{task_id} 標記完成了——才不是特別開心呢。")
            return

        task_desc = _parse_task_command(text)
        if task_desc:
            task_id = task_db.create_task(user_id, task_desc)
            _reply(
                event.reply_token,
                f"記下了，任務 #{task_id}：{task_desc}\n"
                f"本大小姐先存進清單，實際動手還是要等您自己回來開發才會處理——"
                f"才不是懶得幫您做呢，是這種事本大小姐一個人亂改很危險。",
            )
            return

    _reply(event.reply_token, generate_chat_reply(text, persona="ryuzu"))


def register(handler, configuration):
    global _configuration, _owner_user_id
    _configuration = configuration
    _owner_user_id = _load_owner()
    handler.add(MessageEvent, message=TextMessageContent)(on_text)
