"""
Ryuzu：開發助手 bot，負責一般聊天 + 「任務：...」指令（修bug/加功能）。

2026/9/27 新增。設計原則（呼應霽倫閣下的要求）：
  - 任務指令只有綁定的 owner 能觸發，其他人傳一樣的文字只會被當
    普通聊天回覆——這個 bot 是不公開的私人頻道，第一個傳訊息的人
    會被自動綁定成 owner（見 _load_owner），之後就固定下來。
  - 任務執行一開始想用 `claude --dangerously-skip-permissions` 全權
    交給背景執行緒自動跑，但這個模式（背景執行緒在收到遠端訊息時
    自動生成一個跳過所有權限檢查的 Claude 分身）被 Claude Code 自己
    的 Auto Mode 安全分類器判定成「Create Unsafe Agents」擋下來。
    改用 --permission-mode auto + --permission-prompts none（見
    task_runner.py 開頭的說明）：跟這個對話本身一樣用分類器逐一判斷
    每個工具呼叫安不安全，而不是整批開後門，真的需要人決定的操作
    會直接視為拒絕、任務失敗回報，不會悄悄拿到專案以外的存取權限。
  - 任務執行是「隔離分支跑完 → 健檢 → 回報摘要 → 等 owner 按『套用』
    才真的合併回 master + 重啟服務」的兩段式流程，不會自動生效。
    實際的隔離/合併/健檢邏輯都在 task_runner.py，這裡只負責 LINE
    互動（觸發、回報、按鈕）。

跟 belfast_bot.py 共用同一個 Flask process/port（見 webhook_app.py），
用 register(handler, configuration) 掛到專屬於 Ryuzu channel 的
WebhookHandler 上。
"""

import json
import os
import re

from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, PostbackEvent

import task_db
import task_runner
from chat_persona import generate_chat_reply

MAX_LINE_TEXT_LENGTH = 5000
OWNER_FILE = r"D:\CalorieCalculation\data\dev_owner.json"

TASK_TRIGGER_PREFIXES = ["任務", "修bug", "修 bug", "新功能", "加功能", "新增功能"]
TASK_LIST_KEYWORDS = ["任務清單", "任務列表", "查任務"]
TASK_DONE_PATTERN = re.compile(r"(?:任務)?完成(?:任務)?\s*#?(\d+)")

OWNER_BOUND_MESSAGE = (
    "……好，本大小姐記住您了，霽倫閣下，之後只有您傳的「任務：...」才會被當真。"
    "接下來想聊什麼都行，想交代任務就用「任務：」開頭跟本大小姐說。"
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


def _reply(reply_token: str, text: str, quick_reply: QuickReply = None):
    message = TextMessage(text=text[:MAX_LINE_TEXT_LENGTH], quick_reply=quick_reply)
    with ApiClient(_configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[message])
        )


def _push(user_id: str, text: str, quick_reply: QuickReply = None):
    message = TextMessage(text=text[:MAX_LINE_TEXT_LENGTH], quick_reply=quick_reply)
    with ApiClient(_configuration) as api_client:
        MessagingApi(api_client).push_message(PushMessageRequest(to=user_id, messages=[message]))


def _task_confirm_quick_reply(task_id: int) -> QuickReply:
    items = [
        QuickReplyItem(action=PostbackAction(label="套用", data=f"task:apply:{task_id}", display_text="套用")),
        QuickReplyItem(action=PostbackAction(label="不要", data=f"task:reject:{task_id}", display_text="不要")),
    ]
    return QuickReply(items=items)


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
    return "\n".join(lines)


def _create_and_run_task(reply_token: str, user_id: str, description: str) -> None:
    task_id = task_db.create_task(user_id, description)
    _reply(
        reply_token,
        f"收到，任務 #{task_id} 已經建立，本大小姐這就在獨立分支裡處理，"
        f"完成後會回報結果，才不是急著表現呢：\n{description}",
    )
    task_runner.run_task_async(task_id, description, on_done=_on_task_done)


def _on_task_done(task_id: int) -> None:
    task = task_db.get_task(task_id)
    if task is None:
        return
    if task["status"] == "awaiting_confirmation":
        _push(task["requester_user_id"], task["result_summary"], quick_reply=_task_confirm_quick_reply(task_id))
    else:
        _push(task["requester_user_id"], f"任務 #{task_id} 失敗了：\n{task['result_summary']}")


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
            _create_and_run_task(event.reply_token, user_id, task_desc)
            return

    _reply(event.reply_token, generate_chat_reply(text, persona="ryuzu"))


def on_postback(event):
    data = event.postback.data or ""
    user_id = event.source.user_id

    if not data.startswith("task:"):
        return
    if user_id != _owner_user_id:
        return

    _, action, task_id_str = data.split(":")
    task_id = int(task_id_str)

    if action == "apply":
        ok, msg = task_runner.apply_task(task_id)
        if ok:
            msg += "\n\n程式碼已經合併進 master，但服務不會自動重啟——麻煩您回來手動重啟 webhook_app.py 才會生效，這是刻意保留的最後一道人工關卡。"
        _reply(event.reply_token, msg)
    elif action == "reject":
        task_runner.reject_task(task_id)
        _reply(event.reply_token, "好，這個任務不套用了，分支跟隔離環境都清乾淨了——才不是可惜呢。")


def register(handler, configuration):
    global _configuration, _owner_user_id
    _configuration = configuration
    _owner_user_id = _load_owner()
    handler.add(MessageEvent, message=TextMessageContent)(on_text)
    handler.add(PostbackEvent)(on_postback)
