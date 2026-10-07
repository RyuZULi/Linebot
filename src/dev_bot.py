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

2026/9/30 新增 PDF 統整 + RAG（邏輯在 pdf_rag.py，這裡只負責 LINE 互動）：
  - owner 傳 PDF → 背景統整 → push 摘要 + 「加入資料庫/不用了」按鈕
  - 一般聊天時自動判斷要不要查 PDF 資料庫（語意檢索門檻 + LLM 判斷
    段落是否相關），不需要特殊指令
  - 「PDF 清單」「刪除 PDF #3」「刪掉 xxx.pdf」之類的自然語句可以
    列出/刪除文件，沒指定是哪份就跳按鈕讓 owner 選

2026/10/1 新增「開 Claude Code 對話窗」：owner 傳「開一個 claude code 對話窗」
之類的句子（同時提到 claude 跟 開/對話窗/視窗/終端機），就在這台電腦上
開一個新的命令列視窗跑互動式 claude，工作目錄是專案根目錄。只開視窗、
不帶任何 prompt 也不加權限參數，後續操作都由人在視窗裡自己決定。

2026/10/8 新增 YouTube 影片統整（邏輯在 youtube_summary.py）：owner 的訊息裡有
YouTube 連結 → 背景抓字幕、統整 → push 摘要。只做統整，不加入資料庫。

跟 belfast_bot.py 共用同一個 Flask process/port（見 webhook_app.py），
用 register(handler, configuration) 掛到專屬於 Ryuzu channel 的
WebhookHandler 上。
"""

import json
import os
import re

import subprocess
import threading
import traceback

from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, FileMessageContent, PostbackEvent

import pdf_rag
import youtube_summary
import task_db
import task_runner
from chat_persona import generate_chat_reply

MAX_LINE_TEXT_LENGTH = 5000
OWNER_FILE = r"D:\CalorieCalculation\data\dev_owner.json"

TASK_TRIGGER_PREFIXES = ["任務", "修bug", "修 bug", "新功能", "加功能", "新增功能"]
TASK_LIST_KEYWORDS = ["任務清單", "任務列表", "查任務"]
TASK_DONE_PATTERN = re.compile(r"(?:任務)?完成(?:任務)?\s*#?(\d+)")

PDF_KEYWORDS = ["pdf", "文件"]
PDF_LIST_KEYWORDS = ["清單", "列表", "有哪些", "列出"]
PDF_DELETE_KEYWORDS = ["刪除", "刪掉", "移除"]
MAX_QUICK_REPLY_ITEMS = 13

CLAUDE_WINDOW_KEYWORDS = ["開", "對話窗", "視窗", "終端機", "terminal"]

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


def _is_open_claude_command(text: str) -> bool:
    lowered = text.lower()
    return "claude" in lowered and any(k in lowered for k in CLAUDE_WINDOW_KEYWORDS)


def _open_claude_window() -> None:
    """在這台電腦上開一個新的命令列視窗跑互動式 claude（claude 是 npm 裝的 claude.cmd，要透過 cmd 呼叫）。"""
    subprocess.Popen(
        ["cmd", "/c", "start", "Claude Code", "cmd", "/k", "claude"],
        cwd=task_runner.REPO_ROOT,
    )


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


def _pdf_add_quick_reply(doc_id: int) -> QuickReply:
    items = [
        QuickReplyItem(action=PostbackAction(label="加入資料庫", data=f"pdf:add:{doc_id}", display_text="加入資料庫")),
        QuickReplyItem(action=PostbackAction(label="不用了", data=f"pdf:skip:{doc_id}", display_text="不用了")),
    ]
    return QuickReply(items=items)


def _pdf_delete_quick_reply(docs: list) -> QuickReply:
    items = [
        QuickReplyItem(
            action=PostbackAction(
                label=f"#{d['id']} {d['filename']}"[:20], data=f"pdf:delete:{d['id']}", display_text=f"刪除 #{d['id']}"
            )
        )
        for d in docs[:MAX_QUICK_REPLY_ITEMS]
    ]
    return QuickReply(items=items)


def _format_pdf_list(user_id: str) -> str:
    docs = pdf_rag.list_docs(user_id)
    if not docs:
        return "資料庫裡一份 PDF 都沒有，霽倫閣下。直接把檔案丟過來，本大小姐幫您統整。"
    status_label = {"in_rag": "已加入資料庫", "summarized": "只有統整"}
    lines = ["【PDF 文件】"]
    for d in docs:
        lines.append(f"#{d['id']}　{d['filename']}　[{status_label.get(d['status'], d['status'])}]")
    lines.append("\n想刪掉哪份，說「刪除 PDF #編號」就好。")
    return "\n".join(lines)


def _handle_pdf_command(reply_token: str, user_id: str, text: str) -> bool:
    """處理 PDF 清單/刪除的自然語句，有處理回傳 True。"""
    lowered = text.lower()
    mentions_pdf = any(k in lowered for k in PDF_KEYWORDS)

    if any(k in text for k in PDF_DELETE_KEYWORDS):
        matched = pdf_rag.find_docs(user_id, text)
        if not matched and not mentions_pdf:
            return False
        if len(matched) == 1:
            doc = matched[0]
            pdf_rag.delete_doc(doc["id"])
            _reply(reply_token, f"「{doc['filename']}」(#{doc['id']}) 已經從資料庫刪乾淨了——才不會捨不得呢。")
            return True
        candidates = matched or pdf_rag.list_docs(user_id)
        if not candidates:
            _reply(reply_token, "資料庫裡沒有任何 PDF 可以刪，霽倫閣下。")
            return True
        _reply(reply_token, "要刪掉哪一份？選一個吧。", quick_reply=_pdf_delete_quick_reply(candidates))
        return True

    if mentions_pdf and any(k in text for k in PDF_LIST_KEYWORDS):
        _reply(reply_token, _format_pdf_list(user_id))
        return True

    return False


def _process_pdf_async(message_id: str, user_id: str, filename: str):
    """背景執行緒：下載 PDF → 統整 → push 摘要並詢問要不要加入 RAG 資料庫。"""
    doc_id = None
    try:
        with ApiClient(_configuration) as api_client:
            content = MessagingApiBlob(api_client).get_message_content(message_id)
        doc_id = pdf_rag.save_pdf(user_id, filename, content)
        summary = pdf_rag.summarize_doc(doc_id)
        _push(
            user_id,
            f"【{filename} 統整】\n{summary}\n\n要把這份加進資料庫嗎？加進去之後聊天問到相關內容，本大小姐會自己去翻。",
            quick_reply=_pdf_add_quick_reply(doc_id),
        )
    except ValueError as e:
        if doc_id is not None:
            pdf_rag.delete_doc(doc_id)
        _push(user_id, f"……這份沒辦法統整：{e}")
    except Exception:
        traceback.print_exc()
        if doc_id is not None:
            pdf_rag.delete_doc(doc_id)
        _push(user_id, "統整 PDF 的時候出錯了，才不是本大小姐的問題……麻煩再傳一次。")


def _process_youtube_async(url: str, user_id: str):
    """背景執行緒：抓字幕 → 統整 → push。長影片要分段統整，可能要幾分鐘。"""
    try:
        summary = youtube_summary.summarize_youtube(url)
        _push(user_id, f"【影片統整】\n{summary}")
    except ValueError as e:
        _push(user_id, f"……這部影片沒辦法統整：{e}")
    except Exception:
        traceback.print_exc()
        _push(user_id, "統整影片的時候出錯了，才不是本大小姐的問題……過一會兒再貼一次連結試試。")


def _add_pdf_to_rag_async(doc_id: int, user_id: str):
    try:
        count = pdf_rag.add_to_rag(doc_id)
        doc = pdf_rag.get_doc(doc_id)
        _push(user_id, f"「{doc['filename']}」加進資料庫了（{count} 個段落）。之後直接問就好，不用特別下指令。")
    except Exception as e:
        traceback.print_exc()
        _push(user_id, f"加入資料庫失敗了：{e}")


def on_file(event):
    user_id = event.source.user_id
    if user_id != _owner_user_id:
        return
    filename = event.message.file_name or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        _reply(event.reply_token, "本大小姐目前只看得懂 PDF 檔，霽倫閣下。")
        return
    if (event.message.file_size or 0) > pdf_rag.MAX_PDF_BYTES:
        _reply(event.reply_token, "這份 PDF 太大了（上限 20MB），拆小一點再給本大小姐。")
        return
    _reply(event.reply_token, f"收到「{filename}」，本大小姐這就讀完幫您統整，稍等一下。")
    threading.Thread(target=_process_pdf_async, args=(event.message.id, user_id, filename), daemon=True).start()


def _chat_reply(user_id: str, text: str) -> str:
    """一般聊天：先自動判斷要不要查 PDF 資料庫，用不上才走普通聊天。"""
    try:
        rag_answer = pdf_rag.answer_with_rag(user_id, text)
        if rag_answer:
            return rag_answer
    except Exception:
        traceback.print_exc()
    return generate_chat_reply(text, persona="ryuzu")


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

        if _is_open_claude_command(text):
            try:
                _open_claude_window()
                _reply(event.reply_token, f"Claude Code 的對話窗開好了，在電腦上找找看吧，工作目錄是 {task_runner.REPO_ROOT}。")
            except Exception as e:
                traceback.print_exc()
                _reply(event.reply_token, f"開 Claude Code 對話窗失敗了：{e}")
            return

        if _handle_pdf_command(event.reply_token, user_id, text):
            return

        youtube_url = youtube_summary.find_youtube_url(text)
        if youtube_url:
            _reply(event.reply_token, "收到影片連結，本大小姐這就去看字幕幫您統整，長一點的影片要等幾分鐘。")
            threading.Thread(target=_process_youtube_async, args=(youtube_url, user_id), daemon=True).start()
            return

        _reply(event.reply_token, _chat_reply(user_id, text))
        return

    _reply(event.reply_token, generate_chat_reply(text, persona="ryuzu"))


def _on_pdf_postback(reply_token: str, user_id: str, data: str):
    _, action, doc_id_str = data.split(":")
    doc_id = int(doc_id_str)
    doc = pdf_rag.get_doc(doc_id)
    if doc is None or doc["user_id"] != user_id:
        _reply(reply_token, "找不到這份 PDF，可能已經刪掉了。")
        return

    if action == "add":
        if doc["status"] == "in_rag":
            _reply(reply_token, "這份早就在資料庫裡了，霽倫閣下。")
            return
        _reply(reply_token, "好，正在切段落存進資料庫，稍等。")
        threading.Thread(target=_add_pdf_to_rag_async, args=(doc_id, user_id), daemon=True).start()
    elif action == "skip":
        pdf_rag.delete_doc(doc_id)
        _reply(reply_token, "那就不留了，檔案也清掉了——才不是覺得可惜。")
    elif action == "delete":
        pdf_rag.delete_doc(doc_id)
        _reply(reply_token, f"「{doc['filename']}」(#{doc_id}) 已經從資料庫刪掉了。")


def on_postback(event):
    data = event.postback.data or ""
    user_id = event.source.user_id

    if user_id != _owner_user_id:
        return
    if data.startswith("pdf:"):
        _on_pdf_postback(event.reply_token, user_id, data)
        return
    if not data.startswith("task:"):
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
    handler.add(MessageEvent, message=FileMessageContent)(on_file)
    handler.add(PostbackEvent)(on_postback)
