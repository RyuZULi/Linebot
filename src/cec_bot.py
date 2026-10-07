"""
CEC_API助手：公司同仁在 LINE 上查 CEC 建築 Revit API 的操作與錯誤（邏輯在 cec_rag.py）。

跟 Ryuzu、Belfast 是三個獨立的 LINE channel，共用 webhook_app.py 的 Flask process
（路徑 /callback/cec）。這個 bot 對所有人開放，口氣中性專業、不套角色人設——
回答內容是查資料得來的操作步驟與錯誤原因，套人設會增加編造的風險（見 CLAUDE.md
PDF RAG 的實測），同仁也不需要角色扮演。
"""

import traceback

from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    PostbackAction,
    QuickReply,
    QuickReplyItem,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)
from linebot.v3.webhooks import FollowEvent, MessageEvent, PostbackEvent, TextMessageContent

import cec_rag

MAX_LINE_TEXT_LENGTH = 5000
MAX_QUICK_REPLY_ITEMS = 13

WELCOME = (
    "您好，我是 CEC 建築 API 助手，可以回答「CEC 建築 Revit API」按鈕的操作方式和錯誤訊息。\n\n"
    "可以這樣問：\n"
    "・建立切割樓板怎麼用\n"
    "・單線轉樑按不了\n"
    "・有沒有可以算磁磚的功能\n"
    "・直接貼上跳出來的錯誤訊息\n\n"
    "機電、土木 API 或 Autodesk 帳號問題不在我的範圍，我會告訴您該找誰。"
)
HELP_WORDS = {"help", "說明", "怎麼用", "使用說明", "你好", "您好", "hi", "hello"}
NON_TEXT_REPLY = "目前只能回答文字問題。如果是錯誤訊息，請直接複製視窗上的文字貼過來，或打出按鈕名稱跟遇到的狀況。"
ERROR_REPLY = "抱歉，查詢時出了問題，請稍後再試一次。急的話請直接聯絡建築 API 負責人。"
PASTE_ERROR_REPLY = "好的，請把錯誤視窗上的文字直接複製貼過來（不用截圖，打字或複製都可以）。"
DESCRIBE_REPLY = "好的，請描述一下：做到哪一步、畫面出現什麼、跟預期哪裡不一樣。"

FEEDBACK_THANKS = {True: "收到，謝謝回饋！", False: "收到，這題我會記下來請負責人補資料。急的話請直接聯絡建築 API 負責人。"}

_configuration = None


def _reply(reply_token: str, text: str, quick_reply: QuickReply = None):
    with ApiClient(_configuration) as client:
        MessagingApi(client).reply_message(
            ReplyMessageRequest(reply_token=reply_token,
                                messages=[TextMessage(text=text[:MAX_LINE_TEXT_LENGTH], quick_reply=quick_reply)])
        )


def _show_loading(user_id: str):
    """LLM 回答要好幾秒，先讓對方看到「輸入中」的動畫。失敗也不影響回答。"""
    try:
        with ApiClient(_configuration) as client:
            MessagingApi(client).show_loading_animation(ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=30))
    except Exception:
        traceback.print_exc()


def _choices_quick_reply(choices: list) -> QuickReply:
    items = [
        QuickReplyItem(action=PostbackAction(label=zh[:20], data=f"cec:pick:{api}", display_text=zh))
        for api, zh in choices[:MAX_QUICK_REPLY_ITEMS - 1]
    ]
    items.append(QuickReplyItem(action=PostbackAction(label="不確定，全部找找看", data="cec:all", display_text="不確定，全部找找看")))
    return QuickReply(items=items)


def _clarify_quick_reply(options: list) -> QuickReply:
    """回問「發生什麼狀況」：卡片自己的常見問題當選項，另外兩個固定選項只是提示使用者怎麼描述。"""
    items = [
        QuickReplyItem(action=PostbackAction(label=f"{i + 1}. {title}"[:20], data=f"cec:item:{i}", display_text=title[:300]))
        for i, title in options
    ]
    items.append(QuickReplyItem(action=PostbackAction(label="有跳出錯誤訊息", data="cec:paste", display_text="有跳出錯誤訊息")))
    items.append(QuickReplyItem(action=PostbackAction(label="其他狀況", data="cec:other", display_text="其他狀況")))
    return QuickReply(items=items[:MAX_QUICK_REPLY_ITEMS])


def _feedback_quick_reply() -> QuickReply:
    """每個回答下面附 👍／👎，記進 qa_log，累積後才分得出哪些題目答錯（跟 Belfast 收集使用者選擇同一個想法）。"""
    return QuickReply(items=[
        QuickReplyItem(action=PostbackAction(label="👍 有幫助", data="cec:fb:up", display_text="👍 有幫助")),
        QuickReplyItem(action=PostbackAction(label="👎 沒解決", data="cec:fb:down", display_text="👎 沒解決")),
    ])


def _respond(reply_token: str, user_id: str, question: str, **kwargs):
    _show_loading(user_id)
    try:
        result = cec_rag.answer(question, user_id=user_id, **kwargs)
    except Exception:
        traceback.print_exc()
        _reply(reply_token, ERROR_REPLY)
        return
    if result["type"] == "ask":
        _reply(reply_token, result["text"], quick_reply=_choices_quick_reply(result["choices"]))
    elif result["type"] == "clarify":
        _reply(reply_token, result["text"], quick_reply=_clarify_quick_reply(result["options"]))
    elif result["type"] == "answer":
        _reply(reply_token, result["text"], quick_reply=_feedback_quick_reply())
    else:
        _reply(reply_token, result["text"])


def on_text(event):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()
    if not text:
        return
    if text.lower() in HELP_WORDS:
        _reply(event.reply_token, WELCOME)
        return
    _respond(event.reply_token, user_id, text)


def on_postback(event):
    data = event.postback.data or ""
    if not data.startswith("cec:"):
        return
    user_id = event.source.user_id
    if data in ("cec:paste", "cec:other"):
        cec_rag._log({"user": user_id, "event": data})
    if data == "cec:paste":
        _reply(event.reply_token, PASTE_ERROR_REPLY)
        return
    if data == "cec:other":
        _reply(event.reply_token, DESCRIBE_REPLY)
        return
    if data.startswith("cec:item:"):
        _respond(event.reply_token, user_id, "", item_index=int(data.split(":", 2)[2]))
        return
    if data.startswith("cec:fb:"):
        good = data == "cec:fb:up"
        cec_rag.log_feedback(user_id, good)
        _reply(event.reply_token, FEEDBACK_THANKS[good])
        return
    # 選按鈕：原本的問題存在 cec_rag 的 session 裡（過期會回「麻煩再問一次」）
    if data == "cec:all":
        _respond(event.reply_token, user_id, "", search_all=True)
    elif data.startswith("cec:pick:"):
        _respond(event.reply_token, user_id, "", forced_api=data.split(":", 2)[2])


def on_follow(event):
    _reply(event.reply_token, WELCOME)


def on_other_message(event):
    _reply(event.reply_token, NON_TEXT_REPLY)


def register(handler, configuration):
    global _configuration
    _configuration = configuration
    handler.add(MessageEvent, message=TextMessageContent)(on_text)
    handler.add(MessageEvent)(on_other_message)
    handler.add(PostbackEvent)(on_postback)
    handler.add(FollowEvent)(on_follow)
