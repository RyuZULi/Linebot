"""
Belfast：秘書 bot，負責熱量估算 + 飲食紀錄 + 一般聊天。

2026/9/27 從原本的 line_bot.py 拆出來——那時候 Ryuzu 身兼熱量估算跟
開發助手兩個角色，後來決定拆成兩個獨立 LINE bot：Ryuzu 專心做開發
助手（見 dev_bot.py），這邊的熱量估算/飲食紀錄邏輯改用 Belfast 這個
沉穩秘書人設接手，程式邏輯完全繼承自舊版 line_bot.py（D2/D3/D5），
只有人設文字跟掛載方式改了。

跟 dev_bot.py 共用同一個 Flask process/port（見 webhook_app.py），
用 register(handler, configuration) 把這裡的事件處理器掛到專屬於
Belfast channel 的 WebhookHandler 上，不自己開 Flask app。
"""

import os
import tempfile
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
from linebot.v3.webhooks import MessageEvent, ImageMessageContent, TextMessageContent, PostbackEvent

from food_recognition_ensemble import recognize_food_ensemble
from final_estimate import estimate_with_anchor, format_final_reply, format_summary_reply, format_adjusted_reply
import nutrition_lookup
import meal_db
from intent_router import classify_intent, parse_delete_filter
from chat_persona import generate_chat_reply
from parse_calorie_report import parse as parse_calorie_report

MAX_LINE_TEXT_LENGTH = 5000

GUIDE_MESSAGE = (
    "主人，若想知道熱量，將照片傳給我即可，我會先詢問您是要親自計算，"
    "還是您已經知道熱量、直接告知我就好。\n"
    "想查看歷史紀錄請說「紀錄」，想看上一次的品項細節請說「細節」，"
    "若不想留下紀錄，可以說「刪除」（能加註「今天早上」之類的時間範圍）。\n"
    "除此之外，也歡迎與我聊聊其他話題。"
)

PROCESSING_MESSAGE = "收到了，主人。我這就動用兩套辨識模型仔細比對，大約需要 30 到 60 秒，請您稍候。"

PHOTO_INTENT_TEXT = "照片已經收到，主人。這次是要我為您計算熱量，還是您已經知道數字、由您親自告知我？"

REPORT_PROMPT_TEXT = "明白了。請告訴我這是什麼、大概多少大卡，我會記錄下來，之後也能派上用場。"

REPORT_PARSE_FAIL_TEXT = "抱歉，我沒能從中辨識出熱量數字。麻煩您講得再清楚一些，例如「白飯大概280大卡」，主人。"

CANCEL_TEXT = "好的，這一餐就不記錄了，主人。若之後改變主意，隨時可以再傳一次。"

PORTION_CONFIRM_TEXT = "請問這次估算的份量，跟您實際吃的相比如何？"

NO_FOOD_MESSAGE = "十分抱歉，主人，這張照片我無法辨識出任何食物，能否請您重新拍一張更清楚的照片？"

ERROR_MESSAGE = "十分抱歉，主人，剛剛的處理出了點問題，這次不列入紀錄，請稍後再試一次。"

NO_PENDING_MEAL_MESSAGE = "抱歉，主人，我已經找不到那筆紀錄了，麻煩您重新傳一次照片。"

NO_PENDING_PHOTO_MESSAGE = "抱歉，主人，剛才那張照片我已經不記得了，請重新傳送一次。"

TIME_OF_DAY_LABELS = {"morning": "早上", "afternoon": "下午", "evening": "晚上", None: ""}

_configuration = None

# D3：暫存每個使用者最新一次的估算結果，供「份量偏少/差不多/偏多」
# 調整用。純記憶體暫存，重啟會清空——真正的歷史紀錄存在 meal_db。
pending_meals = {}

# D5：收到照片後，先問使用者要「算熱量」/「我來說熱量」/「不用了」，
# 暫存這張照片的 LINE message_id，選了「算熱量」才會真的下載處理。
pending_images = {}

# D5：使用者選了「我來說熱量」之後，等待下一則文字訊息當熱量回報解析。
pending_reports = {}


def _portion_quick_reply() -> QuickReply:
    items = [
        QuickReplyItem(action=PostbackAction(label="偏少", data="adjust:less", display_text="偏少")),
        QuickReplyItem(action=PostbackAction(label="差不多", data="adjust:same", display_text="差不多")),
        QuickReplyItem(action=PostbackAction(label="偏多", data="adjust:more", display_text="偏多")),
    ]
    return QuickReply(items=items)


def _photo_intent_quick_reply() -> QuickReply:
    items = [
        QuickReplyItem(action=PostbackAction(label="算熱量", data="photo:calc", display_text="算熱量")),
        QuickReplyItem(action=PostbackAction(label="我來說熱量", data="photo:report", display_text="我來說熱量")),
        QuickReplyItem(action=PostbackAction(label="不用了", data="photo:cancel", display_text="不用了")),
    ]
    return QuickReply(items=items)


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


def _push_multi(user_id: str, messages: list):
    with ApiClient(_configuration) as api_client:
        MessagingApi(api_client).push_message(PushMessageRequest(to=user_id, messages=messages))


def on_image(event):
    user_id = event.source.user_id
    pending_images[user_id] = event.message.id
    pending_reports.pop(user_id, None)
    _reply(event.reply_token, PHOTO_INTENT_TEXT, quick_reply=_photo_intent_quick_reply())


def on_text(event):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    if pending_reports.get(user_id):
        _handle_report_text(event.reply_token, user_id, text)
        return

    intent = classify_intent(text)
    if intent == "HELP":
        _reply(event.reply_token, GUIDE_MESSAGE)
    elif intent == "HISTORY":
        _reply(event.reply_token, _format_history(user_id))
    elif intent == "DETAIL":
        _reply(event.reply_token, _format_detail(user_id))
    elif intent == "DELETE":
        _reply(event.reply_token, _handle_delete(user_id, text))
    else:
        _reply(event.reply_token, generate_chat_reply(text, persona="belfast"))


def on_postback(event):
    data = event.postback.data or ""
    user_id = event.source.user_id

    if data.startswith("adjust:"):
        adjust_key = data.split(":", 1)[1]
        result = pending_meals.get(user_id)
        if result is None:
            _reply(event.reply_token, NO_PENDING_MEAL_MESSAGE)
            return
        _reply(event.reply_token, format_adjusted_reply(result, adjust_key))
        return

    if data.startswith("photo:"):
        action = data.split(":", 1)[1]

        if action == "calc":
            message_id = pending_images.pop(user_id, None)
            if message_id is None:
                _reply(event.reply_token, NO_PENDING_PHOTO_MESSAGE)
                return
            _reply(event.reply_token, PROCESSING_MESSAGE)
            threading.Thread(target=_process_image_async, args=(message_id, user_id), daemon=True).start()

        elif action == "report":
            pending_images.pop(user_id, None)
            pending_reports[user_id] = True
            _reply(event.reply_token, REPORT_PROMPT_TEXT)

        elif action == "cancel":
            pending_images.pop(user_id, None)
            _reply(event.reply_token, CANCEL_TEXT)

        return


def _handle_report_text(reply_token: str, user_id: str, text: str):
    parsed = parse_calorie_report(text)
    if not parsed:
        _reply(reply_token, REPORT_PARSE_FAIL_TEXT)
        return

    pending_reports.pop(user_id, None)
    food_name, kcal = parsed["food_name"], parsed["kcal"]

    nutrition_lookup.add_user_food(food_name, kcal, source="使用者回報")
    meal_db.save_record(
        user_id,
        source="user_reported",
        summary=f"{food_name} 約 {kcal} 大卡（使用者回報）",
        detail=f"「{food_name}」由主人親自回報，約 {kcal} 大卡，已存入個人資料庫，之後估算也查得到這筆資料。",
        total_calories=kcal,
    )
    _reply(
        reply_token,
        f"記錄完成，主人。「{food_name}」大約 {kcal} 大卡，我已經存進您的紀錄，之後也查得到這筆資料。",
    )


def _format_history(user_id: str) -> str:
    rows = meal_db.get_recent(user_id, limit=10)
    if not rows:
        return "目前還沒有任何紀錄，主人。傳張照片，或告訴我您吃了什麼吧。"
    lines = ["【最近的飲食紀錄】"]
    for r in rows:
        lines.append(f"{r['created_at']}　{r['summary']}")
    return "\n".join(lines)


def _format_detail(user_id: str) -> str:
    row = meal_db.get_latest(user_id)
    if not row:
        return NO_PENDING_MEAL_MESSAGE
    return row["detail"]


def _handle_delete(user_id: str, text: str) -> str:
    date_filter, time_of_day = parse_delete_filter(text)
    count = meal_db.delete_records(user_id, date_filter, time_of_day)
    date_label = "今天" if date_filter == "today" else "昨天"
    time_label = TIME_OF_DAY_LABELS.get(time_of_day, "")
    if count == 0:
        return f"{date_label}{time_label}沒有可以刪除的紀錄，主人。"
    return f"已經刪除{date_label}{time_label}的 {count} 筆紀錄，主人。"


def _process_image_async(message_id: str, user_id: str):
    """在背景執行緒跑：下載圖片 → 集合辨識(B4) → 整合估算(C1~C4)
    → 存進 meal_db → push 總熱量摘要（明細存資料庫，要另外用
    「細節」指令叫出來）。"""
    tmp_path = None
    try:
        with ApiClient(_configuration) as api_client:
            blob_api = MessagingApiBlob(api_client)
            image_bytes = blob_api.get_message_content(message_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        recognition = recognize_food_ensemble(tmp_path)
        if not recognition["items"]:
            _push(user_id, NO_FOOD_MESSAGE)
            return

        items = [
            {
                "name": it["name"],
                "portion_size": it["portion_size"],
                "ensemble_confidence": it.get("ensemble_confidence"),
            }
            for it in recognition["items"]
        ]
        result = estimate_with_anchor(tmp_path, items)
        pending_meals[user_id] = result

        meal_db.save_record(
            user_id,
            source="estimated",
            summary=f"總熱量約 {result['meal']['total_calories']} 大卡（信心程度：{result['meal']['overall_tier']}）",
            detail=format_final_reply(result),
            total_calories=result["meal"]["total_calories"],
        )

        _push_multi(
            user_id,
            [
                TextMessage(text=format_summary_reply(result)[:MAX_LINE_TEXT_LENGTH]),
                TextMessage(text=PORTION_CONFIRM_TEXT, quick_reply=_portion_quick_reply()),
            ],
        )

    except Exception:
        traceback.print_exc()
        try:
            _push(user_id, ERROR_MESSAGE)
        except Exception:
            traceback.print_exc()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def register(handler, configuration):
    global _configuration
    _configuration = configuration
    handler.add(MessageEvent, message=ImageMessageContent)(on_image)
    handler.add(MessageEvent, message=TextMessageContent)(on_text)
    handler.add(PostbackEvent)(on_postback)
