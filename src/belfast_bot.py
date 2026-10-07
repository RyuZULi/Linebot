"""
Belfast：秘書 bot，負責熱量估算 + 飲食紀錄 + 一般聊天。

2026/9/27 從原本的 line_bot.py 拆出來——那時候 Ryuzu 身兼熱量估算跟
開發助手兩個角色，後來決定拆成兩個獨立 LINE bot：Ryuzu 專心做開發
助手（見 dev_bot.py），這邊的熱量估算/飲食紀錄邏輯改用 Belfast 這個
人設接手，程式邏輯完全繼承自舊版 line_bot.py（D2/D3/D5），只有人設
文字跟掛載方式改了。

人設取自《碧藍航線》的 Belfast：沉穩優雅、溫柔體貼的白髮女僕，稱
使用者「主人」，其實很喜歡主人，但這份心意藏在得體的言行舉止底下，
不太直接說出口——這裡的訊息文字（GUIDE_MESSAGE、PROCESSING_MESSAGE
等對話包裝）都套這個語氣；熱量估算的數據表本身（format_final_reply／
format_summary_reply，在 final_estimate.py）維持嚴謹格式，不參雜
角色扮演語氣，避免影響估算內容的可讀性與可信度。

跟 dev_bot.py 共用同一個 Flask process/port（見 webhook_app.py），
用 register(handler, configuration) 把這裡的事件處理器掛到專屬於
Belfast channel 的 WebhookHandler 上，不自己開 Flask app。
"""

import os
import re
import shutil
import tempfile
import threading
import traceback
from datetime import datetime

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
from final_estimate import estimate_with_anchor, format_final_reply, format_summary_reply, estimate_options
import nutrition_lookup
import meal_db
from intent_router import classify_intent, parse_delete_filter, parse_stats_period
from chat_persona import generate_chat_reply
from parse_calorie_report import parse as parse_calorie_report

MAX_LINE_TEXT_LENGTH = 5000

GUIDE_MESSAGE = (
    "主人，想知道熱量的話，把照片交給我就好，我會先問問您是想讓我親自"
    "估算，還是您自己已經有數，跟我說一聲就好。\n"
    "想回顧之前的紀錄，說聲「紀錄」；想看上次的品項細節，說「細節」；"
    "想知道一段時間吃了多少，問我「這週平均每天攝取多少」就好；"
    "不想留下這次的紀錄，跟我說「刪除」（可以加上「今天早上」之類的"
    "時間，我會明白的）。\n"
    "除此之外，不管想聊什麼，我都很樂意陪您聊聊。"
)

PROCESSING_MESSAGE = "好的，主人，交給我吧。這就請三位幫手一起仔細比對，大約需要一分鐘，麻煩您稍等我一下。"

PHOTO_INTENT_TEXT = "照片我收到了，主人。這次是要我親自為您估算，還是您已經知道熱量，想直接告訴我呢？"

REPORT_PROMPT_TEXT = "好，那就麻煩您告訴我這是什麼、大概多少大卡，我會好好記下來，之後也能派上用場。"

REPORT_PARSE_FAIL_TEXT = "抱歉，主人，我沒能從中聽出熱量的數字。可以麻煩您說得再清楚一點嗎？比如「白飯大概280大卡」這樣。"

CANCEL_TEXT = "好的，這一餐我們就不記錄了。不過主人，也別忘了好好照顧自己，這是我在意的事。"

MEAL_CONFIRM_TEXT = (
    "主人覺得哪一個比較接近呢？選一個，我就幫您記下來；"
    "如果您知道實際的熱量，直接告訴我會更準確。\n"
    "想看每一項是怎麼算的，跟我說「細節」就好。"
)

MEAL_UNKNOWN_CONFIRM_TEXT = (
    "這一餐我實在估不出來，真是抱歉，主人。\n"
    "如果您知道大概的熱量，告訴我，我就照您說的記下來。"
)

CORRECTION_PROMPT_TEXT = (
    "好的，請告訴我這一餐大約多少大卡，直接打數字就可以（例如「500」）。\n"
    "如果順便告訴我菜名（例如「炒麵 500大卡」），下次遇到同樣的菜，我就能直接查到了。"
)

NO_FOOD_MESSAGE = "真是抱歉，主人，這張照片我看不太出來是什麼食物。方便的話，可以麻煩您重新拍一張清楚一點的嗎？"

ERROR_MESSAGE = "抱歉，主人，剛才處理的時候出了點小狀況，這次就不算數了。稍後再麻煩您試一次，好嗎？"

PENDING_REMINDER_TEXT = "對了，主人，剛才那一餐還沒記錄喔。要記哪一個呢？選下面的按鈕，或直接告訴我實際的熱量也可以。"

DETAIL_CONFIRM_TEXT ="看完之後，主人想記錄哪一個呢？選下面的按鈕，或直接告訴我實際的熱量也可以。"

NO_PENDING_MEAL_MESSAGE ="抱歉，主人，我這邊已經找不到那筆紀錄了，麻煩您重新傳一次照片。"

NO_PENDING_PHOTO_MESSAGE = "抱歉，主人，剛剛那張照片我這邊已經沒有保留了，請您再傳一次。"

TIME_OF_DAY_LABELS = {"morning": "早上", "afternoon": "下午", "evening": "晚上", None: ""}

_configuration = None

# 算好但「還沒確認」的估算：{"result", "options", "photo_path"}。使用者選了
# 「逐項加總」/「外觀參考」/「手動輸入」/「不記錄」之後才寫進 meal_db
# （2026/10/5 改：之前一算完就直接存，0 大卡這種明顯錯的估計也被存進去）。
# 純記憶體暫存，服務重啟就沒了，使用者重傳照片即可。
pending_meals = {}

# 有記錄的餐點照片留著，跟兩種估計值、使用者的選擇一起存，累積夠多之後
# 用來分析哪種估計比較準（或當評估資料集）。已在 .gitignore。
PHOTO_DIR = r"D:\CalorieCalculation\data\meal_photos"

# 使用者在估算後按了「手動輸入」，等下一則文字訊息當修正數字。
pending_corrections = {}

BARE_NUMBER_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")

# D5：收到照片後，先問使用者要「算熱量」/「我來說熱量」/「不用了」，
# 暫存這張照片的 LINE message_id，選了「算熱量」才會真的下載處理。
pending_images = {}

# D5：使用者選了「我來說熱量」之後，等待下一則文字訊息當熱量回報解析。
pending_reports = {}


def _meal_confirm_quick_reply(options: dict) -> QuickReply:
    items = []
    if options["items"] is not None:
        label = f"逐項 {options['items']:g} 大卡"
        items.append(QuickReplyItem(action=PostbackAction(label=label, data="meal:items", display_text=label)))
    if options["anchor"] is not None:
        label = f"外觀 {options['anchor']:g} 大卡"
        items.append(QuickReplyItem(action=PostbackAction(label=label, data="meal:anchor", display_text=label)))
    items.append(QuickReplyItem(action=PostbackAction(label="手動輸入", data="meal:correct", display_text="手動輸入")))
    items.append(QuickReplyItem(action=PostbackAction(label="不記錄", data="meal:skip", display_text="不記錄")))
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
    pending_corrections.pop(user_id, None)
    _reply(event.reply_token, PHOTO_INTENT_TEXT, quick_reply=_photo_intent_quick_reply())


def on_text(event):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    if pending_corrections.get(user_id):
        if _parse_kcal_text(text):
            _handle_correction_text(event.reply_token, user_id, text)
            return
        # 按了「手動輸入」卻改問別的：當一般問題回答，最後會再提醒這餐還沒記錄
        pending_corrections.pop(user_id, None)

    if pending_reports.get(user_id):
        _handle_report_text(event.reply_token, user_id, text)
        return

    if user_id in pending_meals:
        action = _typed_meal_choice(text)
        if action:
            _confirm_meal(event.reply_token, user_id, action)
            return
        # 確認訊息上寫「知道實際熱量直接告訴我」：直接打「500」「炒麵 500 大卡」就當成手動輸入
        # 問句（「這個 500 大卡是怎麼算的？」）不算
        if _parse_kcal_text(text) and not any(w in text for w in QUESTION_MARKERS):
            _handle_correction_text(event.reply_token, user_id, text)
            return

    intent = classify_intent(text)
    if intent == "HELP":
        reply = GUIDE_MESSAGE
    elif intent == "STATS":
        reply = _format_stats(user_id, text)
    elif intent == "HISTORY":
        reply = _format_history(user_id)
    elif intent == "DETAIL":
        reply = _format_detail(user_id)
    elif intent == "DELETE":
        reply = _handle_delete(user_id, text)
    else:
        reply = generate_chat_reply(text, persona="belfast")

    # 估算還沒確認時被別的問題打斷（「衛福部的資料哪來的？」），回答完主動提醒、重新附上按鈕，
    # 不然按鈕一消失，主人很容易忘了這餐還沒記錄。
    pending = pending_meals.get(user_id)
    if pending:
        reminder = DETAIL_CONFIRM_TEXT if intent == "DETAIL" else PENDING_REMINDER_TEXT
        _reply(event.reply_token, reply + "\n\n" + reminder, quick_reply=_meal_confirm_quick_reply(pending["options"]))
    else:
        _reply(event.reply_token, reply)


def on_postback(event):
    data = event.postback.data or ""
    user_id = event.source.user_id

    if data.startswith("meal:"):
        action = data.split(":", 1)[1]
        pending = pending_meals.get(user_id)
        if pending is None:
            _reply(event.reply_token, NO_PENDING_MEAL_MESSAGE)
            return

        _confirm_meal(event.reply_token, user_id, action)
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
            # 留著 message_id：收到熱量後要下載照片一起存（照片 + 主人給的
            # 正確熱量，是之後評估估算準不準最有價值的資料）。
            pending_reports[user_id] = pending_images.pop(user_id, None) or "no_photo"
            _reply(event.reply_token, REPORT_PROMPT_TEXT)

        elif action == "cancel":
            pending_images.pop(user_id, None)
            _reply(event.reply_token, CANCEL_TEXT)

        return


def _confirm_meal(reply_token: str, user_id: str, action: str):
    """處理「逐項／外觀／手動輸入／不記錄」：按鈕（postback）跟直接打字都走這裡。"""
    pending = pending_meals[user_id]
    if action in ("items", "anchor"):
        kcal = pending["options"][action]
        if kcal is None:
            _reply(reply_token, MEAL_UNKNOWN_CONFIRM_TEXT, quick_reply=_meal_confirm_quick_reply(pending["options"]))
            return
        pending_meals.pop(user_id)
        pending_corrections.pop(user_id, None)
        _save_estimate(user_id, pending, kcal, choice=action)
        _reply(reply_token, f"好的，已經幫您記錄為約 {_kcal(kcal)} 大卡，主人。")

    elif action == "correct":
        pending_corrections[user_id] = True
        _reply(reply_token, CORRECTION_PROMPT_TEXT)

    elif action == "skip":
        _discard_pending(user_id)
        _reply(reply_token, CANCEL_TEXT)


# 估算還沒確認時，直接打字也能選（快速回覆按鈕一有新訊息就會消失，
# 2026/10/7 主人說了「細節」之後按鈕不見，就沒辦法記錄了）。
QUESTION_MARKERS = ["?", "？", "嗎", "怎麼", "為什麼", "如何", "多少"]
TYPED_MEAL_CHOICES = {"逐項": "items", "外觀": "anchor", "手動輸入": "correct", "不記錄": "skip", "不用記錄": "skip"}


def _typed_meal_choice(text: str):
    for word, action in TYPED_MEAL_CHOICES.items():
        if text.startswith(word):
            return action
    return None


def _parse_kcal_text(text: str):
    """「炒麵 500大卡」→ {"food_name": "炒麵", ...}；只打「500」也接受（菜名為「這餐」）。"""
    parsed = parse_calorie_report(text)
    if parsed:
        return parsed
    m = BARE_NUMBER_PATTERN.match(text)
    return {"food_name": "這餐", "kcal": float(m.group(1))} if m else None


def _download_photo(message_id: str, user_id: str):
    """把 LINE 上的照片存到 PHOTO_DIR，失敗就回傳 None（不影響記錄熱量本身）。"""
    try:
        with ApiClient(_configuration) as api_client:
            content = MessagingApiBlob(api_client).get_message_content(message_id)
        os.makedirs(PHOTO_DIR, exist_ok=True)
        path = os.path.join(PHOTO_DIR, f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        with open(path, "wb") as f:
            f.write(content)
        return path
    except Exception:
        traceback.print_exc()
        return None


def _handle_report_text(reply_token: str, user_id: str, text: str):
    parsed = _parse_kcal_text(text)
    if not parsed:
        _reply(reply_token, REPORT_PARSE_FAIL_TEXT)
        return

    message_id = pending_reports.pop(user_id, None)
    photo_path = _download_photo(message_id, user_id) if message_id and message_id != "no_photo" else None
    kcal = parsed["kcal"]
    food_name = parsed["food_name"] if parsed["food_name"] != "這餐" else None
    label = food_name or "這餐"

    meal_db.save_record(
        user_id,
        source="user_reported",
        summary=f"{label}約 {_kcal(kcal)} 大卡（主人輸入）",
        detail=f"「{label}」由主人親自回報，約 {_kcal(kcal)} 大卡。",
        total_calories=kcal,
        raw={"choice": "user", "user_kcal": kcal, "photo_path": photo_path},
    )
    reply = f"記下了，主人。這餐大約 {_kcal(kcal)} 大卡，已經放進您的紀錄裡。"
    # 沒講菜名就不加進知識庫：「這餐 = 146 大卡」這種資料對之後的查詢沒有意義，只會污染。
    if food_name:
        nutrition_lookup.add_user_food(food_name, kcal, source="使用者回報")
        reply = f"記下了，主人。「{food_name}」大約 {_kcal(kcal)} 大卡，已經放進您的紀錄，下次也查得到這道菜。"
    _reply(reply_token, reply)


def _kcal(value: float) -> str:
    """500.0 → "500"、560.8 → "560.8"。"""
    return f"{value:g}"


CHOICE_LABELS = {"items": "逐項加總", "anchor": "外觀參考", "user": "主人輸入"}


def _discard_pending(user_id: str):
    """放棄還沒確認的估算，連同暫存的照片一起刪掉（不記錄就不留照片）。"""
    pending_corrections.pop(user_id, None)
    pending = pending_meals.pop(user_id, None)
    if pending and pending.get("photo_path") and os.path.exists(pending["photo_path"]):
        os.remove(pending["photo_path"])


def _save_estimate(user_id: str, pending: dict, kcal: float, choice: str, food_name: str = None):
    """把確認過的估算寫進 meal_db。raw 裡同時保留兩種估計值、逐項明細、
    照片路徑、使用者最後選哪個（或自己輸入多少）——累積夠多筆之後，才能
    用真實資料分析哪種估計在什麼情況下比較準，而不是靠猜。"""
    result, options = pending["result"], pending["options"]
    anchor = result["anchor"]
    label = food_name or "這餐"
    summary = f"{label}約 {_kcal(kcal)} 大卡（{CHOICE_LABELS[choice]}）"
    meal_db.save_record(
        user_id,
        source="estimated_corrected" if choice == "user" else "estimated",
        summary=summary,
        detail=format_final_reply(result),
        total_calories=kcal,
        raw={
            "choice": choice,
            "user_kcal": kcal if choice == "user" else None,
            "items_total": options["items"],
            "items_uncounted": result["meal"]["uncounted_count"],
            "items_count": len(result["meal"]["items"]),
            "items": [
                {"name": it["name"], "portion_size": it["portion_size"], "tier": it["tier"], "calories": it["calories"]}
                for it in result["meal"]["items"]
            ],
            "anchor_median": options["anchor"],
            "anchor_range": [anchor["min"], anchor["max"]] if anchor else None,
            "photo_path": pending.get("photo_path"),
        },
    )


def _handle_correction_text(reply_token: str, user_id: str, text: str):
    pending = pending_meals.get(user_id)
    if pending is None:
        pending_corrections.pop(user_id, None)
        _reply(reply_token, NO_PENDING_MEAL_MESSAGE)
        return

    parsed = _parse_kcal_text(text)
    if not parsed:
        _reply(reply_token, REPORT_PARSE_FAIL_TEXT)
        return

    pending_corrections.pop(user_id, None)
    pending_meals.pop(user_id)
    kcal = parsed["kcal"]
    food_name = parsed["food_name"] if parsed["food_name"] != "這餐" else None
    _save_estimate(user_id, pending, kcal, choice="user", food_name=food_name)

    reply = f"了解，已經照您說的記錄為 {_kcal(kcal)} 大卡，主人。"
    if food_name:
        nutrition_lookup.add_user_food(food_name, kcal, source="主人修正估算")
        reply += f"「{food_name}」我也記住了，下次就能直接查到。"
    _reply(reply_token, reply)


def _format_stats(user_id: str, text: str) -> str:
    start, end, label = parse_stats_period(text)
    days = meal_db.daily_totals(user_id, start, end)
    period = f"{start.month}/{start.day}" if start == end else f"{start.month}/{start.day}～{end.month}/{end.day}"
    if not days:
        return f"{label}（{period}）還沒有任何紀錄喔，主人。"

    total = round(sum(d["total"] for d in days), 1)
    lines = [f"【{label}的熱量攝取】（{period}）", f"總共：約 {_kcal(total)} 大卡"]
    if start != end:
        avg = round(total / len(days), 1)
        lines.append(f"平均每天：約 {_kcal(avg)} 大卡（以有紀錄的 {len(days)} 天計算）")
        lines.append("")
        for d in days:
            lines.append(f"{d['date'][5:].replace('-', '/')}：{_kcal(d['total'])} 大卡（{d['count']} 筆）")
        lines.append("")
        lines.append("沒有紀錄的日子不算進平均；如果有幾餐忘了記，實際的平均會更高一些，主人。")
    return "\n".join(lines)


def _format_history(user_id: str) -> str:
    rows = meal_db.get_recent(user_id, limit=10)
    if not rows:
        return "目前還沒有留下任何紀錄呢，主人。傳張照片，或告訴我您吃了什麼，我都很樂意幫您記著。"
    lines = ["【最近的飲食紀錄】"]
    for r in rows:
        lines.append(f"{r['created_at']}　{r['summary']}")
    return "\n".join(lines)


def _format_detail(user_id: str) -> str:
    pending = pending_meals.get(user_id)
    if pending:
        return format_final_reply(pending["result"])
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
        return f"{date_label}{time_label}沒有紀錄可以刪除喔，主人。"
    return f"好的，{date_label}{time_label}的 {count} 筆紀錄我已經刪掉了，主人。"


def _process_image_async(message_id: str, user_id: str):
    """在背景執行緒跑：下載圖片 → 集合辨識(B4) → 整合估算(C1~C4)
    → push 主要估計 + 確認按鈕。這裡不寫資料庫，使用者確認後才存。"""
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
                "ensemble_score": it.get("ensemble_score"),
                "agree_count": it.get("agree_count"),
                "main_candidates": it.get("main_candidates"),
            }
            for it in recognition["items"]
        ]
        result = estimate_with_anchor(tmp_path, items)
        options = estimate_options(result)

        _discard_pending(user_id)
        os.makedirs(PHOTO_DIR, exist_ok=True)
        photo_path = os.path.join(PHOTO_DIR, f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        shutil.copyfile(tmp_path, photo_path)
        pending_meals[user_id] = {"result": result, "options": options, "photo_path": photo_path}

        has_estimate = options["items"] is not None or options["anchor"] is not None
        _push_multi(
            user_id,
            [
                TextMessage(text=format_summary_reply(result)[:MAX_LINE_TEXT_LENGTH]),
                TextMessage(
                    text=MEAL_CONFIRM_TEXT if has_estimate else MEAL_UNKNOWN_CONFIRM_TEXT,
                    quick_reply=_meal_confirm_quick_reply(options),
                ),
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
