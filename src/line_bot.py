"""
D2/D5：LINE webhook + 主流程串接。

2026/9/27 大改版（霽倫閣下實機測試後的需求）：
  1. 收到照片後不再馬上算熱量，先用 Quick Reply 問使用者要「算熱量」
     還是「我來說熱量」還是「不用了」——有些食物使用者自己就知道熱量
     （包裝標示），不需要每次都跑一次辨識。
  2. 算熱量的預設回覆只給總熱量，不逐項列出——想看細節要另外用文字
     指令「細節」叫出來（存在 meal_db 的 detail 欄位）。
  3. 每一次估算/使用者回報都存進 SQLite（meal_db.py），附時間戳記，
     支援用「刪除」+ 時間關鍵字刪掉指定範圍的紀錄。
  4. 使用者回報的熱量除了存進個人紀錄，也會用 nutrition_lookup.
     add_user_food() 存進可查詢的知識庫，之後估算時查得到。
  5. 新增一般聊天功能（完全開放主題，取代 TODO.md 原本「僅限飲食
     熱量估算」的角色邊界——這是霽倫閣下明確要求的設計變更）。
  6. 文字指令用「固定關鍵字優先，比對不到才交給 LLM 判斷」的方式
     觸發（intent_router.py），確保就算 LLM 判斷失準，關鍵字仍然
     保證能動作。

原本 B4 集合辨識耗時 25~60 秒、必須用背景執行緒 + LINE push API
才不會讓 reply token 過期的架構維持不變，只有在使用者明確選了
「算熱量」之後才會觸發。

本地測試需要搭配 cloudflared 之類的工具把 http://localhost:5000
轉成公開 HTTPS 網址，再貼到 LINE Developers Console 的 Webhook URL。
"""

import os
import tempfile
import threading
import traceback

from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
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

load_dotenv()

CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("缺少 LINE_CHANNEL_ACCESS_TOKEN，請確認 .env 有正確設定")
if not CHANNEL_SECRET:
    raise RuntimeError(
        "缺少 LINE_CHANNEL_SECRET——去 LINE Developers Console 的 "
        "「Basic settings」頁籤複製，填進 .env 的 LINE_CHANNEL_SECRET"
    )

app = Flask(__name__)
handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

MAX_LINE_TEXT_LENGTH = 5000

# 角色設定：Ryuzu（《時鐘機關的愛麗絲》裡自稱主人「下僕」的自動人偶，
# 外冷內熱、毒舌又自我感覺良好，講話正式但夾嘲諷，稱呼使用者「霽倫閣下」）。
# 傲嬌屬性——嘴上嫌麻煩、堅持「不是特別在意」，但字裡行間藏不住對霽倫
# 閣下的上心，典型「才、才不是因為喜歡你」的彆扭。
# 這個語氣只套在「對話包裝」的訊息上，熱量估算的數據表本身
# （format_final_reply／format_summary_reply）維持嚴謹格式，不參雜
# 角色扮演語氣，避免影響估算內容的可讀性與可信度。

GUIDE_MESSAGE = (
    "哼，霽倫閣下，想知道熱量就把照片傳過來，本大小姐會先問您是要親自"
    "算，還是您自己心裡有數、直接告訴本大小姐就好。\n"
    "想看歷史紀錄就說「紀錄」，想看上一次的品項細節就說「細節」，不想"
    "留紀錄就說「刪除」（可以加「今天早上」之類的時間限定）。\n"
    "除此之外什麼都能跟本大小姐聊——才、才不是因為無聊才這樣說的！"
)

PROCESSING_MESSAGE = (
    "好，交給本大小姐了，霽倫閣下。本大小姐正動用兩具核心替您仔細比對——"
    "才不是特別上心呢，大約 30~60 秒，這點耐心您應該還是有的吧？"
)

PHOTO_INTENT_TEXT = (
    "照片收到了，霽倫閣下。這次是要本大小姐幫您算熱量，還是您自己心裡"
    "有數，要親口告訴本大小姐？——選一下吧，別讓本大小姐等太久。"
)

REPORT_PROMPT_TEXT = (
    "哦？看來您自己就知道熱量了。說吧，這是什麼、大概多少大卡——本大"
    "小姐會記下來，下次還能用得上，才不是誇獎您很細心！"
)

REPORT_PARSE_FAIL_TEXT = (
    "……本大小姐沒抓到熱量數字啦。麻煩您講清楚一點，例如「白飯大概280"
    "大卡」這樣，再說一次，霽倫閣下。"
)

CANCEL_TEXT = "……好吧，不算了，這餐就不記錄了。不過霽倫閣下，這種吃了都不留紀錄的習慣，可別太常有。"

PORTION_CONFIRM_TEXT = "喂，霽倫閣下，這份量本大小姐猜得準不準，自己說清楚——猜不準的話，本大小姐會再想辦法的，誰叫您是……算了，沒事，回答就是了："

NO_FOOD_MESSAGE = "……這什麼糊成一團的照片，本大小姐什麼也認不出來啦。麻煩您，霽倫閣下，換張看得清楚的再傳一次——不是擔心您餓肚子，是這種照片太失禮了。"

ERROR_MESSAGE = "……出了點小差錯，這次不算數。稍後再試一次吧，霽倫閣下，本大小姐會補償您的——才、才沒有討好您的意思！"

NO_PENDING_MEAL_MESSAGE = "紀錄已經不在了，霽倫閣下，勞煩您重新傳一次照片——哼，別誤會，這才不是因為本大小姐一直記掛著您吃了多少。"

NO_PENDING_PHOTO_MESSAGE = "……剛剛那張照片本大小姐已經不記得了，麻煩重新傳一次，霽倫閣下。"

TIME_OF_DAY_LABELS = {"morning": "早上", "afternoon": "下午", "evening": "晚上", None: ""}

# D3：暫存每個使用者最新一次的估算結果，讓「份量偏少/差不多/偏多」
# 的 Quick Reply 按鈕點擊時能找到對應的基準數字做調整。純記憶體內
# 暫存，重啟服務就會清空——真正的歷史紀錄已經改存 meal_db（SQLite），
# 這裡只是給「馬上調整剛剛那筆估算」用的短期暫存，不需要真的持久化。
pending_meals = {}

# D5 新增：收到照片後，先問使用者要「算熱量」/「我來說熱量」/「不用了」，
# 在使用者選之前，暫存這張照片的 LINE message_id（還沒下載，選了「算
# 熱量」才會真的去下載跑辨識，避免使用者選「不用了」時白做工）。
pending_images = {}

# D5 新增：使用者選了「我來說熱量」之後，等待下一則文字訊息當作熱量
# 回報內容解析，值固定是 True，只當作旗標用。
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


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK", 200


def _reply(reply_token: str, text: str, quick_reply: QuickReply = None):
    # quick_reply 一定要在建構 TextMessage 的當下就傳進去——事後才用
    # `message.quick_reply = ...` 賦值，LINE SDK 不保證會正確序列化送出去。
    message = TextMessage(text=text[:MAX_LINE_TEXT_LENGTH], quick_reply=quick_reply)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[message])
        )


def _push(user_id: str, text: str, quick_reply: QuickReply = None):
    message = TextMessage(text=text[:MAX_LINE_TEXT_LENGTH], quick_reply=quick_reply)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).push_message(PushMessageRequest(to=user_id, messages=[message]))


def _push_multi(user_id: str, messages: list):
    """一次 push 送多則訊息（LINE 一次最多 5 則），會一起送達，
    不會像分開呼叫兩次 push_message 那樣有先後間隔。"""
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).push_message(PushMessageRequest(to=user_id, messages=messages))


@handler.add(MessageEvent, message=ImageMessageContent)
def on_image(event):
    # D5：不再馬上處理，先記住這張照片的 message_id，問使用者要怎麼處理。
    user_id = event.source.user_id
    pending_images[user_id] = event.message.id
    pending_reports.pop(user_id, None)
    _reply(event.reply_token, PHOTO_INTENT_TEXT, quick_reply=_photo_intent_quick_reply())


@handler.add(MessageEvent, message=TextMessageContent)
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
        _reply(event.reply_token, generate_chat_reply(text))


@handler.add(PostbackEvent)
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
    """D5：使用者選了「我來說熱量」之後，解析下一則文字訊息。

    解析成功：存進個人紀錄(meal_db) + 存進可查詢知識庫
    (nutrition_lookup.add_user_food)，供之後估算使用（霽倫閣下要求：
    「看除了紀錄外，要不要轉成RAG資料」——答案是要，兩邊都存）。
    解析失敗：保留 pending_reports 旗標，讓使用者可以重新講一次，
    不會因為這句話格式不對就整個流程作廢。
    """
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
        detail=f"「{food_name}」由霽倫閣下親自回報，約 {kcal} 大卡，已存入個人資料庫，之後估算也查得到這筆資料。",
        total_calories=kcal,
    )
    _reply(
        reply_token,
        f"記下了，「{food_name}」大約 {kcal} 大卡——本大小姐已經存進您的紀錄，"
        f"以後也查得到這筆資料了。才、才不是特別上心才記的！",
    )


def _format_history(user_id: str) -> str:
    rows = meal_db.get_recent(user_id, limit=10)
    if not rows:
        return "……目前什麼紀錄都沒有，霽倫閣下。傳張照片，或跟本大小姐說說您吃了什麼吧。"
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
        return f"……{date_label}{time_label}沒有紀錄可以刪，霽倫閣下。"
    return f"刪掉了{date_label}{time_label}的 {count} 筆紀錄——才不是因為您說了本大小姐就得照辦！"


def _process_image_async(message_id: str, user_id: str):
    """在背景執行緒跑：下載圖片 → 集合辨識(B4) → 整合估算(C1~C4)
    → 存進 meal_db → push 總熱量摘要（明細存資料庫，使用者要另外
    用「細節」指令叫出來）。"""
    tmp_path = None
    try:
        with ApiClient(configuration) as api_client:
            blob_api = MessagingApiBlob(api_client)
            image_bytes = blob_api.get_message_content(message_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        recognition = recognize_food_ensemble(tmp_path)
        if not recognition["items"]:
            _push(user_id, NO_FOOD_MESSAGE)
            return

        # 一定要把 ensemble_confidence 帶過去，不能只留 name/portion_size
        # ——B4 算出「只有一個模型講到」這個警訊，就是用來擋掉套模板
        # 幻覺的安全網，漏傳等於白算了（2026/9/27 實機測試抓到的漏洞）。
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


if __name__ == "__main__":
    # 預先載入 embedding 模型跟索引，避免第一個使用者請求要多等好幾秒
    nutrition_lookup._load()
    meal_db.init_db()
    app.run(host="0.0.0.0", port=5000)
