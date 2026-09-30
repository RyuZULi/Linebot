"""
D2/D5：LINE webhook 共用入口。

2026/9/27 拆成兩個獨立的 LINE bot channel（Ryuzu 開發助手 / Belfast
秘書），但共用同一個 Flask process、同一個 port、同一條 cloudflared
tunnel——用不同的 URL 路徑區分兩個 channel 的 webhook：
  /callback/dev      → Ryuzu（開發助手：聊天 + 任務指令）
  /callback/belfast  → Belfast（秘書：熱量估算 + 飲食紀錄 + 聊天）

在 LINE Developers Console 裡，兩個 channel 的 Webhook URL 要分別設成
這兩個路徑（同一個 tunnel 網域，路徑不同），不能兩個都設成同一個
「/callback」，不然 Flask 沒辦法判斷這個請求是哪個 channel 送來的。

實際的事件處理邏輯都寫在 belfast_bot.py / dev_bot.py，這裡只負責
建立各自的 WebhookHandler/Configuration，掛上對應的路徑。
"""

import os

from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少 {name}，請確認 .env 有正確設定")
    return value


DEV_TOKEN = _require_env("DEV_LINE_CHANNEL_ACCESS_TOKEN")
DEV_SECRET = _require_env("DEV_LINE_CHANNEL_SECRET")
BELFAST_TOKEN = _require_env("BELFAST_LINE_CHANNEL_ACCESS_TOKEN")
BELFAST_SECRET = _require_env("BELFAST_LINE_CHANNEL_SECRET")

dev_configuration = Configuration(access_token=DEV_TOKEN)
dev_handler = WebhookHandler(DEV_SECRET)

belfast_configuration = Configuration(access_token=BELFAST_TOKEN)
belfast_handler = WebhookHandler(BELFAST_SECRET)

app = Flask(__name__)

import dev_bot
import belfast_bot

dev_bot.register(dev_handler, dev_configuration)
belfast_bot.register(belfast_handler, belfast_configuration)


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/callback/dev", methods=["POST"])
def callback_dev():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        dev_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK", 200


@app.route("/callback/belfast", methods=["POST"])
def callback_belfast():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        belfast_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK", 200


if __name__ == "__main__":
    import meal_db
    import nutrition_lookup
    import pdf_rag
    import task_db

    # 預先載入 embedding 模型跟索引，避免第一個使用者請求要多等好幾秒
    nutrition_lookup._load()
    meal_db.init_db()
    task_db.init_db()
    pdf_rag.init_db()
    app.run(host="0.0.0.0", port=5000)
