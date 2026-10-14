"""
背景常駐：啟動 LINE bot 伺服器 + cloudflared，掛掉自動重啟，並自動更新 LINE 的 Webhook URL。

用 pythonw 執行就不會有視窗（install_autostart.bat 會把它設成登入 Windows 時自動啟動）。
紀錄寫在 logs/：
  logs/run_bots.log    這支程式自己的紀錄（啟動、重啟、Webhook 更新結果）
  logs/webhook_app.log bot 伺服器的輸出
  logs/cloudflared.log tunnel 的輸出

為什麼要自動更新 Webhook URL：cloudflared 的 quick tunnel 每次啟動網址都會變，
以前要手動到 LINE Developers Console 重填三個 channel。現在拿到新網址後，
用 LINE 官方 API（set_webhook_endpoint）自動設定，再用 test_webhook_endpoint 確認 LINE 打得進來。
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
LOG_DIR = os.path.join(ROOT, "logs")
PORT = 5000
LOCK_PORT = 5099  # 只是用來確保同時只有一份 run_bots 在跑
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"
TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

# channel → (.env 裡的 access token 名稱, webhook 路徑)
CHANNELS = {
    "Belfast": ("BELFAST_LINE_CHANNEL_ACCESS_TOKEN", "/callback/belfast"),
    "Ryuzu": ("DEV_LINE_CHANNEL_ACCESS_TOKEN", "/callback/dev"),
    "CEC_API助手": ("CEC_LINE_CHANNEL_ACCESS_TOKEN", "/callback/cec"),
}

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe")  # 子程式用 python.exe（不是 pythonw）
RESTART_DELAY = 10
HEALTH_TIMEOUT = 300  # 啟動時要載入向量模型和索引，可能要一兩分鐘


def log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "run_bots.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")


def load_env() -> dict:
    env = {}
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"  # 終端機是 cp950，輸出導到檔案時中文要用 UTF-8
    env["PYTHONUTF8"] = "1"
    return env


# ---------- bot 伺服器 ----------

def start_server() -> subprocess.Popen:
    out = open(os.path.join(LOG_DIR, "webhook_app.log"), "a", encoding="utf-8")
    out.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 啟動 =====\n")
    out.flush()
    return subprocess.Popen([PYTHON, "webhook_app.py"], cwd=SRC, stdout=out, stderr=subprocess.STDOUT,
                            env=child_env(), creationflags=NO_WINDOW)


def wait_healthy(proc: subprocess.Popen) -> bool:
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ---------- cloudflared ----------

class Tunnel:
    """跑 cloudflared quick tunnel，從輸出裡抓出這次的網址。"""

    def __init__(self):
        self.url = None
        self.proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
            text=True, encoding="utf-8", errors="replace",
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        with open(os.path.join(LOG_DIR, "cloudflared.log"), "a", encoding="utf-8") as out:
            out.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 啟動 =====\n")
            for line in self.proc.stdout:
                out.write(line)
                out.flush()
                m = TUNNEL_URL_PATTERN.search(line)
                if m and not self.url:
                    self.url = m.group(0)

    def wait_url(self, timeout: int = 60):
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            if self.url:
                return self.url
            time.sleep(1)
        return None


# ---------- LINE Webhook URL ----------

PUBLIC_READY_TIMEOUT = 120
WEBHOOK_RETRIES = 5
WEBHOOK_RETRY_DELAY = 15


def wait_public(base_url: str) -> bool:
    """等新網址從外面真的連得到再設定 Webhook。
    實測 cloudflared 印出網址後 1 秒就設定，LINE 會回「Invalid webhook endpoint URL」
    （新網址的 DNS 還沒生效，LINE 驗證時連不到）。"""
    deadline = time.time() + PUBLIC_READY_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _set_one(token: str, endpoint: str) -> str:
    from linebot.v3.messaging import (ApiClient, Configuration, MessagingApi,
                                      SetWebhookEndpointRequest, TestWebhookEndpointRequest)
    with ApiClient(Configuration(access_token=token)) as client:
        api = MessagingApi(client)
        api.set_webhook_endpoint(SetWebhookEndpointRequest(endpoint=endpoint))
        result = api.test_webhook_endpoint(TestWebhookEndpointRequest(endpoint=endpoint))
    if not result.success:
        raise RuntimeError(f"LINE 測試連線失敗（{result.status_code} {result.reason}）")
    return "OK"


def _short_error(e: Exception) -> str:
    body = getattr(e, "body", None)  # LINE SDK 的錯誤：只留回應內容，不要整串 HTTP 標頭
    return f"{getattr(e, 'status', '')} {body}".strip() if body else str(e)


def update_webhooks(base_url: str):
    if not wait_public(base_url):
        log(f"[警告] {PUBLIC_READY_TIMEOUT} 秒內從外部連不到 {base_url}，仍嘗試設定 Webhook")
    env = load_env()
    for name, (token_key, path) in CHANNELS.items():
        token = env.get(token_key)
        if not token:
            log(f"[略過] {name}：.env 沒有 {token_key}")
            continue
        endpoint = base_url + path
        for attempt in range(1, WEBHOOK_RETRIES + 1):
            try:
                log(f"[Webhook] {name} → {endpoint}  {_set_one(token, endpoint)}")
                break
            except Exception as e:
                if attempt == WEBHOOK_RETRIES:
                    log(f"[Webhook 失敗] {name}：{_short_error(e)}（請到 LINE Developers Console 手動設定）")
                else:
                    log(f"[Webhook 重試 {attempt}] {name}：{_short_error(e)}")
                    time.sleep(WEBHOOK_RETRY_DELAY)


# ---------- 主迴圈 ----------

def acquire_lock():
    """綁一個本機 port 當鎖：已經有一份在跑就直接結束，避免開兩份搶 port 5000。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        return None
    s.listen(1)
    return s


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    lock = acquire_lock()
    if lock is None:
        log("已經有一份 run_bots 在執行，這份直接結束。")
        return
    log("===== run_bots 啟動 =====")

    server, tunnel = None, None
    while True:
        if server is None or server.poll() is not None:
            if server is not None:
                log(f"bot 伺服器結束了（代碼 {server.returncode}），{RESTART_DELAY} 秒後重啟")
                time.sleep(RESTART_DELAY)
            server = start_server()
            log("bot 伺服器啟動中，等待載入模型…")
            if wait_healthy(server):
                log("bot 伺服器就緒")
            else:
                log("bot 伺服器沒有在時限內就緒，詳見 logs/webhook_app.log")
                continue

        if tunnel is None or tunnel.proc.poll() is not None:
            if tunnel is not None:
                log(f"cloudflared 結束了，{RESTART_DELAY} 秒後重開（網址會變，會重新設定 Webhook）")
                time.sleep(RESTART_DELAY)
            tunnel = Tunnel()
            url = tunnel.wait_url()
            if url:
                log(f"tunnel 網址：{url}")
                update_webhooks(url)
            else:
                log("沒拿到 tunnel 網址，詳見 logs/cloudflared.log")
                tunnel.proc.kill()

        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"run_bots 意外結束：{e!r}")
        raise
