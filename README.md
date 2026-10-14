# 拍照估算熱量 LINE 助手

RAG 課程期末專題。三個 LINE bot 共用一個 Flask 程式（`src/webhook_app.py`，port 5000）：

| Bot | 做什麼 | Webhook 路徑 |
|---|---|---|
| **Belfast**（秘書） | 拍照估算熱量、飲食紀錄、一般聊天 | `/callback/belfast` |
| **Ryuzu**（開發助手） | 一般聊天、「任務：」自動開發、PDF 統整／RAG、YouTube 影片統整 | `/callback/dev` |
| **CEC_API助手** | 公司同仁查 CEC 建築 Revit API 的操作與錯誤 | `/callback/cec` |

所有模型都在本機跑（Ollama + HuggingFace），不呼叫雲端 LLM API。
設計細節、踩過的坑、不要改的地方見 [`CLAUDE.md`](CLAUDE.md)。

---

## 環境需求

- Windows（指令以 Windows 為準）
- Python 3.11
- [Ollama](https://ollama.com)（本機跑 LLM／視覺模型，需要啟動中）
- 磁碟空間：模型合計約 30GB。有 NVIDIA 顯示卡 Ollama 會快很多（拍照估算要同時跑三個視覺模型）
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)：把本機 port 5000 開給 LINE
- （選用）MinerU：PDF 的 OCR 與表格辨識，沒有裝會自動退回 pypdf，見下方說明

---

## 安裝

### 1. Python 套件

```bat
pip install -r requirements.txt
```

### 2. 模型

模型檔很大，**不進 git**。需要哪些模型記在 [`models.txt`](models.txt)，照清單安裝：

```bat
install_models.bat
```

直接在檔案總管雙擊也可以。它會：

1. 檢查 Ollama 有沒有安裝、有沒有啟動
2. 照 `models.txt` 逐一安裝：`ollama` 開頭的用 `ollama pull`，`hf` 開頭的從 HuggingFace 下載到
   `%USERPROFILE%\.cache\huggingface`
3. **已經裝好的模型會跳過，不會自動更新**
4. 最後列出沒裝好的模型（網路或磁碟空間問題），再執行一次即可補裝

其他用法：

| 指令 | 作用 |
|---|---|
| `install_models.bat --check` | 只檢查，列出哪些已安裝、哪些缺少，不下載 |
| `install_models.bat --update` | 已經裝好的也重新拉最新版 |

> **為什麼預設不更新？** 熱量估算的相似度門檻、三模型投票門檻、CEC 回答品質，都是用目前的模型版本實測
> 校準的（見 `data/*/*_REPORT.md`）。模型換版本，結果可能跟著改變，所以更新要明確加 `--update`，
> 更新後建議重跑 `python src/cec_route_test.py` 和熱量的評估腳本確認。

**程式改用別的模型時**，記得同步修改 `models.txt`，下一台電腦才裝得到。`models.txt` 格式：

```
# 註解
ollama gemma3:12b
hf     BAAI/bge-m3
```

### 3. LINE 金鑰（`.env`）

```bat
copy .env.example .env
```

打開 `.env`，填入三個 LINE channel 的 Channel access token 與 Channel secret
（[LINE Developers Console](https://developers.line.biz/console/) → 各 channel → Messaging API）。
`.env` 不會進 git。

### 4. 不在 git 裡的資料

下列資料是使用者資料或公司內部資料，已在 `.gitignore`，換電腦要另外複製：

| 路徑 | 內容 | 沒有的話 |
|---|---|---|
| `data/CEC_Revit API/` | CEC_API助手的知識庫（按鈕說明卡片、聯絡人表） | CEC_API助手無法回答 |
| `data/line_bot.db`、`data/meal_photos/` | 飲食紀錄與照片 | 啟動時自動建立空的資料庫 |
| `data/dev_tasks.db`、`data/pdf_docs/` | Ryuzu 的任務與 PDF | 啟動時自動建立 |
| `data/cec_rag/` | CEC 的向量資料庫與問答紀錄 | 啟動時從知識庫自動重建 |

### 5.（選用）MinerU

PDF 統整預設用 MinerU 做 OCR 和表格辨識，失敗才退回 pypdf（掃描檔會抽不到字）。
MinerU 要裝在**獨立的 venv**（`D:\venvs\mineru`，CUDA 版 torch），不要裝進主環境；
**只能本地解析，不要加 `--remote`**（會把文件上傳到外部伺服器）。細節見 `CLAUDE.md`。

---

## 啟動

### 常駐執行（建議）

雙擊 **`install_autostart.bat`**：設定成登入 Windows 時自動在背景啟動，並立刻啟動一次。不需要開著任何視窗。

背景程式（`run_bots.py`）會：
1. 啟動 bot 伺服器，等模型載入完成（約 15～60 秒）
2. 啟動 cloudflared quick tunnel，取得這次的 `https://xxxx.trycloudflare.com` 網址
3. **自動把三個 LINE channel 的 Webhook URL 設成新網址**（LINE 官方 API），並請 LINE 實際測試連線
4. 伺服器或 cloudflared 掛掉時自動重啟；cloudflared 重開網址會變，會再自動更新 Webhook

| 檔案 | 作用 |
|---|---|
| `install_autostart.bat` | 設定開機自動啟動並立刻啟動；`install_autostart.bat --remove` 取消自動啟動 |
| `status_bots.bat` | 查看是否在執行，以及最近的啟動／Webhook 設定紀錄 |
| `stop_bots.bat` | 停止（伺服器、cloudflared 一起停；不會取消自動啟動） |

紀錄在 `logs/`：`run_bots.log`（啟動、重啟、Webhook 結果）、`webhook_app.log`（bot 輸出）、`cloudflared.log`。

**改了程式要套用時**：執行 `stop_bots.bat` 再執行 `install_autostart.bat`；或只結束 bot 伺服器的 python
程序，背景程式會在約 20 秒內用新程式碼自動重啟（tunnel 網址不變）。

注意：
- 只在**登入 Windows 時**執行；電腦睡眠或關機時 bot 就停了，可以到「電源設定」關掉自動睡眠。
- Ollama 也要在執行（安裝後預設會開機自動啟動）。
- quick tunnel 是 Cloudflare 的免費測試用服務，沒有穩定性保證；要固定網址需要自己的網域與 Cloudflare 帳號。

### 手動執行（除錯用）

```bat
:: 1. 啟動 bot（三個一起）；先執行 stop_bots.bat，避免跟背景那份搶 port 5000
cd src
python webhook_app.py

:: 2. 另開一個視窗，把 port 5000 開給 LINE（要用 127.0.0.1，不要用 localhost）
cloudflared tunnel --url http://127.0.0.1:5000
```

手動執行時，要自己到 LINE Developers Console 把三個 channel 的 Webhook URL 設成
`<網址>/callback/belfast`、`<網址>/callback/dev`、`<網址>/callback/cec`。

三個 bot 在同一個程式裡，**重啟會同時影響三個**。

---

## 使用方式

**Belfast**
- 傳食物照片 → 選「算熱量」→ 約一分鐘後回覆「逐項加總」與「外觀參考」兩種估計，選一個或手動輸入才會記錄
- 「細節」看每一項怎麼算的、「紀錄」看最近的紀錄、「這週平均每天攝取多少」、「刪除今天早上」

**Ryuzu**（任務、PDF、YouTube 只有 owner 能用；第一個傳訊息的人自動成為 owner）
- `任務：<要做的事>` → 在隔離分支自動開發，完成後回報，按「套用」才合併（合併後要手動重啟）
- 傳 PDF → 統整後詢問要不要加入資料庫；之後聊天問到相關內容會自動查
- 貼 YouTube 連結 → 抓字幕統整成繁體中文
- 「任務清單」「PDF 清單」「刪除 PDF #3」

**CEC_API助手**（設計上一對一使用，不要拉進群組）
- 直接問按鈕怎麼用、貼上錯誤訊息、或說「XX 壞掉了」（會先問清楚狀況）
- 「新問題」重新開始

---

## 測試

專案沒有完整的自動化測試，CEC_API助手的分流有回歸測試（約 20 秒，不呼叫 LLM）：

```bat
cd src
python cec_route_test.py
```

改了 CEC 的比對規則、門檻或分流順序後一定要跑。
