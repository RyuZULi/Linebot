# 拍照估算熱量 LINE 助手

RAG 課程期末專題。兩個 LINE bot 共用一個 Flask process：

- **Belfast**（秘書）：拍照估算熱量、飲食紀錄、一般聊天 → `src/belfast_bot.py`
- **Ryuzu**（開發助手）：一般聊天、「任務：」自動開發、PDF 統整/RAG → `src/dev_bot.py`

入口是 `src/webhook_app.py`（port 5000），路徑分流 `/callback/belfast`、`/callback/dev`。
回覆使用者的文字一律用**繁體中文**。

## 架構

熱量估算流程：`food_recognition_ensemble`（B4，minicpm-v + qwen3-vl 雙模型投票）
→ `nutrition_lookup`（C1，TFDA → 使用者回報 → 台大自助餐表，逐層 fallback）
→ `portion_lookup`（C2，份量換公克）→ `calorie_estimator`（C3，信心分級）
→ `final_estimate`（C4，加上 Nutrition5k 視覺相似校準錨點）。

本地模型都走 Ollama（`localhost:11434`）；embedding 是 BAAI/bge-m3，
**只載一份**，其他模組透過 `nutrition_lookup._load()` / `_embed_model` 共用。

資料：SQLite（`data/line_bot.db`、`data/dev_tasks.db`、`data/pdf_docs/`）+ Chroma。
這些都是使用者資料，已在 `.gitignore`，不要提交。

## 核心設計原則（不要違反）

- **避免幻覺**：熱量數字必須來自真實資料庫，查不到就老實回「無法確定」，不要硬湊。
- **信心分級**：精確對應 / 相近估算 / 查無資料 / 疑似誤判。
  `calorie_estimator._estimate_item` 裡「單一模型才講到的品項 → 疑似誤判、不計入總熱量」
  這個判斷**必須放在最前面**，否則其他分支會先 return 讓它失效。
- 數值估算用低溫度生成；數據表格式（`format_final_reply` 等）不參雜角色扮演語氣，
  人設只套在對話包裝文字。
- 相似度門檻都是用測試資料校準出來的，不是隨便定的：
  TFDA `SIMILAR_THRESHOLD=0.70`、台大表 `0.65`。改門檻前先看對應的 `*_REPORT.md`。
- 「土豆」在 TFDA 被列為花生俗名，會把馬鈴薯算成 7 倍熱量 → `AMBIGUOUS_ALIASES` 黑名單。
  曾試過用語意分數通用化取代黑名單，失敗（好壞案例只差 0.013），不要再走這條路。

## Windows 環境注意事項

- 終端機是 cp950，印中文會亂碼或 `UnicodeEncodeError`：
  要檢查中文輸出時寫到 UTF-8 檔案再讀，不要直接 print。
- subprocess 讀外部程式輸出時一定要 `encoding="utf-8"`。
- `claude` 是 npm 裝的 `claude.cmd`：subprocess 要用 `["cmd", "/c", "claude", ...]`，
  prompt 用 `input=` 從 stdin 傳（放在命令列參數會被 cmd.exe 在換行處截斷）。
- cloudflared 要指向 `http://127.0.0.1:5000`，不要用 `localhost`（會解析成 IPv6 連不上）。

## 任務自動化（`src/task_runner.py`）

Ryuzu 收到「任務：...」→ 開 `worktrees/task-N` 隔離分支 → 無頭呼叫 claude
（`--permission-mode auto --permission-prompts none`，**不要改成
`--dangerously-skip-permissions`**）→ 語法健檢 → 回報摘要 → 主人在 LINE 按「套用」才合併。
合併後**不會自動重啟**，要手動重啟 `webhook_app.py`（刻意保留的人工關卡）。

如果你是被任務流程叫起來的 session：
- 只在目前這個 worktree 裡工作，不要動 `D:\CalorieCalculation\data` 的正式資料，
  測試請改用暫存路徑，測完清掉。
- 不需要自己 git commit，外面的流程會處理。
- 專案沒有自動化測試套件，驗證方式是寫小腳本直接呼叫函式、用真實或暫存資料跑。

## 已知限制

- PDF 只能抽文字層；掃描檔/圖片表格抽不到（例：`source_manual_2023.pdf` 16 頁只抽得出約 2300 字），沒有 OCR。
- PDF RAG 門檻 `RAG_THRESHOLD=0.55` / `0.40` 只用一份文件初步驗證過。
- 兩個 bot 共用一個 process，重啟會同時影響兩邊。
- cloudflared 用 quick tunnel，重啟後網址會變，要去 LINE Developers Console 重填 Webhook URL。
