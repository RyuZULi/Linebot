# 拍照估算熱量 LINE 助手

RAG 課程期末專題。三個 LINE bot 共用一個 Flask process：

- **Belfast**（秘書）：拍照估算熱量、飲食紀錄、一般聊天 → `src/belfast_bot.py`
- **Ryuzu**（開發助手）：一般聊天、「任務：」自動開發、PDF 統整/RAG → `src/dev_bot.py`
- **CEC_API助手**：公司同仁查 CEC 建築 Revit API 的操作與錯誤 → `src/cec_bot.py`、`src/cec_rag.py`

入口是 `src/webhook_app.py`（port 5000），路徑分流 `/callback/belfast`、`/callback/dev`、`/callback/cec`。
回覆使用者的文字一律用**繁體中文**。

## 架構

熱量估算流程：`food_recognition_ensemble`（B5，minicpm-v + qwen3-vl + gemma3 三模型信心分數投票，
門檻 0.4，見 `data/vision_benchmark/B5_REPORT.md`；不要把門檻調低到 0.3，正確率會從 86% 掉到 56%）
→ `nutrition_lookup`（C1，TFDA → 使用者回報 → 台大自助餐表，逐層 fallback）
→ `portion_lookup`（C2，份量換公克；換算不了時用 `typical_portion` 典型便當份量，每個數字都有出處，見 `data/portion_reference/NOTES.md`）→ `calorie_estimator`（C3，信心分級）
→ `final_estimate`（C4，加上 Nutrition5k 視覺相似校準錨點）。

本地模型都走 Ollama（`localhost:11434`）；embedding 是 BAAI/bge-m3，
**只載一份**，其他模組透過 `nutrition_lookup._load()` / `_embed_model` 共用。

資料：SQLite（`data/line_bot.db`、`data/dev_tasks.db`、`data/pdf_docs/`）+ Chroma。
這些都是使用者資料，已在 `.gitignore`，不要提交。

## 核心設計原則（不要違反）

- **避免幻覺**：熱量數字必須來自真實資料庫，查不到就老實回「無法確定」，不要硬湊。
- **信心分級**：精確對應 / 相近估算 / 查無資料 / 疑似誤判。
  `calorie_estimator._estimate_item` 裡「共識分數未達門檻（低共識）→ 疑似誤判、不計入總熱量」
  這個判斷**必須放在最前面**，否則其他分支會先 return 讓它失效。
- Belfast 估算後同時列出「逐項加總」與「外觀參考」兩個數字，由使用者選或手動輸入，**選完才寫入紀錄**。
  LINE 快速回覆按鈕一有新訊息就消失：估算還沒確認時，回覆「細節」要重新附上確認按鈕，
  也接受直接打字（「逐項」「外觀」「不記錄」、「500」「炒麵 480 大卡」；問句不算）。
  被其他問題打斷時（聊天、統計、紀錄…），回答完主動提醒「剛才那一餐還沒記錄」並重新附上按鈕；
  按了「手動輸入」卻改問別的，也照常回答再提醒，不會卡在「看不懂熱量」。
  不要改回自動挑選或把兩個數字平均；`meal_records.raw_json` 存兩種估計值、逐項明細、照片路徑、
  使用者的選擇，是之後分析哪種估計比較準的資料。
- 數值估算用低溫度生成；數據表格式（`format_final_reply` 等）不參雜角色扮演語氣，
  人設只套在對話包裝文字。
- **PDF RAG 回答（`pdf_rag.answer_with_rag`）不可以套人設**，要用嚴格提示詞 + 溫度 0。
  實測加上 Ryuzu 人設後，文件沒有的題目 6/6 編造，還把編的數字掛上真實文件當出處。
- 相似度門檻都是用測試資料校準出來的，不是隨便定的：
  TFDA `SIMILAR_THRESHOLD=0.70`、台大表 `0.65`。改門檻前先看對應的 `*_REPORT.md`。
- C1 比對：查詢名稱沒有「乾」「粉」時，跳過名稱有這些字的候選（花椰菜乾 291 vs 新鮮約 30 kcal/100g）；
  TFDA 之後的來源一律「完全相同菜名」優先於「相近菜名」（避免「炸豬排」比對到使用者回報的整個「炸豬排便當」）。
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

## PDF 文字抽取（MinerU）

- MinerU 裝在**獨立 venv** `D:\venvs\mineru`（CUDA 版 torch，跟主環境的 CPU 版分開，不要裝進主環境）。
- `pdf_rag.extract_pages` 先用 MinerU（OCR + 表格），失敗才退回 pypdf；結果快取在 `*.pages.json`。
- **只能本地解析，不要加 `--remote`**（會把文件上傳到 mineru.net）。已關閉 MinerU 遙測。
- 本地解析要 `parse_server.local.mode=managed`（已設定），且 MinerU 背景服務要在跑
  （`extract_pages` 會自動 `mineru server start`）。
- 刪除文件時要 `mineru forget <path> --no-dry-run` 清掉 MinerU 自己的快取（`delete_doc` 已處理）。

## 已知限制

- 純照片頁面（例：食物照片）MinerU 也讀不出文字；直排表頭會被打亂（「蛋(公白克質)」）。
- PDF RAG 門檻 `RAG_THRESHOLD=0.55` / `0.40` 只用一份文件初步驗證過。
- 嚴格提示詞偏保守：偶爾會只列品項而沒回答數值（例：問低脂乳品熱量只回品項），寧可少答不亂答。
- 兩個 bot 共用一個 process，重啟會同時影響兩邊。
- cloudflared 用 quick tunnel，重啟後網址會變，要去 LINE Developers Console 重填 Webhook URL。

## CEC_API助手（`cec_rag.py`，規格：`data/CEC_Revit API/AIRAGUse.md`）

- 知識庫是公司內部資料（同仁姓名、內部 Notion），`data/CEC_Revit API/`、`data/cec_rag/` 都在 `.gitignore`，不要提交。
- 照規格書流程：非本庫範圍直接轉介 → 目錄比對（容錯梁/樑、驅/軀、版/板；模糊比對只容許同長度錯一個字）
  → 錯誤訊息字串比對（只比「錯誤訊息對照」與非按鈕名稱的「」）→ 向量檢索 → LLM。判斷與比對都用程式，不讓小模型猜。
- 存入與檢索都透過 LlamaIndex（`VectorStoreIndex` + `ChromaVectorStore` + `MetadataFilters`，每張卡片是一份來源文件，更新時 `delete(ref_doc_id=檔名)`）；通用問題指定段落時用 `get_nodes` 依條件取出，不經相似度排序。
- 回答模型用 `qwen3:8b`（`think: false`）。實測 12 個情境：llama3.2 有 4 題把資料裡有的答案回成「沒有資料」，不要換回去。
- 提示詞的規則放在參考資料**後面**：放前面時 qwen3 會把相鄰兩條拼湊成假建議（「不支援斜板」+ 下一行
  「Deck 樓板請改用…」→ 回答「斜板請改用 Deck」）。網址由程式附上，模型寫的網址行會被刪掉。
- 通用問題依關鍵字直接指定 `_通用問題` 段落（`GENERAL_SECTION_RULES`），不靠向量檢索排序。
- 不套任何角色人設（同 PDF RAG 的理由）。問答記錄在 `data/cec_rag/qa_log.jsonl`，可用來找出同仁常問但資料沒有的題目。
- 記得每位同仁目前在聊哪個按鈕（1 天沒說話就忘掉，或說「新問題」重設；原本 15 分鐘太短，使用者要求延長，不設無限期以免隔天誤判追問）：沒提到按鈕名稱的追問延續同一個話題，
  並附上前 2 輪問答讓模型理解「清單」這類指代。實測原本每句都當新問題，追問會被回問「是哪個功能」。
  換話題（提到別的按鈕、選了別的按鈕、錯誤訊息屬於別的功能）時，先清掉舊話題的問答紀錄再回答。
- 「常見問題」「錯誤訊息對照」切成一條一塊（其他段落仍是一段一塊）；錯誤訊息命中時只給命中的那一條。
  整段一塊時，同一個按鈕的不同問題檢索到同一大段，回答都一樣。
- 籠統的問題回報（含「壞掉／不能用／失敗…」且跟該卡片每條常見問題/錯誤訊息標題的相似度都 < 0.80）
  → 先回問「發生什麼狀況」，選項用卡片自己的常見問題 + 沒有訊息原文的現象描述，另有「貼錯誤訊息」「其他狀況」。
  同一個按鈕只回問一次；點選項後只拿那一條回答。門檻 0.80 是實測校準：具體描述 0.77~0.98、籠統說法 ≤0.62，
  「切割樓板壞掉了」0.773 是靠「含籠統詞」這個條件擋下的，兩個條件不要拆開。「怎麼用」「按不了/灰色」不回問。
- **改比對規則、門檻、分流順序後一定要跑 `python src/cec_route_test.py`**（分流回歸測試，LLM 換成假的，約 20 秒，
  報告寫到 `data/cec_rag/route_test_result.txt`）。新發現的錯誤案例請加進 `CASES`。
- 錯誤訊息比對排在目錄比對**前面**（很多訊息本身含按鈕名稱）；命中錯誤訊息、或使用者點了回問選項時，
  那一條直接照原文輸出、**不經過 LLM**（不會編造、不會多補延伸說明）。LLM 只用在需要綜合多段的情況。
- 正在聊某個按鈕時，句子提到「機電／土木／雲端」先當追問、不轉介；只有明確問別的產品
  （`EXPLICIT_OTHER_PRODUCT`：「機電的 API／按鈕／功能」）或帳號登入（`ACCOUNT_WORDS`）才轉介。
  有話題時給 LLM 的聯絡人只留該按鈕所屬類別（目前全是建築 API → 建築 API 負責人）；通用問題（授權等）不限縮。
- 非本庫範圍：句子有 CEC／註冊／密鑰／本機ID／按鈕時不轉介 Autodesk（「CEC API 授權過期」由通用問題回答）；
  轉介詞本身也出現在本庫按鈕名稱時（「套管」），轉介之外一併列出本庫的按鈕讓使用者選。
- 按鈕名稱／別名尾巴是「檢查、建立、設定、計算、匯出」時，去掉後的核心詞（≥4 字）也算命中（「樓梯干涉」）。
  核心詞跟別的按鈕撞名（「磁磚數量」）會變成回問，這是預期行為。
- **設計上只給一對一聊天用**（群組裡互相問，對話紀錄太亂）。2026/10/7 拉兩位同事進群組只是為了測試，
  發現 B 按了 A 收到的選項，用 B 的 user_id 找不到紀錄 →「選項已過期」。因此對話狀態改以**聊天室**為單位
  （`cec_bot._chat_key`：群組用 group_id、聊天室用 room_id、一對一用 user_id），但不要再為群組情境加功能。
- 回問訊息本文列出編號選項；輸入編號或照打選項原文就當成點選（`_typed_choice`，實測同仁會照清單打字）。
  回問狀況的選項答完一個後仍保留，換話題或 1 天後才清。
  問題幾乎等於某張卡片的常見問題原文（相似度 ≥0.90）時不轉介（「鋼構、機電、連結檔有算嗎？」含「機電」）。
- 回答後附 👍／👎，記在 qa_log（`feedback` 欄位），每筆回答也記用到的段落與分數（`hits`）。
- 已知限制：偶爾會在正確答案後面多補幾點延伸說明，少數解法是模型自己補的。
  8B 模型偶爾把按鈕名稱寫錯一個字（例：干涉→干擾，2026/10/7 看到一次）；頻率低、內容與說明頁連結正確，
  **暫不加後處理**，等 qa_log 累積後再評估。真的要修，用「回答中的片段跟本次用到的按鈕名稱同長度、只差一個字」
  這條通用規則，不要逐詞加正則（使用者明確反對越修越臃腫）。
  0.80 回問門檻是小樣本校準（反例 0.773 只差 0.027，靠「含籠統詞」條件撐住）。
  `_sessions` 沒有加鎖，同一個人極短時間連發可能互相覆蓋，目前規模碰不到。
