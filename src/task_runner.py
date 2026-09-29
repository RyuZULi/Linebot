"""
開發任務的核心自動化邏輯。

流程（呼應霽倫閣下的要求：「不能在其他地方測試，然後確認沒問題再併
進來」）：
  1. 收到「任務：...」指令 → 在獨立的 git worktree/分支裡工作，
     不會碰到正式在跑的服務。
  2. 非互動呼叫本機安裝的 claude CLI 在那個 worktree 裡實際修改
     程式碼——這裡刻意不用 --dangerously-skip-permissions（那個模式
     完全跳過權限檢查，等於給予系統層級的完整存取，之前用這個做法
     連 import 測試都被 Claude Code 自己的 Auto Mode 安全分類器判定
     成「Create Unsafe Agents」擋下來）。改用 --permission-mode auto
     （跟這個對話本身一樣，讓分類器逐一判斷每個工具呼叫安不安全，
     安全的常規檔案編輯/指令會自動放行，真的踩到危險或超出專案範圍
     的操作會被擋下來，不是整批開後門）+ --permission-prompts none
     （真的遇到分類器判斷不了、需要人決定的情況，直接視為拒絕，不會
     卡住等一個沒有人會回答的確認框）。這樣任務失敗的方式是「安全地
     卡住、回報做不到」，不是「悄悄拿到系統層級權限」。
  3. 跑完做基本健檢（語法檢查），健檢通過才把摘要回報給主人。
  4. 一定要主人在 LINE 上按「套用」才會真的合併回 master、重啟服務
     ——這是刻意設計成兩段式的，Claude 自動跑完不代表自動生效，
     降低誤判/幻覺程式碼直接上線的風險。
"""

import ast
import json
import os
import subprocess
import threading

import task_db

REPO_ROOT = r"D:\CalorieCalculation"
WORKTREE_ROOT = os.path.join(REPO_ROOT, "worktrees")
CLAUDE_TIMEOUT_SECONDS = 1200  # 20 分鐘，coding 任務可能要跑一陣子
MAIN_BRANCH = "master"


def _run_git(args: list, cwd: str = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def _sanity_check(worktree_path: str) -> tuple:
    """只做語法檢查（ast.parse 能不能解析 src/*.py），不是完整測試套件
    ——這個專案目前沒有自動化測試，之前每個功能都是手動用 python -c
    直接呼叫函式驗證（見對話紀錄）。這裡至少能擋掉「改完程式碼直接
    壞掉」這種最低級的錯誤，比完全不檢查安全，但不保證邏輯正確。"""
    src_dir = os.path.join(worktree_path, "src")
    if not os.path.isdir(src_dir):
        return True, "（沒有 src 目錄可以檢查，略過語法檢查）"

    problems = []
    for fname in os.listdir(src_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(src_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                ast.parse(f.read())
        except SyntaxError as e:
            problems.append(f"{fname}：{e}")

    if problems:
        return False, "語法檢查失敗：\n" + "\n".join(problems)
    return True, "語法檢查通過（僅確認 src/*.py 能被解析，不是完整測試，邏輯是否正確仍需主人親自確認摘要）。"


def _extract_claude_summary(stdout: str) -> str:
    try:
        data = json.loads(stdout)
        return (data.get("result") or "").strip()[:1500] or "（沒有文字摘要）"
    except Exception:
        return stdout.strip()[:1500]


def run_task_async(task_id: int, description: str, on_done) -> None:
    """在背景執行緒跑。on_done(task_id) 會在狀態變成
    awaiting_confirmation 或 failed 之後被呼叫，用來 push LINE 訊息
    通知主人（由呼叫端傳進來，task_runner 本身不需要知道怎麼發 LINE
    訊息，保持職責分離）。"""
    threading.Thread(target=_run_task, args=(task_id, description, on_done), daemon=True).start()


def _run_task(task_id: int, description: str, on_done) -> None:
    branch_name = f"task-{task_id}"
    worktree_path = os.path.join(WORKTREE_ROOT, branch_name)

    task_db.update_task(task_id, status="running")

    add_result = _run_git(["worktree", "add", worktree_path, "-b", branch_name, MAIN_BRANCH])
    if add_result.returncode != 0:
        task_db.update_task(task_id, status="failed", result_summary=f"建立隔離分支失敗：\n{add_result.stderr}")
        on_done(task_id)
        return

    task_db.update_task(task_id, branch_name=branch_name, worktree_path=worktree_path)

    prompt = (
        "這是一個 LINE bot 專案（拍照估算熱量 + 開發助手），你現在在一個獨立的 "
        "git worktree/分支裡工作，不會影響正式運作中的服務，可以放心修改、"
        "跑測試指令，但請把所有操作限制在這個資料夾以內。請完成以下任務：\n\n"
        f"{description}\n\n"
        "完成後請確認程式碼至少能正常 import、沒有明顯語法錯誤，"
        "但不需要自己下 git commit，外面的流程會處理。"
    )

    try:
        # Windows 上 npm 裝的 claude 是 claude.cmd（批次檔包裝），不是
        # 真的 .exe，shell=False 時 CreateProcess 沒辦法直接執行它。
        # 用明確的 ["cmd", "/c", "claude", ...] 讓 cmd.exe 當執行檔、
        # 自己解析剩下的參數——2026/9/28 在不受這個 Claude Code
        # session 的 Auto Mode 分類器限制的終端機上實測確認這個寫法
        # 可行，returncode 0、且真的執行了任務內容。也一定要指定
        # encoding="utf-8"：Windows 預設用系統的 cp950（繁體中文）去
        # 解碼 stdout，但 claude 輸出的 JSON 是 UTF-8，混用會在讀取
        # 輸出的背景執行緒裡丟 UnicodeDecodeError。
        #
        # 2026/9/29 修正：prompt 不能放進命令列參數——cmd.exe 解析
        # 命令列時遇到換行字元就會把後面的內容切斷，任務描述裡只要有
        # 一個 \n（幾乎一定會有，因為 prompt 本身就用 \n\n 分段）就會
        # 被截斷成不完整的任務內容（實機測試 #5 抓到：claude 收到的
        # prompt 真的斷在第一個換行前面）。改成用 stdin 傳遞——
        # `claude -p` 不帶 prompt 參數時會從 stdin 讀取，不受命令列
        # 長度/換行限制。
        claude_result = subprocess.run(
            [
                "cmd", "/c", "claude", "-p",
                "--permission-mode", "auto",
                "--permission-prompts", "none",
                "--output-format", "json",
            ],
            input=prompt,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CLAUDE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        task_db.update_task(
            task_id, status="failed",
            result_summary=f"Claude 執行超過 {CLAUDE_TIMEOUT_SECONDS // 60} 分鐘，已中止，"
            f"分支保留供之後檢查（{worktree_path}）。",
        )
        on_done(task_id)
        return
    except FileNotFoundError:
        task_db.update_task(task_id, status="failed", result_summary="找不到 claude 指令，請確認 Claude Code CLI 已安裝並在 PATH 上。")
        on_done(task_id)
        return

    if claude_result.returncode != 0:
        task_db.update_task(task_id, status="failed", result_summary=f"Claude 執行失敗：\n{claude_result.stderr[:1500]}")
        on_done(task_id)
        return

    claude_summary = _extract_claude_summary(claude_result.stdout)

    diff_stat = _run_git(["diff", "--stat", MAIN_BRANCH], cwd=worktree_path).stdout.strip()
    if not diff_stat:
        task_db.update_task(
            task_id, status="failed",
            result_summary=f"Claude 說完成了，但沒有偵測到任何檔案變更（也可能是部分操作被權限分類器擋下，任務沒能真正完成）。\n\nClaude 的回覆：\n{claude_summary}",
        )
        on_done(task_id)
        return

    # 幫忙把變更 commit 起來，不依賴 Claude 自己記得 commit（更可靠，
    # 就算 Claude 已經自己 commit 過，這裡的 add+commit 頂多是
    # 「nothing to commit」，不影響分支上已經有的變更）。
    _run_git(["add", "-A"], cwd=worktree_path)
    _run_git(["commit", "-m", f"task-{task_id}: {description}"], cwd=worktree_path)

    ok, check_note = _sanity_check(worktree_path)

    summary = (
        f"【任務 #{task_id}】{description}\n\n"
        f"Claude 的回覆：\n{claude_summary}\n\n"
        f"變更檔案：\n{diff_stat}\n\n"
        f"{check_note}"
    )

    task_db.update_task(task_id, status="awaiting_confirmation" if ok else "failed", result_summary=summary)
    on_done(task_id)


def apply_task(task_id: int) -> tuple:
    """主人在 LINE 上按「套用」之後呼叫：合併回 master、清掉 worktree。
    回傳 (是否成功, 訊息)。是否重啟服務由呼叫端決定。"""
    task = task_db.get_task(task_id)
    if task is None or task["status"] != "awaiting_confirmation":
        return False, "找不到這個任務，或它已經不是「等待確認」的狀態了。"

    branch_name = task["branch_name"]
    worktree_path = task["worktree_path"]

    merge_result = _run_git(["merge", "--no-ff", branch_name, "-m", f"merge {branch_name}"])
    if merge_result.returncode != 0:
        task_db.update_task(task_id, status="failed", result_summary=f"合併失敗（可能有衝突）：\n{merge_result.stderr}")
        return False, f"合併失敗，需要手動處理：\n{merge_result.stderr[:1000]}"

    _run_git(["worktree", "remove", worktree_path, "--force"])
    _run_git(["branch", "-d", branch_name])

    task_db.update_task(task_id, status="merged")
    return True, "已經合併進主分支。"


def reject_task(task_id: int) -> None:
    task = task_db.get_task(task_id)
    if task is None:
        return
    worktree_path = task.get("worktree_path")
    branch_name = task.get("branch_name")
    if worktree_path and os.path.isdir(worktree_path):
        _run_git(["worktree", "remove", worktree_path, "--force"])
    if branch_name:
        _run_git(["branch", "-D", branch_name])
    task_db.update_task(task_id, status="rejected")
