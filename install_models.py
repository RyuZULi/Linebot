"""
照 models.txt 安裝本專案需要的模型。通常直接雙擊 install_models.bat 即可。

用法：
  python install_models.py            安裝清單上還沒有的模型（已經有的跳過）
  python install_models.py --check    只檢查，列出哪些已安裝、哪些還沒有
  python install_models.py --update   已經有的也重新拉最新版

已安裝的模型預設「不更新」：熱量門檻、回答品質都是用目前的模型版本實測校準的，
換版本可能讓結果改變，要更新請明確加 --update。
"""

import os
import shutil
import subprocess
import sys

LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.txt")

# HuggingFace 模型庫常同時放 PyTorch / TensorFlow / ONNX 等多種格式，只下載程式用得到的
# （bge-m3 整個 repo 下載會多出好幾 GB 的 ONNX 檔）。
HF_IGNORE_PATTERNS = ["*.onnx", "onnx/*", "*.onnx_data", "*.h5", "*.msgpack", "*.ot", "*.tflite", "imgs/*"]


def read_list() -> list:
    entries = []
    with open(LIST_PATH, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or parts[0] not in ("ollama", "hf"):
                print(f"[略過] models.txt 第 {lineno} 行看不懂：{line}")
                continue
            entries.append((parts[0], parts[1]))
    return entries


# ---------- Ollama ----------

def ollama_installed() -> set:
    """回傳已安裝的模型名稱（同時收「minicpm-v」和「minicpm-v:latest」兩種寫法）。"""
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "ollama list 執行失敗")
    names = set()
    for line in out.stdout.splitlines()[1:]:
        if line.strip():
            name = line.split()[0]
            names.add(name)
            if name.endswith(":latest"):
                names.add(name[: -len(":latest")])
    return names


def ollama_pull(name: str) -> bool:
    return subprocess.run(["ollama", "pull", name]).returncode == 0


# ---------- HuggingFace ----------

def hf_installed(repo: str) -> bool:
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def hf_download(repo: str) -> bool:
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, ignore_patterns=HF_IGNORE_PATTERNS)
        return True
    except Exception as e:
        print(f"    {e}")
        return False


def main():
    check_only = "--check" in sys.argv
    update = "--update" in sys.argv
    entries = read_list()
    print(f"models.txt 共 {len(entries)} 個模型\n")

    installed_ollama = set()
    if any(src == "ollama" for src, _ in entries):
        if not shutil.which("ollama"):
            print("[錯誤] 找不到 ollama 指令，請先安裝 Ollama：https://ollama.com")
            return 1
        try:
            installed_ollama = ollama_installed()
        except RuntimeError as e:
            print(f"[錯誤] 連不上 Ollama（{e}）。請確認 Ollama 已經啟動（工作列有圖示，或執行 ollama serve）。")
            return 1

    missing, failed = [], []
    for src, name in entries:
        have = name in installed_ollama if src == "ollama" else hf_installed(name)
        label = f"{src:6} {name}"
        if check_only:
            print(f"{'[已安裝]' if have else '[缺少]  '} {label}")
            if not have:
                missing.append(label)
            continue
        if have and not update:
            print(f"[已安裝，略過] {label}")
            continue
        print(f"[{'更新' if have else '下載'}] {label}")
        ok = ollama_pull(name) if src == "ollama" else hf_download(name)
        if not ok:
            failed.append(label)

    print()
    if check_only:
        print("全部都已安裝。" if not missing else f"缺少 {len(missing)} 個，執行 install_models.bat 安裝。")
        return 1 if missing else 0
    if failed:
        print(f"[失敗] 下列 {len(failed)} 個沒有裝好，請檢查網路或磁碟空間後再執行一次：")
        for label in failed:
            print(f"  {label}")
        return 1
    print("完成，清單上的模型都已就緒。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
