"""
CEC_API助手的分流回歸測試：只檢查「走哪條路、對到哪個按鈕／哪一條資料」，LLM 換成假的，幾十秒跑完。
每次改 cec_rag 的比對規則、門檻、分流順序後都跑一次：python cec_route_test.py

題目來源：規格書第 7 節情境、ppt/踩坑與試錯紀錄.md 的 E/F 案例、qa_log 真實提問、advice.md 第一節。
需要知識庫已匯入（webhook_app 啟動時會做；或先跑 python cec_rag.py）。
"""

import os
import sys

import cec_rag

STAIRS = "CEC_Detection.StairsHeightCheck"
FL = "CEC_QuickModeling.CreateFLByCut"
DKFL = "CEC_QuickModeling.CreateDKFLByCut"
L2B = "CEC_QuickModeling.Line2Beam"
CARBON = "CEC_QuantityTakeoff.CalculateEmbodiedCarbonByMass"
SKIP = "CEC_Skip.SelectByList"
DRIVEWAY = "CEC_QuickModeling.DrivewayCreate"

# 每個案例是一段對話：[(輸入, 預期)]。輸入是字串，或 {"pick": api} / {"item": 編號} 表示點按鈕。
# 預期：type（answer/ask/clarify/refer）、route 開頭、topic（session 目前的按鈕）、
#       choices_has（回問選項要包含的按鈕）、text_has（只檢查程式產生的固定文字）、
#       prompt_has / prompt_not（送給 LLM 的提示詞要有／不可有的字，例如聯絡人）。
CASES = [
    ("規格7-1 怎麼用", [("建立切割樓板怎麼用", {"type": "answer", "route": "目錄比對:" + FL})]),
    ("規格7-2 RC/Deck 回問", [
        ("切割樓板之後有一塊板不見了", {"type": "ask", "route": "目錄比對多個", "choices_has": [FL, DKFL]}),
        ({"pick": FL}, {"type": "answer", "route": "使用者選擇:" + FL, "topic": FL}),
    ]),
    ("規格7-3 多卡片同一訊息", [("跳出 請切換到 3D 視圖，並開啟剖面框確認檢查範圍後再執行。",
                         {"type": "answer", "route": "錯誤訊息比對:"})]),
    ("規格7-4 按鈕灰色", [("為什麼按鈕都是灰的", {"type": "answer", "route": "通用問題"})]),
    ("規格7-5 單線轉樑按不了", [("單線轉樑按不了", {"type": "answer", "route": "目錄比對:" + L2B})]),
    ("規格7-6 本機ID", [("本機ID要給誰", {"type": "answer", "route": "通用問題"})]),
    ("規格7-7 找功能", [("有沒有可以算磁磚的功能", {"type": "answer", "route": "功能目錄"})]),
    ("規格7-8 沒說哪個功能", [("報告存在哪裡？", {"type": "ask", "route": "比對不到→回問"})]),
    ("規格7-9 機電", [("機電的套管 API 怎麼用", {"route": "非本庫範圍", "text_has": "機電窗口"})]),
    ("規格7-10 Revit 授權", [("Revit 登不進去", {"type": "refer", "route": "非本庫範圍", "text_has": "Autodesk 窗口"})]),
    ("advice 1-1 CEC 授權過期", [("CEC API 授權過期了，按鈕全灰", {"type": "answer", "route": "通用問題"})]),
    ("advice 1-1 機電套管也列建築套管", [("機電套管要開口要用哪個",
                                 {"type": "ask", "route": "非本庫範圍+本庫相近", "choices_has": ["CEC_Integrate.Opening"]})]),
    ("advice 1-2 BIM 360", [("BIM 360 授權到期", {"type": "refer", "text_has": "Autodesk 窗口"})]),
    ("advice 1-3 樓梯干涉（別名核心詞）", [("樓梯干涉壞掉了", {"type": "clarify", "route": "釐清狀況:" + STAIRS})]),
    ("advice 1-4 對不對不是籠統回報", [("單線轉樑這樣對不對", {"type": "answer", "route": "目錄比對:" + L2B})]),
    ("F2 干涉檢查的期限不能對到門干涉", [("干涉檢查的期限是什麼", {"route_not": "目錄比對:CEC_Detection.DnWDetection"})]),
    ("F2 打錯字 單線轉粱", [("單線轉粱怎麼用", {"type": "answer", "route": "目錄比對:" + L2B})]),
    ("F1 追問延續話題", [
        ("建立切割樓板怎麼用", {"type": "answer", "topic": FL}),
        ("執行後的清單是什麼", {"type": "answer", "route": "延續話題:" + FL}),
    ]),
    ("F1 追問中找別的功能要跳出話題", [
        ("建立切割樓板怎麼用", {"type": "answer", "topic": FL}),
        ("有沒有可以算磁磚的", {"type": "answer", "route": "功能目錄"}),
    ]),
    ("F1-2 籠統回報→回問→點選項", [
        ("救命樓梯干涉壞掉了", {"type": "clarify", "topic": STAIRS}),
        ({"item": 1}, {"type": "answer", "route": "釐清後:" + STAIRS, "text_has": "欄杆扶手"}),
    ]),
    ("F1-2 回問後自己描述", [
        ("建立切割樓板壞掉了", {"type": "clarify", "topic": FL}),
        ("切完之後有一整塊樓板不見了", {"type": "answer", "route": "釐清後:" + FL}),  # 跟選項原文相同 → 當成點選
        ("新樓板的高度跟原本不一樣", {"type": "answer", "route": "延續話題:" + FL}),
    ]),
    ("F1-2 具體問題不回問", [("樓梯淨高檢查的報告存在哪裡", {"type": "answer", "route": "目錄比對:" + STAIRS})]),
    ("F1-2 怎麼用不回問", [("樓梯淨高檢查怎麼用", {"type": "answer"})]),
    ("10/7 回問後連點兩個選項", [
        ("干涉風險匯出隱含碳壞掉了", {"type": "clarify", "topic": CARBON}),
        ({"item": 0}, {"type": "answer", "route": "釐清後:" + CARBON}),
        ({"item": 2}, {"type": "answer", "route": "釐清後:" + CARBON, "text_has": "量體"}),
    ]),
    ("10/7 電腦版：回問狀況時輸入編號", [
        ("干涉風險匯出隱含碳壞掉了", {"type": "clarify", "text_has": "輸入編號"}),
        ("3", {"type": "answer", "route": "釐清後:" + CARBON}),
    ]),
    ("10/7 電腦版：照清單打選項原文（含「機電」不可轉介）", [
        ("干涉風險匯出隱含碳壞掉了", {"type": "clarify"}),
        ("鋼構、機電、連結檔有算嗎？", {"type": "answer", "route": "釐清後:" + CARBON}),
    ]),
    ("10/7 沒有話題時問本庫常見問題原文（含「機電」）", [
        ("鋼構、機電、連結檔有算嗎？", {"type": "answer", "route": "常見問題原題:" + CARBON}),
    ]),
    ("10/7 電腦版：回問是哪個功能時輸入編號", [
        ("切割樓板之後有一塊板不見了", {"type": "ask", "text_has": "1. "}),
        ("2", {"type": "answer", "route": "使用者選擇:"}),
    ]),
    ("10/7 聊某按鈕時提到機電＝追問，聯絡人只給建築", [
        ("干涉風險匯出隱含碳是做什麼的", {"type": "answer", "topic": CARBON}),
        ("那機電管線的元件也會算進去嗎", {"type": "answer", "route": "延續話題:" + CARBON,
                                "prompt_has": "霽倫", "prompt_not": "機電窗口"}),
    ]),
    ("10/7 聊某按鈕時明確問機電 API＝新問題，照樣轉介", [
        ("干涉風險匯出隱含碳是做什麼的", {"type": "answer", "topic": CARBON}),
        ("那機電的 API 有類似功能嗎", {"type": "refer", "route": "非本庫範圍", "text_has": "機電窗口"}),
    ]),
    ("10/7 聊某按鈕時問 Revit 登入＝新問題，照樣轉介", [
        ("干涉風險匯出隱含碳是做什麼的", {"type": "answer", "topic": CARBON}),
        ("Revit 登不進去怎麼辦", {"type": "refer", "text_has": "Autodesk 窗口"}),
    ]),
    ("10/7 聊某按鈕時問授權＝通用問題，聯絡人不限縮", [
        ("干涉風險匯出隱含碳是做什麼的", {"type": "answer", "topic": CARBON}),
        ("CEC 授權過期了要找誰", {"type": "answer", "prompt_has": "機電窗口"}),
    ]),
    ("10/7 實測：聊略過時問車道（沒講完整按鈕名稱）→ 回問是否換話題", [
        ("要怎麼略過檢查", {"type": "answer", "topic": SKIP}),
        ("車道要怎麼建", {"type": "ask", "route": "疑似換話題", "choices_has": [SKIP, DRIVEWAY]}),
        ({"pick": DRIVEWAY}, {"type": "answer", "route": "使用者選擇:" + DRIVEWAY, "topic": DRIVEWAY}),
    ]),
    ("10/7 疑似換話題時選回原本的按鈕", [
        ("要怎麼略過檢查", {"type": "answer", "topic": SKIP}),
        ("車道怎麼用", {"type": "ask", "route": "疑似換話題"}),
        ("1", {"type": "answer", "route": "使用者選擇:" + SKIP, "topic": SKIP}),
    ]),
    ("10/7 正常追問不可誤判換話題", [
        ("要怎麼略過檢查", {"type": "answer", "topic": SKIP}),
        ("清單上的欄位是什麼意思", {"type": "answer", "route": "延續話題:" + SKIP}),
        ("物件移動位置之後會怎樣", {"type": "answer", "route": "延續話題:" + SKIP}),
    ]),
    ("錯誤訊息含按鈕名稱仍走錯誤比對", [("剖面框內無樓梯淨高檢查量體，請確認。",
                              {"type": "answer", "route": "錯誤訊息比對:" + STAIRS, "text_has": "建置樓梯淨高量體"})]),
]


def run():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "（測試用假回答）"

    cec_rag._llm = fake_llm
    cec_rag._log = lambda entry: None
    passed = failed = 0
    lines = []
    for name, turns in CASES:
        user = "route-test:" + name
        cec_rag.reset_session(user)
        for inp, exp in turns:
            if isinstance(inp, dict) and "pick" in inp:
                r = cec_rag.answer("", user_id=user, forced_api=inp["pick"])
            elif isinstance(inp, dict):
                r = cec_rag.answer("", user_id=user, item_index=inp["item"])
            else:
                r = cec_rag.answer(inp, user_id=user)
            topic = cec_rag._sessions.get(user, {}).get("api")
            problems = []
            if "type" in exp and r["type"] != exp["type"]:
                problems.append(f"type={r['type']}")
            if "route" in exp and not (r.get("route") or "").startswith(exp["route"]):
                problems.append(f"route={r.get('route')}")
            if "route_not" in exp and (r.get("route") or "").startswith(exp["route_not"]):
                problems.append(f"route={r.get('route')}")
            if "topic" in exp and topic != exp["topic"]:
                problems.append(f"topic={topic}")
            got = [a for a, _ in r.get("choices") or []]
            if any(a not in got for a in exp.get("choices_has", [])):
                problems.append(f"choices={got}")
            if "text_has" in exp and exp["text_has"] not in r.get("text", ""):
                problems.append("text 缺「" + exp["text_has"] + "」")
            prompt = prompts[-1] if prompts else ""
            if "prompt_has" in exp and exp["prompt_has"] not in prompt:
                problems.append("提示詞缺「" + exp["prompt_has"] + "」")
            if "prompt_not" in exp and exp["prompt_not"] in prompt:
                problems.append("提示詞不該有「" + exp["prompt_not"] + "」")
            prompts.clear()
            label = inp if isinstance(inp, str) else str(inp)
            if problems:
                failed += 1
                lines.append(f"✗ {name}｜{label}｜" + "；".join(problems))
            else:
                passed += 1
                lines.append(f"✓ {name}｜{label}｜{r.get('route')}")
    lines.append(f"\n通過 {passed}，失敗 {failed}")
    return failed, "\n".join(lines)


if __name__ == "__main__":
    failed, report = run()
    # Windows 終端機是 cp950，中文報告寫到 UTF-8 檔案（可用第一個參數指定路徑）
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(cec_rag.STORE_DIR, "route_test_result.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"failed={failed}, report -> {path}")
    sys.exit(1 if failed else 0)
