"""
YouTube 影片字幕統整（Ryuzu 用）。

來源：課程練習 D:\\pyRag\\llamaIndex-tutorial\\lessons\\code\\Project\\linebot-llm\\youtube\\llm.py
（貼連結 → 抓字幕 → 統整成繁體中文）。搬過來時改了兩處：
  - youtube-transcript-api 1.x 改成物件寫法（YouTubeTranscriptApi().list()），原本的
    list_transcripts() 是 0.x 的寫法。
  - 統整不用 LlamaIndex SummaryIndex + tree_summarize（要另外裝 readers 和 llms-ollama 套件），
    改共用 pdf_rag 的「分段統整 → 合併」流程，跟 PDF 統整同一套提示詞風格、同一個模型。
"""

import re
from urllib.parse import parse_qs, urlparse

from opencc import OpenCC
from youtube_transcript_api import YouTubeTranscriptApi

import pdf_rag

# llama3.2 統整時偶爾混出簡體字（實測「他们」）。OpenCC 逐字轉換，但「台」
# 不轉（s2tw 會把「台」改成「臺」，見 cec_rag._to_traditional 的說明）。
_s2tw = OpenCC("s2tw").convert
_KEEP_CHARS = set("台")


def _to_traditional(text: str) -> str:
    converted = _s2tw(text)
    if len(converted) != len(text):
        return converted
    return "".join(o if o in _KEEP_CHARS else c for o, c in zip(text, converted))

PREFERRED_LANGS = ["zh-TW", "zh-Hant", "zh", "zh-Hans", "en"]
YOUTUBE_URL_PATTERN = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/\S+")


def find_youtube_url(text: str):
    """訊息裡有 YouTube 連結就回傳第一個，沒有回傳 None。"""
    m = YOUTUBE_URL_PATTERN.search(text or "")
    return m.group(0) if m else None


VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")


def get_video_id(youtube_link: str):
    url = urlparse(youtube_link)
    raw = None
    if url.hostname == "youtu.be":
        raw = url.path.lstrip("/")
    elif url.hostname and "youtube.com" in url.hostname:
        if url.path == "/watch":
            raw = parse_qs(url.query).get("v", [None])[0]
        elif url.path.startswith(("/shorts/", "/embed/", "/live/")):
            raw = url.path.split("/")[2]
    # 影片 ID 固定 11 碼；網址後面直接接中文（「…?v=xxxx好好笑」）時只取前 11 碼
    m = VIDEO_ID_PATTERN.match(raw or "")
    return m.group(0) if m else None


def fetch_transcript(video_id: str) -> tuple:
    """回傳 (字幕全文, 語言代碼)。沒有字幕時丟 ValueError（訊息直接回給使用者）。"""
    api = YouTubeTranscriptApi()
    try:
        transcripts = list(api.list(video_id))
    except Exception:
        raise ValueError("這部影片沒有字幕，或字幕功能被關閉了。")
    if not transcripts:
        raise ValueError("這部影片沒有任何語言的字幕。")
    by_lang = {t.language_code: t for t in transcripts}
    lang = next((l for l in PREFERRED_LANGS if l in by_lang), transcripts[0].language_code)
    fetched = by_lang[lang].fetch()
    text = " ".join(s.text.replace("\n", " ") for s in fetched)
    if not text.strip():
        raise ValueError("抓到的字幕是空的。")
    return text, lang


def summarize_youtube(youtube_link: str) -> str:
    """抓字幕 → 分段統整 → 合併。連結或字幕有問題時丟 ValueError。"""
    video_id = get_video_id(youtube_link)
    if not video_id:
        raise ValueError("這個連結看不出是哪一部 YouTube 影片。")
    text, lang = fetch_transcript(video_id)

    segments, truncated = pdf_rag._split_segments(text)
    if len(segments) == 1:
        summary = pdf_rag._ollama(
            "請用繁體中文統整底下這部影片的字幕，先用一兩句話說明影片主旨，再條列 3~8 個重點，"
            "只根據字幕內容，不要自己補充影片沒講的資訊。字幕可能有錯字或沒有標點，請依上下文理解。\n\n"
            f"字幕：\n{segments[0]}\n\n統整："
        )
    else:
        partials = [
            pdf_rag._ollama(
                f"底下是一部影片字幕的第 {i}/{len(segments)} 段，請用繁體中文條列這段的重點（最多 5 點），"
                f"只根據內容，不要補充。字幕可能有錯字或沒有標點。\n\n字幕：\n{seg}\n\n重點：",
                num_predict=400,
            )
            for i, seg in enumerate(segments, start=1)
        ]
        joined = "\n\n".join(f"[第 {i} 段]\n{p}" for i, p in enumerate(partials, start=1))
        summary = pdf_rag._ollama(
            "底下是同一部影片各段字幕的重點整理，請用繁體中文合併成一份完整統整：先用一兩句話說明"
            "影片主旨，再條列 5~10 個最重要的重點，去掉重複，不要補充原文沒有的資訊。\n\n"
            f"{joined}\n\n統整："
        )

    summary = _to_traditional(summary)
    if truncated:
        summary += f"\n\n（影片太長，只統整了前 {pdf_rag.MAX_SUMMARY_SEGMENTS * pdf_rag.SUMMARY_SEGMENT_CHARS} 字左右的字幕）"
    return f"{summary}\n\n（字幕語言：{lang}）"
