"""裝置 ↔ 伺服器的訊息格式。

刻意和小智 ESP32 韌體（xiaozhi-esp32）的 WebSocket 協定相容，之後燒小智韌體的硬體可以直接連進來。
協定參考自 xinnan-tech/xiaozhi-esp32-server（MIT），見 THIRD_PARTY_NOTICES.md。

連線：ws://<host>:8000/xiaozhi/v1/
  裝置身分放在 header（device-id / client-id），瀏覽器不能自訂 header，所以也接受同名 query 參數。

裝置 → 伺服器
  文字（JSON）：
    {"type":"hello","audio_params":{"format":"opus","sample_rate":16000,"channels":1,"frame_duration":60}}
    {"type":"listen","state":"start","mode":"auto|manual|realtime"}   開始收音
    {"type":"listen","state":"stop"}                                   manual 模式：按鈕放開 = 講完
    {"type":"listen","state":"detect","text":"你好小晴"}               裝置端喚醒詞（或直接傳文字）
    {"type":"abort"}                                                   打斷正在說的話
  二進位：Opus 封包（16kHz 單聲道，每包 60ms）

伺服器 → 裝置
  {"type":"hello","transport":"websocket","session_id":..,"audio_params":{..24000..}}
  {"type":"stt","text":"使用者說的話"}            辨識結果（裝置螢幕顯示）
  {"type":"llm","text":"😊","emotion":"happy"}    表情（之後給 OLED 臉用）
  {"type":"tts","state":"start"}                   開始說話
  {"type":"tts","state":"sentence_start","text":"這一句"}
  二進位：Opus 封包（24kHz 單聲道，每包 60ms），依播放速度送出
  {"type":"tts","state":"stop"}                    說完了（韌體收到後切回收音）
"""
from __future__ import annotations

import json


def _dump(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def hello(session_id: str, sample_rate: int, frame_ms: int) -> str:
    return _dump({
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": {"format": "opus", "sample_rate": sample_rate, "channels": 1, "frame_duration": frame_ms},
    })


def stt(session_id: str, text: str) -> str:
    return _dump({"type": "stt", "text": text, "session_id": session_id})


def tts(session_id: str, state: str, text: str | None = None) -> str:
    msg = {"type": "tts", "state": state, "session_id": session_id}
    if text is not None:
        msg["text"] = text
    return _dump(msg)


def emotion(session_id: str, name: str, emoji: str) -> str:
    return _dump({"type": "llm", "text": emoji, "emotion": name, "session_id": session_id})
