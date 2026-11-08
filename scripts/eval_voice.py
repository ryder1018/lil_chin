"""Step 3 語音端到端評估：假裝自己是 ESP32 裝置，用 WebSocket + Opus 跟機器人後端（server/）對話。

對 eval/voice_testset.yaml 的每一題：
  1. 連線、hello 握手（跟真的裝置一樣）
  2. 以真實速度（每 60ms 一個 frame）送出語音，接著送靜音，讓伺服器的 VAD 自己判斷講完了
  3. 記錄伺服器回來的每個訊息時間：stt（辨識結果）、tts sentence_start、第一個音訊封包、tts stop
  4. 把機器人的回答音訊解碼存成 wav

量測：
  端到端延遲 = 第一個回覆音訊封包 - 使用者講完最後一個字   ← 目標 < 1.5 秒
    = VAD 等待靜音 + ASR + LLM 第一句 + TTS 第一句 + 傳輸
  ASR CER    = 辨識結果 vs 原句 的字錯誤率（轉成台灣繁體、去標點後比）

先啟動服務（bash scripts/robot.sh start），再：
  conda activate ai_robot && python scripts/eval_voice.py --tag step3
伺服器那邊每一輪的詳細計時（ASR、大腦、TTS）在 logs/conversations/日期.jsonl 的 timing 欄位。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import opuslib_next as opus
import websockets
from websockets.protocol import State
import yaml

ROOT = Path(__file__).resolve().parent.parent
UTT = ROOT / "eval/voice/utterances"
FRAME_MS = 60
IN_SR = 16000
IN_FRAME = IN_SR * FRAME_MS // 1000  # 960 samples

try:  # 跟 text_filter 一樣的簡轉繁（ASR 輸出簡體）
    sys.path.insert(0, str(ROOT))
    from server.text_filter import to_taiwan
except Exception:
    to_taiwan = None

PUNCT = re.compile(r"[\s，。！？、；：,.!?;:~～\"'「」『』（）()…\-]")


def norm(text: str) -> str:
    if to_taiwan:
        text = to_taiwan(text)
    return PUNCT.sub("", text).lower()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def speech_end_offset(pcm: np.ndarray, thresh_db: float = -40.0) -> float:
    """音檔裡最後一個有聲音的 frame 結束時間（秒）。TTS 合成的音檔尾巴本身有靜音，不能只扣固定值。"""
    win = IN_SR // 100  # 10ms
    n = len(pcm) // win
    if n == 0:
        return 0.0
    x = pcm[: n * win].astype(np.float32).reshape(n, win) / 32768
    db = 20 * np.log10(np.sqrt((x**2).mean(axis=1)) + 1e-9)
    voiced = np.nonzero(db > thresh_db)[0]
    return ((voiced[-1] + 1) * win / IN_SR) if len(voiced) else len(pcm) / IN_SR


def load_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == IN_SR and w.getnchannels() == 1, f"{path} 要 16kHz 單聲道"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


async def run_turn(ws, enc, pcm: np.ndarray, out_sr: int, dec, silence_s: float, timeout: float) -> dict:
    """送一句話，收完整個回覆。回傳時間軸與內容。"""
    ev: dict = {"stt": None, "sentences": [], "llm_emotion": None, "tool_msgs": [], "tools": []}
    audio_out: list[bytes] = []
    done = asyncio.Event()
    t = {}

    async def receiver():
        async for msg in ws:
            now = time.perf_counter()
            if isinstance(msg, bytes):
                if "speech_end" in t and "first_audio" not in t and "stt" in t:
                    t["first_audio"] = now
                if "stt" in t:
                    audio_out.append(dec.decode(msg, out_sr * FRAME_MS // 1000))
                continue
            m = json.loads(msg)
            typ, st = m.get("type"), m.get("state")
            if typ == "stt":
                text = m.get("text", "")
                if text.startswith("%"):  # 有些伺服器（例如 xiaozhi）呼叫工具時會送「% 工具名」當作 stt
                    ev["tools"].append(text.lstrip("% ").strip())
                elif "stt" not in t:  # 只留第一個（真正的辨識結果）
                    t["stt"] = now
                    ev["stt"] = text
            elif typ == "llm":
                ev["llm_emotion"] = m.get("emotion")
            elif typ == "tts" and st == "sentence_start":
                t.setdefault("first_sentence", now)
                ev["sentences"].append(m.get("text", ""))
            elif typ == "tts" and st == "stop":
                t["tts_stop"] = now
                done.set()
                return
            elif typ == "mcp" or typ == "iot":
                ev["tool_msgs"].append(m)

    rx = asyncio.create_task(receiver())
    await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))

    # 依真實速度送語音
    frames = [pcm[i : i + IN_FRAME] for i in range(0, len(pcm), IN_FRAME)]
    t0 = time.perf_counter()
    for k, f in enumerate(frames):
        if len(f) < IN_FRAME:
            f = np.pad(f, (0, IN_FRAME - len(f)))
        await ws.send(enc.encode(f.tobytes(), IN_FRAME))
        await asyncio.sleep(max(0, t0 + (k + 1) * FRAME_MS / 1000 - time.perf_counter()))
    # 真正講完的時間點 = 開始送的時間 + 音檔裡最後一個有聲音的位置
    t["speech_end"] = t0 + speech_end_offset(pcm)
    # 接著送靜音，讓 VAD 判斷講完了（真實裝置也會一直送）
    silence = np.zeros(IN_FRAME, dtype=np.int16).tobytes()
    t1 = time.perf_counter()
    k = 0
    while "stt" not in t and time.perf_counter() - t1 < silence_s:
        await ws.send(enc.encode(silence, IN_FRAME))
        k += 1
        await asyncio.sleep(max(0, t1 + k * FRAME_MS / 1000 - time.perf_counter()))
    try:
        await asyncio.wait_for(done.wait(), timeout)
    except asyncio.TimeoutError:
        rx.cancel()

    def ms(a, b="speech_end"):
        return round((t[a] - t[b]) * 1000) if a in t and b in t else None

    pcm_out = b"".join(audio_out)
    return {
        "stt": ev["stt"],
        "reply": "｜".join(ev["sentences"]),
        "sentences": ev["sentences"],
        "emotion": ev["llm_emotion"],
        "tool_msgs": ev["tool_msgs"],
        "tools": ev["tools"],
        "stt_ms": ms("stt"),  # VAD 等靜音 + ASR
        "first_sentence_ms": ms("first_sentence"),
        "e2e_ms": ms("first_audio"),  # ← 主指標
        "reply_audio_s": round(len(pcm_out) / 2 / out_sr, 2),
        "_pcm": pcm_out,
    }


async def run_item(url: str, item: dict, args) -> dict:
    headers = {"device-id": f"eval-{uuid.uuid4().hex[:8]}", "client-id": "eval", "protocol-version": "1"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "hello", "version": 1, "transport": "websocket",
            "audio_params": {"format": "opus", "sample_rate": IN_SR, "channels": 1, "frame_duration": FRAME_MS},
        }))
        hello = json.loads(await asyncio.wait_for(ws.recv(), 10))
        out_sr = hello.get("audio_params", {}).get("sample_rate", 24000)
        enc = opus.Encoder(IN_SR, 1, opus.APPLICATION_VOIP)
        dec = opus.Decoder(out_sr, 1)
        turns = []
        for k, text in enumerate(item["turns"]):
            pcm = load_pcm16(UTT / f"{item['id']}_{k}.wav")
            r = await run_turn(ws, enc, pcm, out_sr, dec, args.silence, args.timeout)
            r["ref"] = text
            r["cer"] = round(cer(text, r["stt"] or ""), 3) if r["stt"] is not None else 1.0
            wav_path = ROOT / f"eval/voice/replies/{args.tag}/{item['id']}_{k}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(out_sr)
                w.writeframes(r.pop("_pcm"))
            # 道別時伺服器會在說完後主動關閉連線（xiaozhi-server 則是送「% 工具名」的 stt，見 receiver）
            try:
                await asyncio.wait_for(ws.wait_closed(), 1.0)
                r["ended"] = True
            except asyncio.TimeoutError:
                r["ended"] = False
            r["ended"] = r["ended"] or "handle_exit_intent" in r["tools"]
            turns.append(r)
            if ws.state is not State.OPEN:
                break
        return {"id": item["id"], "focus": item.get("focus"), "turns": turns}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


async def main_async(args) -> None:
    items = yaml.safe_load((ROOT / "eval/voice_testset.yaml").read_text())["items"]
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]
    results = []
    for it in items:
        r = await run_item(args.url, it, args)
        results.append(r)
        for k, t in enumerate(r["turns"]):
            print(f"{r['id']}_{k} e2e {t['e2e_ms']}ms (stt {t['stt_ms']}ms) CER {t['cer']:.2f}")
            tools = f"  🔧 {', '.join(t['tools'])}" if t["tools"] else ""
            tools += "  👋（結束對話）" if t["ended"] else ""
            print(f"   👤 {t['ref']}\n   👂 {t['stt']}\n   🤖 {t['reply']}{tools}")

    turns = [t for r in results for t in r["turns"]]
    want_exit = {i["id"] for i in items if i.get("exit")}
    # 道別題只看最後一句
    exit_turns = [r["turns"][-1] for r in results if r["id"] in want_exit]
    other_turns = [t for r in results for k, t in enumerate(r["turns"])
                   if not (r["id"] in want_exit and k == len(r["turns"]) - 1)]
    e2e = [t["e2e_ms"] for t in turns]
    summary = {
        "turns": len(turns),
        "e2e_ms_median": pct(e2e, 0.5),
        "e2e_ms_p90": pct(e2e, 0.9),
        "e2e_ms_max": max((x for x in e2e if x is not None), default=None),
        "e2e_under_1500": f"{sum(1 for x in e2e if x is not None and x < 1500)}/{len(e2e)}",
        "stt_ms_median": pct([t["stt_ms"] for t in turns], 0.5),
        "first_sentence_ms_median": pct([t["first_sentence_ms"] for t in turns], 0.5),
        "cer_mean": round(statistics.mean(t["cer"] for t in turns), 3),
        "no_reply": sum(1 for t in turns if not t["reply"]),
        # 道別題（exit: true）有結束對話 / 其他題被誤判成道別
        "exit_ok": f"{sum(t['ended'] for t in exit_turns)}/{len(exit_turns)}",
        "exit_false": f"{sum(t['ended'] for t in other_turns)}/{len(other_turns)}",
    }
    print("\n=== 摘要 ===")
    for k, v in summary.items():
        print(f"{k:26s} {v}")
    out = ROOT / f"logs/eval_voice_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    print(f"\n結果：{out}\n回答音檔：eval/voice/replies/{args.tag}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/xiaozhi/v1/")
    ap.add_argument("--tag", default="step3")
    ap.add_argument("--only", default="")
    ap.add_argument("--silence", type=float, default=3.0, help="講完後最多送幾秒靜音等 VAD")
    ap.add_argument("--timeout", type=float, default=30.0, help="等回覆講完的上限")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
