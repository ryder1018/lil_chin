"""一個裝置連線 = 一個 Session。

狀態機：
    listening ──(VAD 判斷講完 / manual 放開按鈕)──▶ thinking ──(第一句語音)──▶ speaking
        ▲                                                                        │
        └──────────────────────(說完 tts stop / 被打斷 abort)─────────────────────┘

一輪對話（turn）的流水線，三段同時進行：
    大腦串流吐句子 ──▶ 句子佇列 ──▶ TTS 合成 + Opus 編碼 ──▶ AudioSender 照播放速度送出
所以第一句合成完就開始播，後面的句子邊播邊合成。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import protocol
from .audio import AudioSender, OpusIn, OpusOut
from .brain import Brain, Reply
from .text_filter import to_taiwan

if TYPE_CHECKING:
    from .app import Context

PREROLL_FRAMES = 10  # 開始說話前保留的音訊（60ms × 10），避免第一個字被切掉


class Session:
    def __init__(self, ws, ctx: "Context", device_id: str, client_id: str):
        self.ws = ws
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.id = uuid.uuid4().hex[:12]
        self.device_id = device_id
        self.client_id = client_id
        self.log = ctx.logger.getChild(self.id)

        a = self.cfg["audio"]
        self.opus_in = OpusIn(a["in_sample_rate"], a["frame_ms"])
        self.opus_out = OpusOut(a["out_sample_rate"], a["frame_ms"])
        self.sender = AudioSender(self._send_bytes, a["frame_ms"])
        self.vad = ctx.vad.stream()
        self.brain = ctx.new_brain()

        self.mode = "auto"
        self.state = "listening"
        self.preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self.speech: list[bytes] = []
        self.turn: asyncio.Task | None = None
        self.last_activity = time.monotonic()
        self.closing = False
        self.warm_task: asyncio.Task | None = None
        self.last_warm = 0.0

    # ---------- 傳送 ----------
    async def _send_bytes(self, data: bytes | str) -> None:
        await self.ws.send(data)

    async def send(self, msg: str) -> None:
        try:
            await self.ws.send(msg)
        except Exception:
            pass

    # ---------- 主迴圈 ----------
    async def run(self) -> None:
        self.sender.start()
        idle = asyncio.create_task(self._idle_watch())
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    await self._on_audio(msg)
                else:
                    await self._on_text(msg)
        finally:
            idle.cancel()
            if self.turn:
                self.turn.cancel()
            self.sender.stop()
            self.log.info("連線結束")

    async def _on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        typ = msg.get("type")
        if typ == "hello":
            a = self.cfg["audio"]
            await self.send(protocol.hello(self.id, a["out_sample_rate"], a["frame_ms"]))
            self.log.info(f"hello：device={self.device_id} client={self.client_id} audio={msg.get('audio_params')}")
        elif typ == "listen":
            state = msg.get("state")
            if "mode" in msg:
                self.mode = msg["mode"]
            if state == "start":
                self._reset_listening()
            elif state == "stop" and self.speech:  # manual 模式：放開按鈕 = 講完
                self._start_turn(audio=b"".join(self.speech))
            elif state == "detect":
                text = (msg.get("text") or "").strip()
                if text and not self._is_wakeup(text):
                    self._start_turn(text=text)  # 直接傳文字（網頁的輸入框）
                else:
                    self.log.info(f"喚醒：{text}")
                    self._reset_listening()
        elif typ == "abort":
            await self.abort("裝置要求")
        # mcp / iot 之後接硬體再處理

    def _is_wakeup(self, text: str) -> bool:
        t = to_taiwan(text).strip("，。！？!?, ")
        return t in self.cfg["character"]["wakeup_words"]

    # ---------- 收音 ----------
    def _reset_listening(self) -> None:
        self.vad.reset()
        self.preroll.clear()
        self.speech = []
        if self.state != "speaking" and self.state != "thinking":
            self.state = "listening"

    async def _on_audio(self, packet: bytes) -> None:
        busy = self.state in ("thinking", "speaking")
        if busy and self.mode != "realtime":
            return  # 半雙工：自己在講話時不聽（還沒有回音消除）
        try:
            pcm = self.opus_in.decode(packet)
        except Exception:
            return
        if self.mode == "manual":
            self.speech.append(pcm)
            return

        event = self.vad.feed(pcm)
        if self.vad.speaking or event == "end":
            self._prewarm()
        if self.vad.speaking or event == "end":
            if not self.speech:
                self.speech = list(self.preroll)
            self.speech.append(pcm)
        else:
            self.preroll.append(pcm)

        if event == "start":
            self.last_activity = time.monotonic()
            if busy:  # realtime 模式：使用者開口 = 打斷
                await self.abort("使用者插話")
        elif event == "end":
            self.last_activity = time.monotonic()
            audio, self.speech = b"".join(self.speech), []
            self.preroll.clear()
            self._start_turn(audio=audio)

    def _prewarm(self) -> None:
        """說話中每 0.8 秒讓大腦預熱一次（見 Brain.prewarm），講完那一刻也會再打一次。"""
        now = time.monotonic()
        if self.warm_task and not self.warm_task.done():
            return
        if now - self.last_warm < 0.8:
            return
        self.last_warm = now
        self.warm_task = asyncio.create_task(self.brain.prewarm())

    # ---------- 一輪對話 ----------
    def _start_turn(self, audio: bytes | None = None, text: str | None = None) -> None:
        if self.turn and not self.turn.done():
            self.turn.cancel()
            self.sender.clear()
        self.state = "thinking"
        self.turn = asyncio.create_task(self._turn(audio, text, time.monotonic()))

    async def _turn(self, audio: bytes | None, text: str | None, t_end: float) -> None:
        loop = asyncio.get_running_loop()
        rec = Reply(user="")
        timing: dict[str, float | None] = {"vad_silence_ms": self.cfg["vad"]["min_silence_ms"] if audio else 0}
        asr_info: dict = {}
        try:
            if audio is not None:
                r = await loop.run_in_executor(self.ctx.asr_pool, self.ctx.asr.transcribe, audio)
                timing["asr_ms"] = round(r.ms)
                asr_info = {"raw": r.text, "emotion": r.emotion, "language": r.language, "event": r.event,
                            "audio_s": round(len(audio) / 2 / self.cfg["audio"]["in_sample_rate"], 2)}
                text = r.text
            text = to_taiwan(text or "").strip()
            if not any(c.isalnum() for c in text):
                self.log.info("沒聽到內容，繼續聽")
                self.state = "listening"
                return

            self.log.info(f"👤 {text}")
            await self.send(protocol.stt(self.id, text))
            await self.send(protocol.tts(self.id, "start"))

            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            speaker = asyncio.create_task(self._speak(sentences, timing, t_end))
            async for s in self.brain.reply(text, rec):
                await sentences.put(s)
            await sentences.put(None)
            await speaker
            await self.sender.drain()
            await self.send(protocol.tts(self.id, "stop"))
            self.log.info(f"🤖 {''.join(rec.spoken)}" + ("（結束對話）" if rec.exit else ""))
        except asyncio.CancelledError:
            rec.spoken.append("［被打斷］")
            raise
        except Exception as e:
            self.log.exception(f"這輪對話出錯：{e}")
            await self.send(protocol.tts(self.id, "stop"))
        finally:
            if self.turn is asyncio.current_task():  # 被新的一輪取代時，不要動新一輪的狀態
                self.sender.reset()
                self.state = "listening"
                self.vad.reset()
            if rec.user:
                self._write_log(rec, asr_info, timing)
        if rec.exit:
            await self.close("道別")

    async def _speak(self, sentences: asyncio.Queue, timing: dict, t_end: float) -> None:
        loop = asyncio.get_running_loop()
        tts = self.ctx.tts
        while (s := await sentences.get()) is not None:
            t = time.monotonic()
            audio = await loop.run_in_executor(self.ctx.tts_pool, tts.synthesize, s)
            if "tts_first_ms" not in timing:
                timing["tts_first_ms"] = round((time.monotonic() - t) * 1000)
                timing["first_sentence_ready_ms"] = round((t - t_end) * 1000)
            packets = self.opus_out.encode(audio)
            await self.sender.put([protocol.tts(self.id, "sentence_start", s)])
            await self.sender.put(packets)
            self.state = "speaking"
            if "first_audio_ms" not in timing:
                while self.sender.first_sent_at is None and self.sender.q.qsize():
                    await asyncio.sleep(0.005)
                if self.sender.first_sent_at:
                    # 伺服器判斷講完 → 第一個語音封包送出
                    timing["first_audio_ms"] = round((self.sender.first_sent_at - t_end) * 1000)

    async def abort(self, why: str) -> None:
        if self.turn and not self.turn.done():
            self.log.info(f"打斷（{why}）")
            self.turn.cancel()
            self.sender.clear()
            await self.send(protocol.tts(self.id, "stop"))
        self.state = "listening"

    # ---------- 其他 ----------
    async def _idle_watch(self) -> None:
        limit = self.cfg["session"]["idle_close_s"]
        while True:
            await asyncio.sleep(5)
            if self.state == "listening" and time.monotonic() - self.last_activity > limit:
                await self.close(f"{limit} 秒沒人說話")
                return

    async def close(self, why: str) -> None:
        if self.closing:
            return
        self.closing = True
        self.log.info(f"關閉連線（{why}）")
        try:
            await self.ws.close()
        except Exception:
            pass

    def _write_log(self, rec: Reply, asr_info: dict, timing: dict) -> None:
        d = Path(self.ctx.conv_dir)
        d.mkdir(parents=True, exist_ok=True)
        t = dict(timing)
        if t.get("first_audio_ms") is not None:
            # 使用者真正講完 → 第一個語音封包（= 伺服器端的端到端延遲）
            t["e2e_ms"] = t["first_audio_ms"] + (t.get("vad_silence_ms") or 0)
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "session": self.id,
            "device": self.device_id,
            "user": rec.user,
            "asr": asr_info,
            "reply": "".join(rec.spoken),
            "raw": rec.raw,
            "tool_calls": rec.tool_calls or None,
            "exit": rec.exit,
            "exit_by": rec.exit_by,
            "timing": t,
            "llm": {k: v for k, v in asdict(rec).items()
                    if k in ("ttft_ms", "first_segment_ms", "total_ms", "prompt_tokens", "cached_tokens")},
        }
        with (d / f"{datetime.now():%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
