"""Opus 編解碼 + 依播放速度送出音訊。"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

import numpy as np
import opuslib_next as opus

PRE_BUFFER = 3  # 前幾個封包立刻送，讓裝置先有緩衝，之後照播放速度送


class OpusIn:
    """裝置送來的 Opus 封包 → 16-bit PCM bytes。"""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 60):
        self.dec = opus.Decoder(sample_rate, 1)
        self.frame = sample_rate * frame_ms // 1000

    def decode(self, packet: bytes) -> bytes:
        return self.dec.decode(packet, self.frame)


class OpusOut:
    """float32 / int16 音訊 → 一串 60ms 的 Opus 封包。"""

    def __init__(self, sample_rate: int = 24000, frame_ms: int = 60):
        self.enc = opus.Encoder(sample_rate, 1, opus.APPLICATION_AUDIO)
        self.enc.bitrate = 24000
        self.enc.complexity = 10
        self.enc.signal = opus.SIGNAL_VOICE
        self.frame = sample_rate * frame_ms // 1000

    def encode(self, audio: np.ndarray) -> list[bytes]:
        if audio.dtype != np.int16:
            audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        n = -(-len(audio) // self.frame) * self.frame  # 補到整數個 frame
        audio = np.pad(audio, (0, n - len(audio)))
        return [self.enc.encode(audio[i : i + self.frame].tobytes(), self.frame) for i in range(0, n, self.frame)]


class AudioSender:
    """把 Opus 封包照「播放速度」送給裝置。

    一次全部丟過去，ESP32 的緩衝會爆；太慢則會斷斷續續。
    做法（參考 xiaozhi-esp32-server 的 AudioRateController）：前 PRE_BUFFER 包立刻送，
    之後第 k 包在「開始播放時間 + (k - PRE_BUFFER) × 60ms」送出。
    TTS 來不及（佇列空了）時重新對齊時間，避免之後一口氣補送。
    """

    def __init__(self, send: Callable[[bytes], Awaitable[None]], frame_ms: int = 60):
        self._send = send
        self.dt = frame_ms / 1000
        self.q: asyncio.Queue[bytes | str] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.gen = 0  # 每次 clear() +1；已經從佇列拿出來、正在等時間的封包，送之前檢查還是不是同一代
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.play_start = 0.0
        self.first_sent_at: float | None = None

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def put(self, packets: list[bytes | str]) -> None:
        """bytes = 音訊封包（照播放速度送）；str = 文字訊息（例如 sentence_start），夾在音訊之間依序送出。"""
        for p in packets:
            await self.q.put(p)

    def clear(self) -> None:
        """打斷：丟掉還沒送的封包。"""
        while not self.q.empty():
            self.q.get_nowait()
        self.gen += 1
        self.reset()

    async def drain(self) -> None:
        """等到佇列送完，而且裝置應該已經播完。"""
        while not self.q.empty():
            await asyncio.sleep(0.02)
        if self.count:
            remain = self.play_start + self.count * self.dt - time.monotonic()
            if remain > 0:
                await asyncio.sleep(remain)

    async def _run(self) -> None:
        while True:
            pkt = await self.q.get()
            gen = self.gen
            if isinstance(pkt, str):
                try:
                    await self._send(pkt)
                except Exception:
                    return
                continue
            now = time.monotonic()
            k = self.count
            if k == 0:
                self.play_start = now
            due = self.play_start + (k - PRE_BUFFER) * self.dt
            if now - due > PRE_BUFFER * self.dt:  # 斷糧太久：裝置緩衝已經播空，重新對齊
                self.play_start = now - k * self.dt
                due = now
            if due > now:
                await asyncio.sleep(due - now)
            if gen != self.gen:  # 等待期間被打斷了
                continue
            try:
                await self._send(pkt)
            except Exception:
                return
            if self.first_sent_at is None:
                self.first_sent_at = time.monotonic()
            self.count += 1

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
