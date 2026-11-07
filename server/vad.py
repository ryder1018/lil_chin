"""Silero VAD：判斷「有沒有人在說話」和「說完了沒」。

判斷邏輯參考自 xiaozhi-esp32-server 的 SileroVAD（MIT），見 THIRD_PARTY_NOTICES.md：
  - 每 512 個樣本（32ms）算一次語音機率
  - 雙門檻：≥ threshold 算有聲、≤ threshold_low 算沒聲，中間沿用上一個狀態（避免抖動）
  - 最近 5 個判斷裡有 ≥ 3 個有聲，才算「正在說話」
  - 正在說話之後，靜音超過 min_silence_ms 就算「說完了」
和原版不同：靜音長度用「音訊樣本數」算，不用牆上時鐘，所以網路抖動或測試時加速送音訊都不會誤判。
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import onnxruntime as ort
import silero_vad

CHUNK = 512  # 16kHz 下 32ms
CONTEXT = 64
SR = 16000


def _model_path() -> str:
    return os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")


class SileroVAD:
    """整個伺服器共用一個 ONNX session；每個連線各自有一個 VADStream 保存狀態。"""

    def __init__(self, threshold: float = 0.5, threshold_low: float = 0.3,
                 min_silence_ms: int = 500, min_speech_ms: int = 300):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"], sess_options=opts)
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.min_silence = min_silence_ms * SR // 1000
        self.min_speech = min_speech_ms * SR // 1000

    def stream(self) -> "VADStream":
        return VADStream(self)


class VADStream:
    def __init__(self, vad: SileroVAD):
        self.vad = vad
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)
        self.buf = b""
        self.window: deque[bool] = deque(maxlen=5)
        self.last_is_voice = False
        self.speaking = False        # 目前判定在說話
        self.samples = 0             # 已處理的樣本數（音訊時間）
        self.speech_start = 0
        self.last_voice = 0

    def _prob(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self.context, chunk.reshape(1, -1)], axis=1).astype(np.float32)
        out, self.state = self.vad.session.run(
            None, {"input": x, "state": self.state, "sr": np.array(SR, dtype=np.int64)}
        )
        self.context = x[:, -CONTEXT:]
        return float(out.item())

    def feed(self, pcm: bytes) -> str | None:
        """餵 16-bit PCM。回傳事件：'start'（開始說話）、'end'（說完了）、或 None。"""
        self.buf += pcm
        event = None
        while len(self.buf) >= CHUNK * 2:
            raw, self.buf = self.buf[: CHUNK * 2], self.buf[CHUNK * 2 :]
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            p = self._prob(chunk)
            self.samples += CHUNK
            if p >= self.vad.threshold:
                is_voice = True
            elif p <= self.vad.threshold_low:
                is_voice = False
            else:
                is_voice = self.last_is_voice
            self.last_is_voice = is_voice
            self.window.append(is_voice)
            have_voice = self.window.count(True) >= 3

            if have_voice:
                if not self.speaking:
                    self.speaking = True
                    self.speech_start = self.samples
                    event = event or "start"
                self.last_voice = self.samples
            elif self.speaking and self.samples - self.last_voice >= self.vad.min_silence:
                self.speaking = False
                if self.last_voice - self.speech_start >= self.vad.min_speech:
                    return "end"
        return event

    @property
    def silence_ms(self) -> int:
        return (self.samples - self.last_voice) * 1000 // SR
