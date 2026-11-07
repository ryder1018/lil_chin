"""語音辨識：SenseVoice-Small（FunASR）。

SenseVoice 的輸出長這樣：<|zh|><|SAD|><|Speech|><|withitn|>我今天好累。
前面的標籤是語種、情緒、聲音事件，這裡拆出來：文字給大腦、情緒之後可以拿來調整回應語氣。
"""
from __future__ import annotations

import contextlib
import io
import re
import time
from dataclasses import dataclass

TAG_RE = re.compile(r"<\|([^|]+)\|>")
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
# SenseVoice 偶爾在中文字之間多吐空格
CJK_SPACE = re.compile(r"(?<=[　-〿一-鿿！-｠])\s+(?=[　-〿一-鿿！-｠])")


@dataclass
class ASRResult:
    text: str
    language: str | None
    emotion: str | None
    event: str | None
    ms: float


def parse(raw: str) -> tuple[str, str | None, str | None, str | None]:
    tags = TAG_RE.findall(raw)
    text = CJK_SPACE.sub("", TAG_RE.sub("", raw)).strip()
    lang = tags[0] if tags else None
    emo = next((t for t in tags if t in EMOTIONS), None)
    event = next((t for t in tags if t in {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Cough", "Breath"}), None)
    return text, lang, emo, event


class SenseVoiceASR:
    def __init__(self, model_dir: str, language: str = "zh", device: str = "cuda:0"):
        from funasr import AutoModel  # 載入很慢，放在這裡

        self.language = language
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = AutoModel(model=model_dir, disable_update=True, device=device)
        self.transcribe(b"\x00\x00" * 16000)  # 暖機

    def transcribe(self, pcm: bytes) -> ASRResult:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            r = self.model.generate(input=pcm, cache={}, language=self.language, use_itn=True, batch_size_s=60)
        text, lang, emo, event = parse(r[0]["text"] if r else "")
        return ASRResult(text, lang, emo, event, (time.perf_counter() - t0) * 1000)
