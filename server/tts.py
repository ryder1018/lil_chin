"""語音合成。目前引擎：Kokoro-82M-v1.1-zh。

換引擎（例如之後的 BreezyVoice）時，只要寫一個有同樣 synthesize() 的類別，在 make_tts() 加一個分支。
"""
from __future__ import annotations

import threading
import time

import numpy as np

KOKORO_REPO = "hexgrad/Kokoro-82M-v1.1-zh"


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, voice: str = "zf_017", speed: float = 1.1, device: str = "cuda"):
        import opencc
        import torch
        from kokoro import KModel, KPipeline

        self.torch = torch
        self.voice_name = voice
        self.speed = speed
        self.model = KModel(repo_id=KOKORO_REPO).to(device).eval()
        self.pipe = KPipeline(lang_code="z", repo_id=KOKORO_REPO, model=self.model)
        # misaki 的中文 G2P（jieba 斷詞 + pypinyin）是用簡體語料建的，先轉簡體再算讀音比較準
        self.t2s = opencc.OpenCC("tw2s")
        self.lock = threading.Lock()
        self.voices: dict[str, object] = {}
        self.synthesize("你好")  # 暖機

    def _voice(self, name: str):
        if name not in self.voices:
            self.voices[name] = self.pipe.load_voice(name)
        return self.voices[name]

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> np.ndarray:
        """文字 → float32 音訊（24kHz）。"""
        text = self.t2s.convert(text)
        with self.lock, self.torch.inference_mode():
            parts = [r.audio for r in self.pipe(text, voice=self._voice(voice or self.voice_name),
                                                speed=speed or self.speed) if r.audio is not None]
        if not parts:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        return self.torch.cat(parts).cpu().numpy()


def make_tts(cfg: dict):
    engine = cfg.get("engine", "kokoro")
    if engine == "kokoro":
        return KokoroTTS(cfg.get("voice", "zf_017"), cfg.get("speed", 1.1), cfg.get("device", "cuda"))
    raise ValueError(f"不支援的 TTS 引擎：{engine}")


if __name__ == "__main__":
    tts = KokoroTTS()
    t = time.perf_counter()
    a = tts.synthesize("辛苦你了，有我在，想說什麼都可以慢慢說。")
    print(f"{(time.perf_counter() - t) * 1000:.0f} ms → {len(a) / 24000:.2f} 秒音訊")
