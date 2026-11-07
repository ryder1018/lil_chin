"""把 eval/voice_testset.yaml 的句子合成成 16kHz wav（模擬使用者說話）。

直接載入 Kokoro（server/tts.py），用男聲跟小晴的女聲區隔。已經存在的檔案不會覆蓋，
所以你可以把某些檔案換成自己的錄音（16kHz 單聲道 wav）。

  python scripts/make_voice_testset.py [--voice zm_010] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.tts import KokoroTTS  # noqa: E402
OUT = ROOT / "eval/voice/utterances"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="zm_010")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    items = yaml.safe_load((ROOT / "eval/voice_testset.yaml").read_text())["items"]
    tts = KokoroTTS(voice=args.voice, speed=args.speed)
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        for k, text in enumerate(it["turns"]):
            path = OUT / f"{it['id']}_{k}.wav"
            if path.exists() and not args.force:
                continue
            audio = resample_poly(tts.synthesize(text), 2, 3)  # 24k → 16k
            # 前後各留一點靜音，比較像真的講話
            pad = np.zeros(int(16000 * 0.3), dtype=np.float32)
            sf.write(path, np.concatenate([pad, audio, pad]), 16000, subtype="PCM_16")
            n += 1
    print(f"合成 {n} 個檔案到 {OUT}")


if __name__ == "__main__":
    main()
