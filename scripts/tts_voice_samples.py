"""產生多個女聲的試聽檔，幫小晴挑聲音。

  python scripts/tts_voice_samples.py               # 預設挑 12 個聲音
  python scripts/tts_voice_samples.py --all         # 全部 55 個女聲

試聽檔在 eval/voice/voice_samples/，Windows 檔案總管可以開：
  \\\\wsl.localhost\\<你的發行版>\\home\\<使用者>\\try\\ai_robot\\eval\\voice\\voice_samples
選好後：改 server/config.yaml 的 tts.voice，然後 bash scripts/robot.sh restart server
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
from huggingface_hub import list_repo_files

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.tts import KOKORO_REPO, KokoroTTS  # noqa: E402
OUT = ROOT / "eval/voice/voice_samples"
TEXT = "你回來啦！今天辛苦了，有什麼想跟我說的嗎？我一直在這裡等你喔。"
DEFAULT = ["zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_017", "zf_018", "zf_019",
           "zf_021", "zf_022", "zf_023", "zf_024"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()

    voices = DEFAULT
    if args.all:
        voices = sorted(Path(f).stem for f in list_repo_files(KOKORO_REPO)
                        if f.startswith("voices/zf_"))
    OUT.mkdir(parents=True, exist_ok=True)
    tts = KokoroTTS()
    for v in voices:
        sf.write(OUT / f"{v}.wav", tts.synthesize(args.text, voice=v), tts.sample_rate)
    print(f"{len(voices)} 個試聽檔 → {OUT}")


if __name__ == "__main__":
    main()
