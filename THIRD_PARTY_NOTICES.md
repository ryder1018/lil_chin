# 第三方聲明

## xiaozhi-esp32-server

本專案的後端（`server/`）是自己寫的，但下列部分參考或移植自
[xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server)（MIT License）：

| 本專案檔案 | 參考的內容 |
|---|---|
| `server/protocol.py` | 裝置 ↔ 伺服器的 WebSocket 訊息格式（hello / listen / stt / tts / abort），為了和小智 ESP32 韌體相容 |
| `server/vad.py` | Silero VAD 的雙門檻 + 5 格滑動窗口判斷（改成用音訊樣本數計算靜音長度） |
| `server/segmenter.py` | 串流切句規則：第一句遇到逗號就切、之後遇到句末標點才切 |
| `server/audio.py` | 音訊封包依播放速度送出的概念（AudioRateController） |
| `server/brain.py` | 用 few-shot 示範讓小模型學會呼叫工具的做法 |

xiaozhi-esp32-server 的原始碼沒有被複製進本專案；開發時放在 `third_party/`（不納入版本控制）作為參考。

```
MIT License

Copyright (c) 2025 xinnan-tech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 使用的模型（不包含在本專案，需自行下載）

| 模型 | 用途 | 來源 |
|---|---|---|
| Qwen3-VL-4B-Instruct（GGUF） | 大腦 | Qwen/Qwen3-VL-4B-Instruct-GGUF |
| SenseVoice-Small | 語音辨識 | FunAudioLLM/SenseVoiceSmall |
| Silero VAD | 語音活動偵測 | silero-vad 套件內附 |
| Kokoro-82M-v1.1-zh | 語音合成 | hexgrad/Kokoro-82M-v1.1-zh |

各模型的授權以其原始發布頁為準。
