"""機器人後端入口：WebSocket（裝置對話）+ HTTP（OTA、瀏覽器假裝置網頁）。

  python -m server                 # 用 server/config.yaml
  python -m server --config x.yaml

模型（VAD、ASR、TTS）啟動時載入一次，所有連線共用；大腦（llama-server）另外啟動。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx
from aiohttp import web
from websockets.asyncio.server import ServerConnection, serve

from .asr import SenseVoiceASR
from .brain import Brain
from .config import ROOT, load_config, resolve
from .session import Session
from .tts import make_tts
from .vad import SileroVAD

WS_PATH = "/xiaozhi/v1/"
log = logging.getLogger("robot")


class Context:
    """整個伺服器共用的東西：設定、模型、執行緒池。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = log
        t = time.perf_counter()
        v = cfg["vad"]
        self.vad = SileroVAD(v["threshold"], v["threshold_low"], v["min_silence_ms"], v.get("min_speech_ms", 300))
        log.info("載入 ASR（SenseVoice）…")
        if cfg["asr"]["device"] == "cpu":
            import torch
            torch.set_num_threads(cfg["asr"].get("threads", 4))
        self.asr = SenseVoiceASR(str(resolve(cfg["asr"]["model_dir"])), cfg["asr"]["language"], cfg["asr"]["device"])
        log.info(f"載入 TTS（{cfg['tts']['engine']}，聲音 {cfg['tts']['voice']}）…")
        self.tts = make_tts(cfg["tts"])
        # GPU 上的模型一次跑一個請求就好
        self.asr_pool = ThreadPoolExecutor(1, thread_name_prefix="asr")
        self.tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        ch = cfg["character"]
        self.persona = resolve(ch["persona"]).read_text(encoding="utf-8")
        self.voice_rules = resolve(ch["voice_rules"]).read_text(encoding="utf-8")
        self.exit_check = resolve(ch["exit_check"]).read_text(encoding="utf-8") if ch.get("exit_check") else None
        self.conv_dir = resolve(cfg["logging"]["conversation_dir"])
        log.info(f"模型載入完成（{time.perf_counter() - t:.1f} 秒）")

    def new_brain(self) -> Brain:
        c = self.cfg
        return Brain(c["llm"]["url"], self.persona, self.voice_rules, c["character"]["goodbye_reply"],
                     c["llm"]["max_tokens"], c["llm"]["history_turns"], self.http,
                     exit_check=self.exit_check)


# ---------- WebSocket ----------
async def handle_ws(ws: ServerConnection, ctx: Context) -> None:
    path = ws.request.path
    if not urlparse(path).path.rstrip("/").startswith(WS_PATH.rstrip("/")):
        await ws.close(1008, "wrong path")
        return
    q = {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}
    h = ws.request.headers
    device_id = h.get("device-id") or q.get("device-id") or "unknown"
    client_id = h.get("client-id") or q.get("client-id") or device_id
    log.info(f"新連線 {ws.remote_address[0]} device={device_id}")
    await Session(ws, ctx, device_id, client_id).run()


# ---------- HTTP：OTA + 網頁 ----------
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "client-id, content-type, device-id, authorization",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def make_http(ctx: Context) -> web.Application:
    app = web.Application()
    web_dir = ROOT / "web"

    async def ota(request: web.Request) -> web.Response:
        """小智韌體開機會先問 OTA：拿到 WebSocket 位址後再連線。"""
        if request.method == "OPTIONS":
            return web.Response(headers=CORS)
        if request.method == "GET":
            return web.Response(text="OTA 正常。WebSocket：" + ctx.cfg["server"]["public_ws_url"], headers=CORS)
        return web.json_response({
            "server_time": {"timestamp": int(time.time() * 1000), "timezone_offset": 480},
            "firmware": {"version": "0.0.0", "url": ""},
            "websocket": {"url": ctx.cfg["server"]["public_ws_url"], "token": ""},
        }, headers=CORS)

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(web_dir / "index.html")

    app.router.add_route("*", "/xiaozhi/ota/", ota)
    app.router.add_get("/", index)
    app.router.add_static("/web/", web_dir)
    return app


async def main_async(cfg: dict) -> None:
    ctx = Context(cfg)
    s = cfg["server"]
    runner = web.AppRunner(make_http(ctx))
    await runner.setup()
    await web.TCPSite(runner, s["host"], s["http_port"]).start()
    async with serve(lambda ws: handle_ws(ws, ctx), s["host"], s["ws_port"], max_size=None, ping_interval=None):
        log.info(f"✅ 小魚上線：網頁 http://localhost:{s['http_port']}/  裝置 ws://<本機IP>:{s['ws_port']}{WS_PATH}")
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "websockets", "aiohttp.access", "jieba", "modelscope", "funasr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main_async(load_config(args.config)))


if __name__ == "__main__":
    main()
