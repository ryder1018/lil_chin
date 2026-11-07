"""小晴的大腦：組 prompt、管對話歷史、串流呼叫 llama-server、切句清理、處理工具。

  使用者的話（ASR，可能是簡體）
    → 轉台灣繁體
    → [system: 人設 + 語音規則 + 現在時間] + [工具示範] + [最近 N 輪對話] + 這一句
    → llama-server 串流
    → Segmenter 切句 → text_filter 清理 → 一句一句交給 TTS
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx

from .segmenter import Segmenter
from .text_filter import for_speech, to_taiwan

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

EXIT_TOOL = {
    "type": "function",
    "function": {
        "name": "handle_exit_intent",
        "description": "對方要道別、去睡覺、結束聊天時呼叫，會在道別後結束這次對話",
        "parameters": {
            "type": "object",
            "properties": {"say_goodbye": {"type": "string", "description": "溫柔的道別語，一句話"}},
            "required": ["say_goodbye"],
        },
    },
}
TOOLS = [EXIT_TOOL]

FALLBACK_REPLY = "嗯？我剛剛恍神了，可以再說一次嗎？"


def fewshot(goodbye: str) -> list[dict]:
    """示範一次「道別 → 呼叫工具」。小模型沒看過示範的話很少主動呼叫工具。
    （做法參考 xiaozhi-esp32-server 的 few-shot 注入；只放在這次請求，不進對話歷史）"""
    return [
        {"role": "user", "content": "掰掰"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "fewshot_exit", "type": "function",
            "function": {"name": "handle_exit_intent", "arguments": json.dumps({"say_goodbye": goodbye}, ensure_ascii=False)},
        }]},
        {"role": "tool", "tool_call_id": "fewshot_exit", "content": "已結束對話"},
        {"role": "assistant", "content": goodbye},
    ]


@dataclass
class Reply:
    """一輪回覆的紀錄（給 log 和評估用）。"""
    user: str
    raw: str = ""
    spoken: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    exit: bool = False
    ttft_ms: float | None = None
    first_segment_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None


class Brain:
    """一個連線一個 Brain（各自有對話歷史）。"""

    def __init__(self, url: str, persona: str, voice_rules: str, goodbye: str,
                 max_tokens: int = 300, history_turns: int = 10, client: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/")
        self.persona = persona.strip()
        self.voice_rules = voice_rules.strip()
        self.goodbye = goodbye
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.history: list[dict] = []
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    def system_prompt(self, now: datetime | None = None) -> str:
        now = now or datetime.now()
        rules = self.voice_rules.format(date=f"{now:%Y-%m-%d}", weekday=WEEKDAYS[now.weekday()], time=f"{now:%H:%M}")
        # 時間放最後：每分鐘會變，放後面才不會讓前面的 prefix cache 失效
        return f"{self.persona}\n\n{rules}"

    def messages(self, user: str) -> list[dict]:
        hist = self.history[-self.history_turns * 2 :]
        return [{"role": "system", "content": self.system_prompt()}, *fewshot(self.goodbye), *hist,
                {"role": "user", "content": user}]

    async def prewarm(self) -> None:
        """使用者開始說話時先打一個「只生成 1 個 token」的請求給 llama-server。

        兩個效果：
        1. 喚醒 GPU：WSL2 下 GPU 閒置幾秒後，第一次推論會慢 ~200ms；在 llama-server 自己的程序裡喚醒才有用
           （在別的程序跑計算反而會跟它搶 GPU 時間片）。
        2. 先把 [system + 示範 + 歷史] 算進 KV cache：真正的問題來時只需要算新的那一句。
        """
        hist = self.history[-self.history_turns * 2 :]
        msgs = [{"role": "system", "content": self.system_prompt()}, *fewshot(self.goodbye), *hist,
                {"role": "user", "content": "嗯"}]
        try:
            r = await self.client.post(f"{self.url}/chat/completions",
                                       json={"messages": msgs, "max_tokens": 1, "tools": TOOLS})
            r.raise_for_status()
        except Exception:
            pass

    async def reply(self, user_text: str, rec: Reply | None = None) -> AsyncIterator[str]:
        """串流產生「可以唸的句子」。結束後 rec 裡有完整紀錄。"""
        user = to_taiwan(user_text)
        rec = rec if rec is not None else Reply(user=user)
        rec.user = user
        body = {"messages": self.messages(user), "stream": True, "max_tokens": self.max_tokens, "tools": TOOLS}
        seg = Segmenter()
        calls: dict[int, dict] = {}
        t0 = time.perf_counter()

        def emit(pieces: list[str]):
            for p in pieces:
                clean = for_speech(p)
                if clean:
                    if rec.first_segment_ms is None:
                        rec.first_segment_ms = (time.perf_counter() - t0) * 1000
                    rec.spoken.append(clean)
                    yield clean

        async with self.client.stream("POST", f"{self.url}/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if t := chunk.get("timings"):
                    rec.prompt_tokens, rec.cached_tokens = t.get("prompt_n"), t.get("cache_n")
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if piece := delta.get("content"):
                        if rec.ttft_ms is None:
                            rec.ttft_ms = (time.perf_counter() - t0) * 1000
                        rec.raw += piece
                        for s in emit(seg.feed(piece)):
                            yield s
        # 最後一段如果是逗號結尾（「～」轉來的），改成句號
        for s in emit([t.rstrip().rstrip("，,、～~") + "。" if t.rstrip()[-1:] in "，,、～~" else t for t in seg.flush()]):
            yield s

        rec.tool_calls = list(calls.values())
        for c in rec.tool_calls:
            if c["name"] == "handle_exit_intent":
                rec.exit = True
                if not rec.spoken:
                    try:
                        bye = json.loads(c["arguments"] or "{}").get("say_goodbye") or self.goodbye
                    except json.JSONDecodeError:
                        bye = self.goodbye
                    for s in emit([bye]):
                        yield s
        if not rec.spoken:
            for s in emit([FALLBACK_REPLY]):
                yield s
        rec.total_ms = (time.perf_counter() - t0) * 1000

        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": "".join(rec.spoken)}]
