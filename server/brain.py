"""機器人的大腦：組 prompt、管對話歷史、串流呼叫 llama-server、切句清理、處理工具。

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
        "description": "對方要道別、去睡覺、結束聊天時，先用一句話溫柔道別，再呼叫這個工具結束這次對話",
        "parameters": {"type": "object", "properties": {}},
    },
}
TOOLS = [EXIT_TOOL]

FALLBACK_REPLY = "嗯？我剛剛恍神了，可以再說一次嗎？"

# 道別偵測：4B 模型道別時常常只說「晚安」卻忘了呼叫工具（實測 24 次只叫 7 次），
# 所以沒呼叫工具時，再用獨立的分類 prompt（prompts/exit_check.md）問一次「對方是不是要走了」。
#   - 不帶人設：帶著人設問的話她很愛回答「是」（誤判 74/195）；獨立 prompt 誤判 1–2/71、道別 15/15（scripts/eval_exit.py）。
#   - 在她開口之後才跑（約 100ms），不影響延遲。
#   - llama-server 的 RAM prompt cache（--cache-ram，預設開）會把主對話的 KV cache 還原，下一輪照樣命中。
EXIT_GRAMMAR = 'root ::= "是" | "否"'

# 示範用的道別（和 config 的 goodbye_reply 不同句：示範句和預設句一樣的話，模型會直接照抄而不呼叫工具）
FEWSHOT_BYES = [("掰掰", "掰掰，路上小心喔。"), ("好睏，我要去睡了", "晚安，好好休息喔。")]


def fewshot() -> list[dict]:
    """示範「道別 → 先說道別語 → 再呼叫工具」。小模型沒看過示範的話很少主動呼叫工具。
    （做法參考 xiaozhi-esp32-server 的 few-shot 注入；只放在這次請求，不進對話歷史）

    道別語放在一般文字（content）而不是工具參數裡：文字可以逐句串流，第一句馬上就能送 TTS；
    工具參數要等整段 JSON 生完才拿得到，會多等 ~400ms。
    """
    msgs: list[dict] = []
    for i, (user, bye) in enumerate(FEWSHOT_BYES):
        call_id = f"fewshot_exit_{i}"
        msgs += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": bye, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "handle_exit_intent", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": call_id, "content": "已結束對話"},
        ]
    return msgs


@dataclass
class Reply:
    """一輪回覆的紀錄（給 log 和評估用）。"""
    user: str
    raw: str = ""
    spoken: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    exit: bool = False
    exit_by: str | None = None  # "tool"（大腦自己呼叫工具）/ "check"（事後問出來的）
    ttft_ms: float | None = None
    first_segment_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None


class Brain:
    """一個連線一個 Brain（各自有對話歷史）。"""

    def __init__(self, url: str, persona: str, voice_rules: str, goodbye: str,
                 max_tokens: int = 300, history_turns: int = 10, client: httpx.AsyncClient | None = None,
                 exit_check: str | None = None):
        self.url = url.rstrip("/")
        self.persona = persona.strip()
        self.voice_rules = voice_rules.strip()
        self.goodbye = goodbye
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.history: list[dict] = []
        self.exit_check = exit_check.strip() if exit_check else None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    def system_prompt(self, now: datetime | None = None) -> str:
        now = now or datetime.now()
        rules = self.voice_rules.format(date=f"{now:%Y-%m-%d}", weekday=WEEKDAYS[now.weekday()], time=f"{now:%H:%M}")
        # 時間放最後：每分鐘會變，放後面才不會讓前面的 prefix cache 失效
        return f"{self.persona}\n\n{rules}"

    def messages(self, user: str) -> list[dict]:
        hist = self.history[-self.history_turns * 2 :]
        return [{"role": "system", "content": self.system_prompt()}, *fewshot(), *hist,
                {"role": "user", "content": user}]

    async def prewarm(self) -> None:
        """使用者開始說話時先打一個「只生成 1 個 token」的請求給 llama-server。

        兩個效果：
        1. 喚醒 GPU：WSL2 下 GPU 閒置幾秒後，第一次推論會慢 ~200ms；在 llama-server 自己的程序裡喚醒才有用
           （在別的程序跑計算反而會跟它搶 GPU 時間片）。
        2. 先把 [system + 示範 + 歷史] 算進 KV cache：真正的問題來時只需要算新的那一句。
        """
        hist = self.history[-self.history_turns * 2 :]
        msgs = [{"role": "system", "content": self.system_prompt()}, *fewshot(), *hist,
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
                rec.exit, rec.exit_by = True, "tool"
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
        said = "".join(rec.spoken)
        if not rec.exit and said != FALLBACK_REPLY and await self.wants_exit(user):
            rec.exit, rec.exit_by = True, "check"

        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": said}]

    async def wants_exit(self, user: str) -> bool:
        """問分類 prompt：對方這句話是不是要結束聊天？（沒設定 exit_check 就只靠工具呼叫）"""
        if not self.exit_check:
            return False
        msgs = [{"role": "system", "content": self.exit_check}, {"role": "user", "content": user}]
        try:
            r = await self.client.post(f"{self.url}/chat/completions",
                                       json={"messages": msgs, "max_tokens": 1, "temperature": 0, "grammar": EXIT_GRAMMAR})
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") == "是"
        except Exception:
            return False
