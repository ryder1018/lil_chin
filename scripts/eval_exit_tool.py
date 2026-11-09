"""主模型在部署格式下，會不會自己呼叫 handle_exit_intent（不靠事後的道別分類）。

需要 llama-server 在跑：
  python scripts/eval_exit_tool.py

用 server/brain.py 的 Brain（人設 + 語音規則 + 時間 + 道別示範 + 工具），每句都是新對話。
題目沿用 eval_exit.py 的 POS（道別）/ NEG（不是道別）。沒呼叫工具的道別，實際部署時還有道別分類補上。
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import server.brain as B  # noqa: E402
from eval_exit import NEG, POS  # noqa: E402

URL = "http://127.0.0.1:8080/v1"


async def main() -> None:
    persona = (ROOT / "prompts/girlfriend_v4.md").read_text()
    rules = (ROOT / "prompts/voice_rules.md").read_text()
    hits = {"pos": 0, "neg": 0}
    for kind, lst in [("pos", POS), ("neg", NEG)]:
        for u in lst:
            b = B.Brain(URL, persona, rules, "掰掰")
            rec = B.Reply(user=u)
            async for _ in b.reply(u, rec):
                pass
            hit = any(c["name"] == "handle_exit_intent" for c in rec.tool_calls)
            hits[kind] += hit
            if (kind == "pos") != hit:
                print(f"{'沒呼叫' if kind == 'pos' else '誤呼叫'}  {u} → {''.join(rec.spoken)}")
    print(f"道別時自己呼叫工具 {hits['pos']}/{len(POS)}，不是道別卻呼叫 {hits['neg']}/{len(NEG)}")


if __name__ == "__main__":
    asyncio.run(main())
