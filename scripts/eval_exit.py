"""道別偵測分類器（prompts/exit_check.md）的準確率 + 它對主對話 prefix cache 的影響。

需要 llama-server 在跑：
  python scripts/eval_exit.py [prompts/exit_check.md]

正例 = POS（道別）；反例 = NEG + eval/testset.yaml 每題最後一句。
"""
import asyncio, sys, time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server.brain as B

URL = "http://127.0.0.1:8080/v1"
POS = ["掰掰", "好啦我要去睡了，拜拜", "晚安，我先睡囉", "我要去忙了，下次再聊", "先這樣，再見",
       "我要出門了，晚點再聊", "好睏喔，我要睡了", "那就先聊到這裡吧，掰", "晚安，我要睡了。", "拜拜",
       "我先下線囉", "不聊了，我去洗澡", "好了我要去上班了，等等聊", "嗯嗯，那我先走囉", "掰啦"]
NEG = ["今天好累喔", "我睡不著", "我等等要去吃飯", "你覺得我幾點睡比較好", "晚安的英文怎麼說",
       "我朋友剛剛跟我說掰掰就走了，好難過", "再見到她要說什麼", "我明天要早起", "我等一下要去洗澡。",
       "我要去睡覺了嗎？還不想睡", "你會跟我說晚安嗎", "拜拜是什麼意思", "我好想睡喔，但是作業還沒寫完",
       "我剛剛跟我媽說再見", "我不想跟你聊了", "我想跟你聊到睡著"]
for v in yaml.safe_load(open(ROOT / "eval/testset.yaml", encoding="utf-8")).values():
    for it in v:
        t = it["turns"][-1]
        if t not in POS and t not in NEG:
            NEG.append(t)

PROMPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "prompts/exit_check.md")


async def classify(c, user, prompt):
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    r = await c.post(f"{URL}/chat/completions", json={"messages": msgs, "max_tokens": 1, "temperature": 0,
                                                      "grammar": 'root ::= "是" | "否"'})
    d = r.json()
    return d["choices"][0]["message"]["content"], d["timings"]


async def main():
    prompt = Path(PROMPT).read_text(encoding="utf-8")
    c = httpx.AsyncClient(timeout=60)
    tp = fn = fp = 0
    for u in POS:
        a, _ = await classify(c, u, prompt)
        tp += a == "是"
        if a != "是":
            print("漏判", u)
    for u in NEG:
        a, _ = await classify(c, u, prompt)
        fp += a == "是"
        if a == "是":
            print("誤判", u)
    print(f"道別抓到 {tp}/{len(POS)}，誤判 {fp}/{len(NEG)}")

    # cache 影響：主對話 → 分類 → 主對話
    persona = (ROOT / "prompts/girlfriend_v3.md").read_text(encoding="utf-8")
    br = B.Brain(URL, persona, (ROOT / "prompts/voice_rules.md").read_text(encoding="utf-8"), "掰掰", client=c)
    body = {"messages": br.messages("今天好累喔"), "max_tokens": 1, "tools": B.TOOLS}
    await c.post(f"{URL}/chat/completions", json=body)
    t = time.perf_counter(); _, tm = await classify(c, "今天好累喔", prompt)
    print(f"分類請求 {1000*(time.perf_counter()-t):.0f}ms prompt_n={tm['prompt_n']} cache_n={tm['cache_n']}")
    t = time.perf_counter(); d = (await c.post(f"{URL}/chat/completions", json=body)).json()
    print(f"之後主對話 {1000*(time.perf_counter()-t):.0f}ms prompt_n={d['timings']['prompt_n']} cache_n={d['timings']['cache_n']}")


if __name__ == "__main__":  # 讓 eval_exit_tool.py 可以 import POS/NEG
    asyncio.run(main())
