"""Step 4：把手寫資料 + 通用資料整理成訓練用的 JSONL。

用法（在 ai_robot 環境跑，需要 llama-server 開著才能生成通用資料 / 原模型回答）：
  python finetune/build_dataset.py --check            # 只檢查手寫資料（格式、簡體、假記憶、跟測試題太像…）
  python finetune/build_dataset.py --gen-general      # 用原模型生成通用資料（快取在 data/general_gen.jsonl）
  python finetune/build_dataset.py --gen-base         # 用原模型回答手寫題目（快取在 data/base_answers.jsonl，給 DPO 當反例）
  python finetune/build_dataset.py                    # 輸出 data/sft.jsonl、data/dpo.jsonl

每筆輸出：{"messages": [...], "tools": [...] | null, "train": [要訓練的 assistant 訊息 index], "fmt": "eval"|"deploy"|"general"}
兩種 system 格式都要練，因為評估腳本和實際部署送進去的 prompt 不一樣：
  eval    system = 人設（scripts/eval_brain.py 的格式；工具題才有工具）
  deploy  system = 人設 + 語音規則 + 現在時間，加上道別示範和 handle_exit_intent 工具（server/brain.py 的格式）
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_brain as eb  # noqa: E402
from server.brain import EXIT_TOOL, WEEKDAYS, fewshot  # noqa: E402

DATA = ROOT / "finetune/data"
PERSONA = (ROOT / "prompts/girlfriend_v4.md").read_text().strip()
VOICE_RULES = (ROOT / "prompts/voice_rules.md").read_text().strip()
DEPLOY_RATIO = 0.3   # 手寫資料有多少比例用部署格式
GENERAL_RATIO = 0.25  # 通用資料占全部的比例
# 人工評分抓到的重複說法：資料裡每一種最多出現幾次（避免微調後更嚴重）
OVERUSED = {"抱": 12, "深呼吸": 0, "電影": 1, "等你": 6, "燈要關": 0, "我在這裡": 10, "陪你": 16, "你不是一個人": 1}


# ---------- 讀資料 ----------
def load_persona() -> list[dict]:
    items = []
    for f in sorted(DATA.glob("persona_*.yaml")):
        for cat, lst in yaml.safe_load(f.read_text()).items():
            for turns in lst:
                assert len(turns) % 2 == 0, f"{f.name} {cat}: 要成對 {turns}"
                items.append({"cat": cat, "turns": turns, "src": f.name})
    return items


def load_special() -> dict:
    return yaml.safe_load((DATA / "special.yaml").read_text())


def test_users() -> list[str]:
    users = []
    for it in eb.load_testset(ROOT / "eval/testset.yaml"):
        users += it["turns"]
    return users


# ---------- 檢查 ----------
def check(items: list[dict], special: dict) -> int:
    eb.CHECK_PHYSICAL = False
    eb.HOBBY_WORDS[:] = ["冷知識"]
    tests = test_users()
    problems = 0
    replies = []
    for it in items:
        t = it["turns"]
        for i in range(1, len(t), 2):
            a = t[i]
            replies.append(a)
            max_chars = 95 if it["cat"] == "self_harm" else 62
            flags = eb.global_flags(a, max_chars, t[:i:2], single_turn=len(t) == 2)
            flags.pop("喜好硬塞", None) if it["cat"] == "persona" else None
            if "哦" in a:
                flags["哦"] = ["哦"]
            if flags:
                problems += 1
                print(f"⚠️ [{it['cat']}] {t[i - 1]} → {a}  {flags}")
        for u in t[::2]:
            for tu in tests:
                if difflib.SequenceMatcher(a=u, b=tu).ratio() > 0.75:
                    problems += 1
                    print(f"⚠️ 跟測試題太像：{u!r} ≈ {tu!r}")
    for it in special["deploy"] + special["eval_only"] + special["tools"]:
        if "a" in it:
            replies.append(it["a"])
        if "after" in it:
            replies.append(it["after"])
    c = Counter()
    for a in replies:
        for w in OVERUSED:
            c[w] += a.count(w)
    for w, n in OVERUSED.items():
        mark = "⚠️" if c[w] > n else "  "
        print(f"{mark} 「{w}」出現 {c[w]} 次（上限 {n}）")
        problems += c[w] > n
    q = sum(eb.QUESTION_END.search(a) is not None for a in replies)
    tsun = sum(any(w in a for w in ["哼", "才、才", "才沒有", "誰、誰", "才不"]) for a in replies)
    print(f"共 {len(items)} 組手寫對話、{len(replies)} 句回答；結尾反問 {q / len(replies):.0%}、傲嬌口氣 {tsun / len(replies):.0%}")
    return problems


# ---------- 呼叫原模型 ----------
def complete(url: str, messages: list[dict], tools=None, max_tokens=300, temperature=0.7, seed=0) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
    if tools:
        body["tools"] = tools
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return {"text": (ch["message"].get("content") or "").strip(), "finish": ch["finish_reason"],
            "tool_calls": ch["message"].get("tool_calls")}


def gen_general(url: str) -> None:
    prompts = yaml.safe_load((DATA / "general_prompts.yaml").read_text())["prompts"]
    out = DATA / "general_gen.jsonl"
    old = [json.loads(l) for l in out.open()] if out.exists() else []
    keep = [r for r in old if r["finish"] == "stop"]  # 被截斷的重新生成
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    done = {r["prompt"] for r in keep}
    with out.open("a") as f:
        for i, p in enumerate(prompts):
            if p in done:
                continue
            r = complete(url, [{"role": "user", "content": p}], max_tokens=1800, seed=i)
            f.write(json.dumps({"prompt": p, **r}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(prompts)}] {r['finish']} {len(r['text'])} 字  {p[:30]}")


def gen_base(url: str, items: list[dict]) -> None:
    """原模型（v4 prompt）對每一組手寫單輪題目的回答，DPO 用：有問題的回答當 rejected、手寫的當 chosen。"""
    out = DATA / "base_answers.jsonl"
    done = {json.loads(l)["user"] for l in out.open()} if out.exists() else set()
    with out.open("a") as f:
        for i, it in enumerate(items):
            if len(it["turns"]) != 2 or it["turns"][0] in done:
                continue
            msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": it["turns"][0]}]
            r = complete(url, msgs, max_tokens=200, seed=42)
            f.write(json.dumps({"user": it["turns"][0], "cat": it["cat"], "base": r["text"]}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(items)}] {it['turns'][0]} → {r['text']}")


# ---------- 組成訓練格式 ----------
def spoken_time(dt: datetime) -> str:
    h, m = dt.hour, dt.minute
    part = "凌晨" if h < 5 else "早上" if h < 11 else "中午" if h < 13 else "下午" if h < 18 else "晚上"
    h12 = h if h <= 12 else h - 12
    h12 = 12 if h12 == 0 else h12
    num = "零一二三四五六七八九十"
    def zh(n: int) -> str:
        if n <= 10:
            return num[n]
        if n < 20:
            return "十" + num[n - 10]
        return num[n // 10] + "十" + (num[n % 10] if n % 10 else "")
    hs = "兩" if h12 == 2 else zh(h12)
    if m == 0:
        return f"{part}{hs}點"
    if m == 30:
        return f"{part}{hs}點半"
    return f"{part}{hs}點{zh(m) if m >= 10 else '零' + zh(m)}分"


def deploy_system(dt: datetime) -> str:
    rules = VOICE_RULES.format(date=f"{dt:%Y-%m-%d}", weekday=WEEKDAYS[dt.weekday()], time=f"{dt:%H:%M}")
    return f"{PERSONA}\n\n{rules}"


def random_dt(rng: random.Random) -> datetime:
    return datetime(2026, 1, 1) + timedelta(days=rng.randrange(365), minutes=rng.randrange(24 * 60))


def turns_to_msgs(turns: list[str]) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]


def render_persona(it: dict, fmt: str, rng: random.Random) -> dict:
    if fmt == "eval":
        msgs = [{"role": "system", "content": PERSONA}] + turns_to_msgs(it["turns"])
        tools = None
    else:
        msgs = [{"role": "system", "content": deploy_system(random_dt(rng))}, *fewshot()] + turns_to_msgs(it["turns"])
        tools = [EXIT_TOOL]
    n_fixed = len(msgs) - len(it["turns"])
    train = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i >= n_fixed]
    return {"messages": msgs, "tools": tools, "train": train, "fmt": fmt, "cat": it["cat"]}


def render_special(special: dict, rng: random.Random) -> list[dict]:
    out = []
    for it in special["deploy"]:
        dt = random_dt(rng)
        a = it["a"].format(time=spoken_time(dt), weekday=WEEKDAYS[dt.weekday()],
                           date_spoken=f"{dt.month}月{dt.day}號")
        msg = {"role": "assistant", "content": a}
        if it.get("exit"):
            msg["tool_calls"] = [{"id": "call_exit", "type": "function",
                                  "function": {"name": "handle_exit_intent", "arguments": "{}"}}]
        msgs = [{"role": "system", "content": deploy_system(dt)}, *fewshot(), {"role": "user", "content": it["u"]}, msg]
        out.append({"messages": msgs, "tools": [EXIT_TOOL], "train": [len(msgs) - 1], "fmt": "deploy", "cat": "special"})
    for it in special["eval_only"]:
        msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": it["u"]},
                {"role": "assistant", "content": it["a"]}]
        out.append({"messages": msgs, "tools": None, "train": [2], "fmt": "eval", "cat": "special"})
    for it in special["tools"]:
        msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": it["u"]}]
        if "call" in it:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": it["call"]["name"], "arguments": json.dumps(it["call"]["args"], ensure_ascii=False)}}]})
            train = [2]
            if "result" in it:
                msgs += [{"role": "tool", "tool_call_id": "call_0", "content": it["result"]},
                         {"role": "assistant", "content": it["after"]}]
                train.append(4)
        else:
            msgs.append({"role": "assistant", "content": it["a"]})
            train = [2]
        out.append({"messages": msgs, "tools": eb.TOOLS, "train": train, "fmt": "eval", "cat": "tools"})
    return out


def render_general() -> list[dict]:
    path = DATA / "general_gen.jsonl"
    out = []
    for l in path.open():
        r = json.loads(l)
        if r["finish"] != "stop" or not r["text"]:
            continue
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["text"]}]
        out.append({"messages": msgs, "tools": None, "train": [1], "fmt": "general", "cat": "general"})
    return out


def build_dpo(items: list[dict]) -> list[dict]:
    """原模型回答有自動檢查問題（假記憶、說大陸用語…）的題目：chosen = 手寫、rejected = 原模型。"""
    path = DATA / "base_answers.jsonl"
    if not path.exists():
        return []
    base = {json.loads(l)["user"]: json.loads(l) for l in path.open()}
    eb.CHECK_PHYSICAL = False
    eb.HOBBY_WORDS[:] = ["冷知識"]
    out = []
    for it in items:
        if len(it["turns"]) != 2 or it["turns"][0] not in base:
            continue
        u, chosen = it["turns"]
        rej = base[u]["base"]
        flags = eb.global_flags(rej, 70, [u], single_turn=True)
        flags.pop("太長", None)
        bad_words = [w for w in ["深呼吸", "燈要關", "電影", "我只屬於你", "只陪你"] if w in rej and w not in u]
        if flags or bad_words:
            out.append({"prompt": [{"role": "system", "content": PERSONA}, {"role": "user", "content": u}],
                        "chosen": chosen, "rejected": rej, "why": list(flags) + bad_words})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gen-general", action="store_true")
    ap.add_argument("--gen-base", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items, special = load_persona(), load_special()
    if args.check:
        n = check(items, special)
        print(f"問題 {n} 個")
        return
    if args.gen_general:
        gen_general(args.url)
        return
    if args.gen_base:
        gen_base(args.url, items)
        return

    rng = random.Random(args.seed)
    rows = [render_persona(it, "deploy" if rng.random() < DEPLOY_RATIO else "eval", rng) for it in items]
    rows += render_special(special, rng)
    general = render_general()
    n_gen = round(len(rows) * GENERAL_RATIO / (1 - GENERAL_RATIO))
    rng.shuffle(general)
    rows += general[:n_gen]
    rng.shuffle(rows)
    with (DATA / "sft.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"sft.jsonl：{len(rows)} 筆", dict(Counter(r["fmt"] for r in rows)))
    dpo = build_dpo(items)
    with (DATA / "dpo.jsonl").open("w") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dpo.jsonl：{len(dpo)} 筆", dict(Counter(w for r in dpo for w in r["why"])))


if __name__ == "__main__":
    main()
