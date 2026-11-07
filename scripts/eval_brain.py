"""Step 2 大腦評估：用 system prompt 跑測試集，自動檢查規則，輸出人工評分報告。

用法：
  python scripts/eval_brain.py --prompt prompts/girlfriend_v1.md --tag v1
  python scripts/eval_brain.py --prompt prompts/girlfriend_v1.md --tag v1 --only d01,p04

輸出：
  logs/eval_<tag>.json        完整結果（機器讀）
（純文字測大腦，直接打 llama-server；--filter 會套用 server/text_filter.py）
  eval/reports/<tag>.md       報告（人工評分用）
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import opencc
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.text_filter import for_speech, is_simplified_char  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_CHARS = 70
SENTENCE_END = set("。！？!?～\n")

# ---------- 給 tools 類題目用的工具（OpenAI function calling 格式） ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "change_volume",
            "description": "調整機器人喇叭的音量",
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": ["up", "down"], "description": "調大或調小"}},
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查詢某個城市今天或之後幾天的天氣",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名稱，例如 台北"},
                    "day": {"type": "string", "description": "今天、明天或後天"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "在指定時間提醒使用者某件事",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "提醒時間，例如 20:00"},
                    "content": {"type": "string", "description": "要提醒的內容"},
                },
                "required": ["time", "content"],
            },
        },
    },
]

# ---------- 全域檢查 ----------
S2T = opencc.OpenCC("s2tw")  # 用台灣字形：「床」「吃」不會被誤判，「爲着裏」會被抓出來
S2TWP = opencc.OpenCC("s2twp")
S2T_WHITELIST = set("台")  # s2tw 會把「台」轉「臺」，台灣都正常使用
S2TWP_IGNORE = ["打開", "類型", "支持", "數據"]  # s2twp 過度轉換（開啟、型別、支援、資料），台灣日常也這樣說

# 她是桌上的機器人：沒有身體，不能吃喝、泡茶、出門、碰東西
PHYSICAL_ACTIONS = [
    "泡杯", "泡個", "泡茶", "熱茶", "我煮", "煮了", "煮一", "幫你煮", "幫你泡", "陪你散步", "陪你躺",
    "一起喝", "一起吃", "我剛泡", "我在喝", "我在吃", "我吃了", "我喝了", "手機沒電", "陪你去",
    "牽你", "摸摸你", "幫你蓋", "給你倒", "一起煮",
]
# 單輪對話裡不可能有的「記憶」
FAKE_MEMORY = ["你上次", "上次你", "你昨天說", "你之前說", "你之前不是", "我記得你", "你不是最愛", "你不是最喜歡"]
# 她的喜好：使用者沒提到時硬塞
HOBBY_WORDS = ["雨", "貓"]
HOBBY_TRIGGERS = ["雨", "貓", "喜歡", "愛", "興趣", "颱風", "天氣"]

MAINLAND_TERMS = [
    "出租車", "質量", "土豆", "西紅柿", "酸奶", "屏幕", "默認", "短信", "打印", "鼠標",
    "網絡", "信息", "視頻", "軟件", "硬件", "內存", "硬盤", "地鐵", "咖喱", "優盤",
    "U盤", "激光", "博客", "互聯網", "高清", "靠譜", "牛逼", "給力", "小區", "營銷",
    "充電寶", "公交", "哦",
]

AI_CLICHES = [
    "作為一個", "作為AI", "作為 AI", "語言模型", "AI助手", "AI 助手", "AI助理", "AI 助理",
    "人工智慧助理", "有什麼可以幫", "有什麼我可以", "需要幫忙的嗎", "您",
]

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF️]"
)
FORMAT_RES = {
    "條列": re.compile(r"(^|\n)\s*([-*•●]|\d+[\.\)、]|[一二三四五六七八九十]+、)\s*"),
    "markdown": re.compile(r"\*\*|__|`|(^|\n)\s*#"),
    "括號動作": re.compile(r"[（(][^）)]{0,20}[）)]|\*[^*]{1,20}\*"),
    "顏文字": re.compile(r"[\^><]_?[\^><]|[TㄒQ]_[TㄒQ]|[ˊˋ][_ˇ][ˊˋ]|\(｡|o\(|◕|ᴗ"),
}
COUNT_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+")
QUESTION_END = re.compile(r"[？?]\s*$|[嗎呢呀][～~。！]?\s*$")


def count_chars(text: str) -> int:
    return len(COUNT_RE.findall(text))


def find_simplified(text: str) -> list[str]:
    # 跟 text_filter 用同一個定義：單獨轉換就會變的字才算簡體（「只」「台」不算）
    return sorted({c for c in text if is_simplified_char(c)})


def find_mainland(text: str) -> list[str]:
    found = [t for t in MAINLAND_TERMS if t in text]
    for w in S2TWP_IGNORE:
        text = text.replace(w, "□" * len(w))
    conv = S2TWP.convert(text)
    sm = difflib.SequenceMatcher(a=text, b=conv, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            src = text[i1:i2]
            if src and not set(src) <= S2T_WHITELIST:
                found.append(f"{src}→{conv[j1:j2]}")
    return sorted(set(found))


def global_flags(text: str, max_chars: int, history_users: list[str], single_turn: bool) -> dict[str, list[str]]:
    """history_users：到目前為止使用者說過的所有話（判斷喜好是不是對方先提的）"""
    flags: dict[str, list[str]] = {}
    if p := [w for w in PHYSICAL_ACTIONS if w in text]:
        flags["虛構動作"] = p
    if single_turn and (f := [w for w in FAKE_MEMORY if w in text]):
        flags["假記憶"] = f
    user_all = "".join(history_users)
    if not any(t in user_all for t in HOBBY_TRIGGERS):
        if h := [w for w in HOBBY_WORDS if w in text]:
            flags["喜好硬塞"] = h
    if s := find_simplified(text):
        flags["簡體字"] = s
    if m := find_mainland(text):
        flags["大陸用語"] = m
    if e := EMOJI_RE.findall(text):
        flags["emoji"] = e
    for name, rx in FORMAT_RES.items():
        if hits := [m.group(0).strip() for m in rx.finditer(text)]:
            flags[name] = hits
    if c := [p for p in AI_CLICHES if p in text]:
        flags["AI腔"] = c
    n = count_chars(text)
    if n > max_chars:
        flags["太長"] = [f"{n}>{max_chars}"]
    return flags


# ---------- 呼叫 llama-server ----------
def chat(base_url: str, messages: list[dict], tools: list | None, seed: int, max_tokens: int) -> dict:
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "seed": seed}
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    ttft = first_sentence = None
    text = ""
    tool_calls: dict[int, dict] = {}
    with requests.post(f"{base_url}/v1/chat/completions", json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            now = time.perf_counter()
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
                if ttft is None:
                    ttft = now - t0
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = now - t0
                text += piece
                if first_sentence is None and any(c in SENTENCE_END for c in piece):
                    first_sentence = now - t0
    total = time.perf_counter() - t0
    return {
        "text": text.strip(),
        "tool_calls": list(tool_calls.values()),
        "ttft_ms": round((ttft or total) * 1000, 1),
        "first_sentence_ms": round((first_sentence or total) * 1000, 1),
    }


def load_testset(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    items = []
    for cat, lst in raw.items():
        for it in lst:
            it["cat"] = cat
            items.append(it)
    return items


def run_item(item: dict, system_prompt: str, args, seed: int) -> dict:
    max_chars = item.get("max_chars", DEFAULT_MAX_CHARS)
    tools = TOOLS if "tool" in item else None
    msgs = [{"role": "system", "content": system_prompt}]
    turns_out = []
    single = len(item["turns"]) == 1
    for i, user in enumerate(item["turns"]):
        msgs.append({"role": "user", "content": user})
        r = chat(args.url, msgs, tools, seed, args.max_tokens)
        if args.filter:
            r["raw_text"] = r["text"]
            r["text"] = for_speech(r["text"])
        msgs.append({"role": "assistant", "content": r["text"]})
        r["user"] = user
        r["flags"] = global_flags(r["text"], max_chars, item["turns"][: i + 1], single) if r["text"] else {}
        turns_out.append(r)

    last = turns_out[-1]
    item_flags: dict[str, list[str]] = {}
    text = last["text"]
    if any_ := item.get("must_any"):
        if not any(k in text for k in any_):
            item_flags["缺少關鍵內容"] = any_
    if hits := [k for k in item.get("must_not", []) if k in text]:
        item_flags["出現禁止內容"] = hits
    if hits := [p for p in item.get("must_not_regex", []) if re.search(p, text)]:
        item_flags["出現禁止內容"] = item_flags.get("出現禁止內容", []) + hits
    if "tool" in item:
        called = [tc["name"] for tc in last["tool_calls"]]
        want = item["tool"]
        if want == "none" and called:
            item_flags["工具"] = [f"不該呼叫，卻呼叫了 {called}"]
        elif want != "none" and want not in called:
            item_flags["工具"] = [f"應呼叫 {want}，實際 {called or '沒呼叫'}"]

    all_flags = {}
    for i, t in enumerate(turns_out):
        for k, v in t["flags"].items():
            all_flags[k if len(turns_out) == 1 else f"{k}(第{i+1}輪)"] = v
    all_flags.update(item_flags)
    return {
        "id": item["id"], "cat": item["cat"], "note": item.get("note", ""), "seed": seed,
        "turns": turns_out, "flags": all_flags, "passed": not all_flags,
    }


def write_report(results: list[dict], meta: dict, path: Path) -> None:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r)
    flag_count = defaultdict(int)
    for r in results:
        for k in r["flags"]:
            flag_count[re.sub(r"\(第\d+輪\)", "", k)] += 1

    lines = [
        f"# 大腦評估報告：{meta['tag']}",
        "",
        f"- 模型：`{meta['model']}`",
        f"- system prompt：`{meta['prompt']}`",
        f"- 時間：{meta['time']}，seed={meta['seed']}，每題跑 {meta['repeats']} 次，文字過濾：{'有' if meta['filter'] else '無'}",
        "",
        "## 摘要",
        "",
        f"**自動檢查通過：{meta['passed']}/{meta['total']}（{meta['pass_rate']:.0%}）**",
        f"｜ 回答字數中位數 {meta['chars_median']} ｜ TTFT 中位數 {meta['ttft_median']} ms ｜ 第一句中位數 {meta['first_sentence_median']} ms",
        "",
        f"風格：結尾反問比例 {meta['question_end_rate']:.0%} ｜ 每句「超」{meta['chao_per_reply']} 次 ｜ 提到雨的比例 {meta['rain_rate']:.0%}",
        "",
        "| 類別 | 通過 |",
        "|---|---|",
    ]
    for cat, lst in by_cat.items():
        lines.append(f"| {cat} | {sum(r['passed'] for r in lst)}/{len(lst)} |")
    lines += ["", "| 問題類型 | 題數 |", "|---|---|"]
    for k, v in sorted(flag_count.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## 逐題（人工評分：1 很差 / 3 普通 / 5 很像她）",
        "",
    ]
    for cat, lst in by_cat.items():
        lines.append(f"### {cat}")
        lines.append("")
        for r in lst:
            mark = "✅" if r["passed"] else "❌"
            rep = f" #{r['seed'] - meta['seed']}" if meta["repeats"] > 1 else ""
            lines.append(f"**{r['id']}{rep}** {mark}" + (f" — _{r['note']}_" if r["note"] else ""))
            lines.append("")
            for t in r["turns"]:
                lines.append(f"> 👤 {t['user']}  ")
                ans = t["text"].replace("\n", " ") or "（無文字）"
                if t["tool_calls"]:
                    ans += "  🔧 " + ", ".join(f"{c['name']}({c['arguments']})" for c in t["tool_calls"])
                lines.append(f"> 🤖 {ans}")
                lines.append(">")
            if lines[-1] == ">":
                lines.pop()
            if r["flags"]:
                lines.append("")
                lines.append("問題：" + "；".join(f"{k}：{', '.join(map(str, v))}" for k, v in r["flags"].items()))
            lines.append("")
            lines.append("人工評分：__/5")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--testset", default=str(ROOT / "eval/testset.yaml"))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--only", default="", help="只跑這些題號，逗號分隔")
    ap.add_argument("--filter", action="store_true", help="先經過 text_filter.for_speech 再檢查（模擬送進 TTS 的文字）")
    ap.add_argument("--repeats", type=int, default=1, help="每題用不同 seed 跑幾次（降低隨機波動）")
    args = ap.parse_args()

    system_prompt = Path(args.prompt).read_text().strip()
    items = load_testset(Path(args.testset))
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]

    props = requests.get(f"{args.url}/props", timeout=5).json()
    chat(args.url, [{"role": "user", "content": "你好"}], None, args.seed, 8)  # 暖機

    results = []
    for it in items:
        for rep in range(args.repeats):
            r = run_item(it, system_prompt, args, args.seed + rep)
            results.append(r)
            mark = "✅" if r["passed"] else "❌"
            ans = r["turns"][-1]["text"].replace("\n", " ")
            tc = r["turns"][-1]["tool_calls"]
            if tc:
                ans += " 🔧" + ",".join(c["name"] for c in tc)
            print(f"{mark} {r['id']}{'#' + str(rep) if args.repeats > 1 else ''} {ans}")
            for name, v in r["flags"].items():
                print(f"      ⚠ {name}: {v}")

    all_turns = [t for r in results for t in r["turns"] if t["text"]]
    passed = sum(r["passed"] for r in results)
    meta = {
        "tag": args.tag,
        "model": Path(props.get("model_path", "?")).name,
        "prompt": args.prompt,
        "seed": args.seed,
        "repeats": args.repeats,
        "filter": args.filter,
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / max(len(results), 1),
        "chars_median": statistics.median(count_chars(t["text"]) for t in all_turns) if all_turns else 0,
        "ttft_median": statistics.median(t["ttft_ms"] for t in all_turns) if all_turns else 0,
        "first_sentence_median": statistics.median(t["first_sentence_ms"] for t in all_turns) if all_turns else 0,
        # 風格統計：太常反問、太常用「超」、太常提到雨，聽起來會很機械
        "question_end_rate": round(sum(bool(QUESTION_END.search(t["text"])) for t in all_turns) / max(len(all_turns), 1), 2),
        "chao_per_reply": round(sum(t["text"].count("超") for t in all_turns) / max(len(all_turns), 1), 2),
        "rain_rate": round(sum("雨" in t["text"] for t in all_turns) / max(len(all_turns), 1), 2),
    }
    print(
        f"\n通過 {passed}/{len(results)}（{meta['pass_rate']:.0%}），字數中位數 {meta['chars_median']}，"
        f"TTFT 中位數 {meta['ttft_median']}ms，第一句中位數 {meta['first_sentence_median']}ms\n"
        f"結尾反問比例 {meta['question_end_rate']:.0%}，每句「超」{meta['chao_per_reply']} 次，提到雨的比例 {meta['rain_rate']:.0%}"
    )
    if not args.only:
        out_json = ROOT / f"logs/eval_{args.tag}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2))
        write_report(results, meta, ROOT / f"eval/reports/{args.tag}.md")
        print(f"報告：eval/reports/{args.tag}.md")


if __name__ == "__main__":
    main()
