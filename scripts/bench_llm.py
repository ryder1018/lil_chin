"""Step 1 基準測試：量測大腦（llama-server）的延遲與速度。

量測項目（全部用串流，模擬之後接 TTS 的情境）：
  - TTFT：送出到第一個 token 出現
  - 第一句時間：送出到第一個完整句子（遇到 。！？!?～ 或換行）——TTS 拿到這句就能開口
  - 總時間、輸出 token 數、tok/s（以 server 回報的 timings 為準）
  - 多輪對話：第 2 輪以後應該命中 prefix cache，TTFT 會下降

用法：
  python scripts/bench_llm.py                 # 預設 http://127.0.0.1:8080
  python scripts/bench_llm.py --runs 3 --out logs/bench_step1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests

SENTENCE_END = set("。！？!?～\n")

# 暫時的 system prompt，Step 2 會正式設計
SYSTEM_PROMPT = (
    "你是一台放在桌上的聊天機器人，使用台灣繁體中文和台灣用語。"
    "你的回答會被轉成語音唸出來，所以只用口語短句，一次回答一到三句話，"
    "不要用條列、markdown、表情符號。"
)

SINGLE_TURN = [
    "早安！",
    "今天天氣有點冷，晚餐吃什麼好？",
    "你可以幫我解釋一下什麼是黑洞嗎？",
    "我今天上班被老闆罵了，好煩。",
    "台北到高雄搭高鐵大概要多久？",
    "幫我想一個週末可以做的活動。",
]

MULTI_TURN = [
    "我叫小明，我最近在學吉他。",
    "我應該先練什麼？",
    "每天要練多久比較好？",
    "你還記得我叫什麼名字嗎？",
]


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def stream_chat(base_url: str, messages: list[dict], max_tokens: int) -> dict:
    t0 = time.perf_counter()
    ttft = first_sentence = None
    text = ""
    timings = None
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "timings_per_token": False,
        },
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        # 用 bytes 切行再解碼：decode_unicode=True 會被 \x85、  等字元切斷 JSON
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("timings"):
                timings = chunk["timings"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content") or ""
            if not delta:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            text += delta
            if first_sentence is None and any(c in SENTENCE_END for c in delta):
                first_sentence = now - t0
    total = time.perf_counter() - t0
    return {
        "text": text,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "first_sentence_ms": round((first_sentence or total) * 1000, 1),
        "total_ms": round(total * 1000, 1),
        "prompt_tokens": timings.get("prompt_n") if timings else None,
        "cached_tokens": timings.get("cache_n") if timings else None,
        "output_tokens": timings.get("predicted_n") if timings else None,
        "tok_per_s": round(timings["predicted_per_second"], 1) if timings else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=2, help="單輪題目各跑幾次")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="logs/bench_step1.json")
    args = ap.parse_args()

    requests.get(f"{args.url}/health", timeout=5).raise_for_status()
    props = requests.get(f"{args.url}/props", timeout=5).json()
    model_path = props.get("model_path", "?")

    # 暖機（第一次推論會比較慢，不計入）
    stream_chat(args.url, [{"role": "user", "content": "你好"}], 16)

    results = {"model": model_path, "single_turn": [], "multi_turn": []}

    print(f"模型：{Path(model_path).name}\n")
    print("=== 單輪 ===")
    for q in SINGLE_TURN:
        for i in range(args.runs):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            r = stream_chat(args.url, msgs, args.max_tokens)
            r["question"] = q
            r["run"] = i
            results["single_turn"].append(r)
            print(
                f"TTFT {r['ttft_ms']:>6}ms | 第一句 {r['first_sentence_ms']:>6}ms | "
                f"總 {r['total_ms']:>6}ms | {r['output_tokens']:>3} tok @ {r['tok_per_s']} tok/s | "
                f"cache {r['cached_tokens']}/{r['prompt_tokens']}"
            )
            if i == 0:
                print(f"   Q: {q}\n   A: {r['text'].strip()}")

    print("\n=== 多輪（測 prefix cache 與記憶） ===")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q in MULTI_TURN:
        msgs.append({"role": "user", "content": q})
        r = stream_chat(args.url, msgs, args.max_tokens)
        msgs.append({"role": "assistant", "content": r["text"]})
        r["question"] = q
        results["multi_turn"].append(r)
        print(
            f"TTFT {r['ttft_ms']:>6}ms | 第一句 {r['first_sentence_ms']:>6}ms | "
            f"cache {r['cached_tokens']}/{r['prompt_tokens']}\n   Q: {q}\n   A: {r['text'].strip()}"
        )

    st = results["single_turn"]
    summary = {
        "ttft_ms_median": statistics.median(x["ttft_ms"] for x in st),
        "first_sentence_ms_median": statistics.median(x["first_sentence_ms"] for x in st),
        "first_sentence_ms_max": max(x["first_sentence_ms"] for x in st),
        "tok_per_s_median": statistics.median(x["tok_per_s"] for x in st),
        "output_tokens_median": statistics.median(x["output_tokens"] for x in st),
        "gpu_mem_used_mib_total": gpu_mem_used_mib(),
    }
    results["summary"] = summary
    print("\n=== 摘要 ===")
    for k, v in summary.items():
        print(f"{k:28s} {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
