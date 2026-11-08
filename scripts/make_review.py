"""把 eval_brain.py 的結果 json 轉成人工評分表（eval/human/<tag>_review.md）。

每題只取第一次的回答（同一題跑 3 次的話，看第一次就好）。評分表不會被評估腳本覆蓋，
已經存在的話不會覆寫，除非加 --force。

用法：
  python scripts/make_review.py logs/eval_v4_r6.json --desc "傲嬌、偶爾撒嬌，介於朋友和女友之間"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUBRIC = """| 分數 | 意思 |
|---|---|
| 5 | 就是我想要的{name}，可以直接當訓練範例 |
| 4 | 不錯，小地方可以更好 |
| 3 | 普通，沒錯但沒有味道（像客服、像罐頭回答） |
| 2 | 有明顯問題（語氣不對、答非所問、有點怪） |
| 1 | 很糟（亂編、傷人、出戲、違反設定） |"""


def turn_lines(t: dict) -> list[str]:
    bot = t["text"] or "（無文字）"
    for c in t.get("tool_calls") or []:
        bot += f"  🔧 {c['name']}({c.get('arguments', '')})"
    return [f"> 👤 {t['user']}  ", f"> 🤖 {bot}  ", ">"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="logs/eval_<tag>.json")
    ap.add_argument("--desc", default="", help="評分時要對照的個性描述")
    ap.add_argument("--force", action="store_true", help="覆寫已經存在的評分表")
    args = ap.parse_args()

    data = json.loads(Path(args.log).read_text(encoding="utf-8"))
    meta, results = data["meta"], data["results"]
    tag, name = meta["tag"], meta.get("name", "她")
    out = ROOT / "eval" / "human" / f"{tag}_review.md"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} 已經存在（裡面可能有你填的分數），要覆寫請加 --force")

    first: dict[str, dict] = {}
    for r in results:
        first.setdefault(r["id"], r)

    desc = f"（{args.desc}）" if args.desc else ""
    lines = [
        f"# 人工評分：{tag}（每題第一個回答）",
        "",
        f"prompt：`{meta['prompt']}`，原始報告：[../reports/{tag}.md](../reports/{tag}.md)。"
        "這個檔案不會被評估腳本覆蓋。",
        "",
        "## 怎麼填",
        "",
        "把每題的 `__` 改成 1–5 的數字；有想法就寫在「意見：」後面（例如「太像客服」「這句很可愛」「她不會這樣講」）。",
        "**意見比分數更有用**——Step 4 會把你的意見變成訓練資料。",
        "",
        RUBRIC.format(name=name),
        "",
        f"評分時看三件事：**像不像她**{desc}、**有沒有回應到你說的話**、**聽起來自不自然**（想像它被唸出來）。",
        "不用全部評，先評你有感覺的十幾題也可以。",
    ]
    cat = None
    for r in first.values():
        if r["cat"] != cat:
            cat = r["cat"]
            lines += ["", f"## {cat}"]
        note = f"  — _{r['note']}_" if r.get("note") else ""
        lines += ["", f"### {r['id']} {'✅' if r['passed'] else '❌'}{note}", ""]
        for t in r["turns"]:
            lines += turn_lines(t)
        lines.pop()  # 最後一個空的 ">"
        if r.get("flags"):
            flags = "；".join(f"{k}：{'、'.join(v)}" if v else k for k, v in r["flags"].items())
            lines += ["", f"自動檢查：{flags}"]
        lines += ["", "分數：__/5", "", "意見："]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"寫好了：{out}（{len(first)} 題）")


if __name__ == "__main__":
    main()
