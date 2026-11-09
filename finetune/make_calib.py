"""imatrix 校正文字：把 sft.jsonl 的對話用 chat template 展開成純文字。
人設 system prompt 很長、每筆都一樣，只保留前幾筆，避免校正資料被它佔滿。"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models/hf/Qwen3-VL-4B-Instruct"
tok = AutoTokenizer.from_pretrained(str(BASE))
tpl = (BASE / "chat_template.jinja").read_text()
rows = [json.loads(l) for l in open(ROOT / "finetune/data/sft.jsonl")]
parts = []
for i, r in enumerate(rows):
    msgs = r["messages"] if i < 10 else [m for m in r["messages"] if m["role"] != "system"]
    parts.append(tok.apply_chat_template(msgs, tools=r.get("tools") if i < 10 else None,
                                         chat_template=tpl, tokenize=False))
Path(sys.argv[1]).write_text("\n".join(parts))
print(f"校正文字 {sum(map(len, parts))} 字 → {sys.argv[1]}")
