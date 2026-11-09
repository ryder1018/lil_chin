"""Step 4：Unsloth QLoRA 微調 Qwen3-VL-4B（只練語言部分，vision 凍結）。

用法（ai_robot_train 環境；先停掉 llama-server 讓出 VRAM）：
  python finetune/train_sft.py --tag sft_r1
  python finetune/train_sft.py --tag sft_r1 --merge     # 訓練完順便合併成 16-bit（轉 GGUF 用）

輸出：
  finetune/out/<tag>/lora      LoRA adapter
  finetune/out/<tag>/merged    合併後的 16-bit 模型（--merge）
  finetune/out/<tag>/train_log.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from unsloth import FastVisionModel  # noqa: I001  要在 transformers 之前 import

import torch
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models/hf/Qwen3-VL-4B-Instruct"
TEMPLATE = (BASE / "chat_template.jinja").read_text()


def tokenize(row: dict, tok, max_len: int) -> dict | None:
    """整段對話轉 token；只有 row["train"] 指定的 assistant 訊息（內容 + <|im_end|>）算 loss。"""
    msgs, tools = row["messages"], row.get("tools")
    render = lambda m, gen=False: tok.apply_chat_template(  # noqa: E731
        m, tools=tools, chat_template=TEMPLATE, tokenize=False, add_generation_prompt=gen)
    full = render(msgs)
    spans = []
    for i in row["train"]:
        start = render(msgs[:i], gen=True)
        end = render(msgs[: i + 1])
        assert full.startswith(end) and end.startswith(start), "chat template 不是前綴穩定的"
        spans.append((len(start), len(end) - 1))  # 不含最後的 \n
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(enc["offset_mapping"]):
        if any(s <= a and b <= e for s, e in spans):
            labels[k] = ids[k]
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(ROOT / "finetune/data/sft.jsonl"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2560)  # v4 人設 prompt 就快 1900 token
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    out = ROOT / "finetune/out" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    model, processor = FastVisionModel.from_pretrained(str(BASE), load_in_4bit=True, use_gradient_checkpointing="unsloth")
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=args.seed,
    )

    rows = [json.loads(l) for l in open(args.data)]
    data = [t for r in rows if (t := tokenize(r, tok, args.max_len))]
    print(f"訓練資料 {len(data)}/{len(rows)} 筆，算 loss 的 token {sum(sum(x != -100 for x in d['labels']) for d in data)}")
    # 檢查一筆：印出算 loss 的部分
    d = random.Random(0).choice(data)
    print("範例 loss 部分：", tok.decode([t for t, l in zip(d["input_ids"], d["labels"]) if l != -100]))

    FastVisionModel.for_training(model)
    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,  # transformers 5：小於 1 = 比例
        logging_steps=5, save_strategy="no", optim="adamw_8bit", weight_decay=0.01, bf16=True,
        seed=args.seed, report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    res = trainer.train()
    print(res)
    print(f"VRAM 峰值 {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
    model.save_pretrained(str(out / "lora"))
    processor.save_pretrained(str(out / "lora"))
    json.dump({"args": vars(args), "log": trainer.state.log_history, "n": len(data),
               "peak_vram_gb": torch.cuda.max_memory_reserved() / 2**30},
              open(out / "train_log.json", "w"), ensure_ascii=False, indent=1)
    if args.merge:
        model.save_pretrained_merged(str(out / "merged"), processor, save_method="merged_16bit")
        print("合併完成：", out / "merged")


if __name__ == "__main__":
    main()
