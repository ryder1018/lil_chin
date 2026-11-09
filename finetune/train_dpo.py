"""Step 4（第二階段）：在 SFT 的 LoRA 上做 DPO。

資料：finetune/data/dpo.jsonl（build_dataset.py 產生）
  chosen   = 手寫的小魚回答
  rejected = 原模型（v4 prompt）對同一題的回答，而且有自動檢查問題（假記憶、簡體、硬扯看電影…）
reference model = SFT 模型本身（訓練前先把它對每組的 log prob 算好存起來），所以不用再載一份模型。

用法：python finetune/train_dpo.py --sft sft_r1 --tag dpo_r1 [--merge]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from unsloth import FastVisionModel  # noqa: I001

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models/hf/Qwen3-VL-4B-Instruct"
TEMPLATE = (BASE / "chat_template.jinja").read_text()


def encode(tok, prompt_msgs: list[dict], answer: str) -> tuple[torch.Tensor, int]:
    p = tok.apply_chat_template(prompt_msgs, chat_template=TEMPLATE, tokenize=False, add_generation_prompt=True)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return torch.tensor([p_ids + a_ids]), len(p_ids)


def seq_logp(model, ids: torch.Tensor, n_prompt: int) -> torch.Tensor:
    """只對回答那段算 log prob：整段 2000+ token × 15 萬詞表的 float32 logits 會塞爆 8GB VRAM。"""
    ids = ids.to(model.device)
    k = ids.shape[1] - n_prompt + 1  # 只算最後 k 個位置的 logits（第 n_prompt-1 個到最後）
    logits = model(input_ids=ids, logits_to_keep=k).logits[:, -k:-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 2, ids[:, n_prompt:, None]).squeeze(-1)
    return lp.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT 的 tag（讀 finetune/out/<sft>/lora）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rpo-alpha", type=float, default=0.2, help="加上 chosen 的 NLL，避免 DPO 把兩邊機率一起壓低")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    out = ROOT / "finetune/out" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    sft_cfg = json.load(open(ROOT / "finetune/out" / args.sft / "train_log.json"))["args"]

    model, processor = FastVisionModel.from_pretrained(str(BASE), load_in_4bit=True, use_gradient_checkpointing="unsloth")
    tok = processor.tokenizer
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=sft_cfg["rank"], lora_alpha=sft_cfg["rank"], lora_dropout=0, bias="none")
    sd = load_file(str(ROOT / "finetune/out" / args.sft / "lora/adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, sd)
    assert not res.unexpected_keys, res.unexpected_keys[:5]

    pairs = [json.loads(l) for l in open(ROOT / "finetune/data/dpo.jsonl")]
    data = []
    FastVisionModel.for_inference(model)
    with torch.no_grad():
        for p in pairs:
            c, nc = encode(tok, p["prompt"], p["chosen"])
            r, nr = encode(tok, p["prompt"], p["rejected"])
            data.append({"c": c, "nc": nc, "r": r, "nr": nr,
                         "ref_c": seq_logp(model, c, nc).item(), "ref_r": seq_logp(model, r, nr).item()})
    print(f"DPO pairs {len(data)}，reference（SFT）偏好 chosen 的比例 {sum(d['ref_c'] > d['ref_r'] for d in data) / len(data):.0%}")

    FastVisionModel.for_training(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(data) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * max(0.0, 1 - s / max(total, 1)))
    log, step = [], 0
    rng = random.Random(0)
    for ep in range(args.epochs):
        rng.shuffle(data)
        acc_loss = acc_win = 0.0
        for i, d in enumerate(data):
            lc = seq_logp(model, d["c"], d["nc"])
            lr_ = seq_logp(model, d["r"], d["nr"])
            margin = args.beta * ((lc - d["ref_c"]) - (lr_ - d["ref_r"]))
            loss = -F.logsigmoid(margin)
            if args.rpo_alpha:
                loss = loss + args.rpo_alpha * (-lc / (d["c"].shape[1] - d["nc"]))
            (loss / args.accum).backward()
            acc_loss += loss.item()
            acc_win += float(margin.item() > 0)
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    rec = {"step": step, "epoch": ep, "loss": acc_loss / (5 * args.accum), "win": acc_win / (5 * args.accum)}
                    log.append(rec); print(rec)
                    acc_loss = acc_win = 0.0
    print(f"VRAM 峰值 {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
    model.save_pretrained(str(out / "lora"))
    processor.save_pretrained(str(out / "lora"))
    json.dump({"args": vars(args), "log": log, "n": len(data)}, open(out / "train_log.json", "w"), indent=1)
    if args.merge:
        model.save_pretrained_merged(str(out / "merged"), processor, save_method="merged_16bit")


if __name__ == "__main__":
    main()
