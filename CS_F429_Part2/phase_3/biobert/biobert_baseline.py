"""BioBERT baseline for BC5CDR chemical/disease NER.

The modern half of the classical-vs-modern comparison. Trains on exactly the
same data/{train,dev,test}.json as crf_baseline.py and scores with the same
ner_metrics.py, so the two numbers are directly comparable.

  python biobert_baseline.py                       # train, select on dev, evaluate test once
  python biobert_baseline.py --epochs 5 --lr 3e-5
  python biobert_baseline.py --eval-only           # load saved checkpoint, evaluate
  python biobert_baseline.py --smoke               # 200 sentences, 1 epoch (environment check)

Writes:
  models/biobert/                     best checkpoint (by dev F1)
  results/biobert_metrics.json        dev + test scores, matches crf_metrics.json layout
  biobert_test_predictions.json       same shape as crf_test_predictions.json
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ner_metrics.py lives at the Part 2 root, two levels up from phase_3/biobert/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import ner_metrics  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models", "biobert")
RESULTS_PATH = os.path.join(ROOT, "results", "biobert_metrics.json")
PRED_PATH = os.path.join(ROOT, "biobert_test_predictions.json")

CHECKPOINT = "dmis-lab/biobert-base-cased-v1.1"
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

IGNORE = -100  # loss mask for subword continuations and special tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ data

class NERDataset(Dataset):
    """Pre-tokenized sentences -> subword ids with first-subword labelling.

    BERT splits words into subwords, but the tags are per word. We label the
    first subword of each word and mask the rest with -100, then read the
    prediction for a word off its first subword. Words pushed past max_len by
    truncation keep no subword at all; evaluate() back-fills those with "O",
    which costs recall rather than silently shortening the sequence.
    """

    def __init__(self, sentences, tokenizer, max_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        s = self.sentences[i]
        enc = self.tokenizer(
            s["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
        )
        word_ids = enc.word_ids()
        labels, prev = [], None
        for w in word_ids:
            if w is None or w == prev:
                labels.append(IGNORE)
            else:
                labels.append(LABEL2ID[s["tags"][w]])
            prev = w
        enc = dict(enc)
        enc["labels"] = labels
        # word_ids are needed at eval time to map subwords back to words
        enc["word_ids"] = [-1 if w is None else w for w in word_ids]
        enc["index"] = i
        return enc


def make_collate(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {k: [] for k in ("input_ids", "attention_mask", "labels", "word_ids")}
        for b in batch:
            pad = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [IGNORE] * pad)
            out["word_ids"].append(b["word_ids"] + [-1] * pad)
        t = {k: torch.tensor(v) for k, v in out.items()}
        t["index"] = torch.tensor([b["index"] for b in batch])
        return t

    return collate


# ------------------------------------------------------------------ eval

@torch.no_grad()
def predict(model, loader, sentences, device, amp_dtype):
    """Returns a list of word-level tag sequences, one per sentence, aligned
    with `sentences` and always the same length as that sentence's tokens."""
    model.eval()
    preds = [None] * len(sentences)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids=ids, attention_mask=mask).logits
        best = logits.argmax(-1).cpu()
        word_ids = batch["word_ids"]
        for row, sent_i in enumerate(batch["index"].tolist()):
            n_words = len(sentences[sent_i]["tokens"])
            tags = ["O"] * n_words  # truncated tail stays O
            prev = -1
            for pos, w in enumerate(word_ids[row].tolist()):
                if w >= 0 and w != prev:
                    tags[w] = LABELS[best[row, pos].item()]
                prev = w
            preds[sent_i] = tags
    return preds


def evaluate(model, sentences, tokenizer, args, device, amp_dtype, name):
    ds = NERDataset(sentences, tokenizer, args.max_len)
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )
    pred = predict(model, loader, sentences, device, amp_dtype)
    gold = [s["tags"] for s in sentences]
    res = ner_metrics.evaluate(gold, pred)
    print(f"\n{name} results (entity level, exact span + type match)")
    print(ner_metrics.report(res))
    o = res["overall"]
    print(f"Overall P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    return pred, res


# ------------------------------------------------------------------ train

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB visible)")
    else:
        print("WARNING: no CUDA device, training on CPU will be slow.")

    # bf16 on Ampere and newer avoids the loss-scaling fiddle fp16 needs
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    train_data, dev_data, test_data = (load_split(s) for s in ("train", "dev", "test"))
    if args.smoke:
        train_data, dev_data, test_data = train_data[:200], dev_data[:200], test_data[:200]
        args.epochs = 1
    print(f"train {len(train_data)}  dev {len(dev_data)}  test {len(test_data)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = AutoConfig.from_pretrained(
        args.checkpoint,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, config=config)
    model.to(device)

    loader = DataLoader(
        NERDataset(train_data, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )

    # no weight decay on biases and LayerNorm, the standard BERT fine-tuning recipe
    decay = [p for n, p in model.named_parameters()
             if not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters()
                if any(k in n for k in ("bias", "LayerNorm.weight"))]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total * args.warmup_ratio), total)

    best_f1, best_epoch = -1.0, -1
    os.makedirs(MODEL_DIR, exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "labels")}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            seen += 1
            if step % args.log_every == 0 or step == len(loader):
                print(f"  epoch {epoch} step {step}/{len(loader)}  "
                      f"loss {running / seen:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{time.time() - start:.0f}s", flush=True)
                running, seen = 0.0, 0

        _, dev_res = evaluate(model, dev_data, tokenizer, args, device, amp_dtype,
                              f"Dev (epoch {epoch})")
        f1 = dev_res["overall"]["f1"]
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  new best dev F1 {f1:.4f} -> saved to {MODEL_DIR}")

    print(f"\nBest dev F1 {best_f1:.4f} at epoch {best_epoch}; "
          f"reloading that checkpoint for the single test evaluation.")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)

    _, dev_res = evaluate(model, dev_data, tokenizer, args, device, amp_dtype, "Dev (best)")
    pred, test_res = evaluate(model, test_data, tokenizer, args, device, amp_dtype, "Test")
    save_outputs(args, test_data, pred, dev_res, test_res, best_epoch, time.time() - start)


def eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    dev_data, test_data = load_split("dev"), load_split("test")
    _, dev_res = evaluate(model, dev_data, tokenizer, args, device, amp_dtype, "Dev")
    pred, test_res = evaluate(model, test_data, tokenizer, args, device, amp_dtype, "Test")
    save_outputs(args, test_data, pred, dev_res, test_res, None, None)


def save_outputs(args, test_data, pred, dev_res, test_res, best_epoch, elapsed):
    if args.smoke:
        print("\n(smoke run: not writing results or predictions)")
        return
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "model": args.checkpoint,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "train_seconds": round(elapsed, 1) if elapsed else None,
            "dev": dev_res,
            "test": test_res,
        }, f, indent=2)
    with open(PRED_PATH, "w") as f:
        json.dump([{"tokens": s["tokens"], "gold": s["tags"], "pred": p}
                   for s, p in zip(test_data, pred)], f)
    print(f"\nWrote {RESULTS_PATH}\nWrote {PRED_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the environment")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    eval_only(args) if args.eval_only else train(args)


if __name__ == "__main__":
    main()
