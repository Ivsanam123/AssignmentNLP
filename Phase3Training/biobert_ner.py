from google.colab import drive
drive.mount('/content/drive')
import os
assert os.path.isdir('/content/drive/MyDrive'), "Drive not mounted, stop and fix this first"

import sys
sys.argv = ["biobert_ner.py", "--out", "/content/drive/MyDrive/biobert_run"]

"""BioBERT fine-tuning for BC5CDR (Chemical / Disease NER). Colab-ready, SINGLE FILE.

(ner_metrics.py is now built in, so nothing else needs to be uploaded.)

Full run   : python biobert_ner.py
Live demo  : python biobert_ner.py --epochs 1 --max_train 800 --out demo_run

In Colab/Jupyter, run it as a shell command:  !python biobert_ner.py --epochs 1 --max_train 800 --out demo_run

Outputs (under --out):
  model/                       best model by dev F1 (reload with BertForTokenClassification)
  training_log.json            per-epoch train loss + dev P/R/F1
  biobert_metrics.json         dev + test metrics, same layout as crf_metrics.json
  biobert_test_predictions.json  [{tokens, gold, pred}] same layout as crf_test_predictions.json
"""
import argparse
import json
import os
import random
import time
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, get_linear_schedule_with_warmup

BASE = "https://huggingface.co/datasets/tner/bc5cdr/resolve/main/dataset/"
FILES = {"train": "train.json", "dev": "valid.json", "test": "test.json"}


# ============================================================== NER metrics (formerly ner_metrics.py)
# Entity-level metrics, seqeval/conlleval style: an entity is correct only if span AND type match.
def _split(tag):
    if tag == "O" or tag == "":
        return "O", ""
    if "-" in tag:
        p, t = tag.split("-", 1)
        return p, t
    return tag, ""


def get_entities(tags):
    """Return list of (type, start, end_exclusive) for one tag sequence."""
    ents = []
    cur_type, start = None, None
    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel closes the last entity
        prefix, typ = _split(tag)
        if cur_type is not None:
            if prefix in ("B", "O") or (prefix == "I" and typ != cur_type):
                ents.append((cur_type, start, i))
                cur_type, start = None, None
        if prefix == "B" or (prefix == "I" and cur_type is None and typ):
            cur_type, start = typ, i
    return ents


def entity_sets(gold_seqs, pred_seqs):
    gold, pred = set(), set()
    for k, (g, p) in enumerate(zip(gold_seqs, pred_seqs)):
        gold.update((k,) + e for e in get_entities(g))
        pred.update((k,) + e for e in get_entities(p))
    return gold, pred


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def ner_evaluate(gold_seqs, pred_seqs):
    """Return {'overall': {...}, 'per_type': {type: {...}}, 'macro': {...}}."""
    gold, pred = entity_sets(gold_seqs, pred_seqs)
    types = sorted({e[1] for e in gold} | {e[1] for e in pred})
    out = {"per_type": {}}
    for t in types:
        g = {e for e in gold if e[1] == t}
        p = {e for e in pred if e[1] == t}
        tp = len(g & p)
        pr, rc, f1 = prf(tp, len(p), len(g))
        out["per_type"][t] = dict(precision=pr, recall=rc, f1=f1, support=len(g),
                                  predicted=len(p), tp=tp)
    tp = len(gold & pred)
    pr, rc, f1 = prf(tp, len(pred), len(gold))
    out["overall"] = dict(precision=pr, recall=rc, f1=f1, support=len(gold),
                          predicted=len(pred), tp=tp)
    ps = [v["precision"] for v in out["per_type"].values()]
    rs = [v["recall"] for v in out["per_type"].values()]
    fs = [v["f1"] for v in out["per_type"].values()]
    n = max(len(fs), 1)
    out["macro"] = dict(precision=sum(ps) / n, recall=sum(rs) / n, f1=sum(fs) / n)
    return out


def ner_report(res, digits=4):
    """Pretty text table similar to seqeval.classification_report."""
    w = max([len(t) for t in res["per_type"]] + [9])
    lines = [f"{'':>{w}}  precision    recall  f1-score   support", ""]
    for t, v in res["per_type"].items():
        lines.append(f"{t:>{w}}  {v['precision']:9.{digits}f} {v['recall']:9.{digits}f} "
                     f"{v['f1']:9.{digits}f} {v['support']:9d}")
    lines.append("")
    o, m = res["overall"], res["macro"]
    lines.append(f"{'micro avg':>{w}}  {o['precision']:9.{digits}f} {o['recall']:9.{digits}f} "
                 f"{o['f1']:9.{digits}f} {o['support']:9d}")
    lines.append(f"{'macro avg':>{w}}  {m['precision']:9.{digits}f} {m['recall']:9.{digits}f} "
                 f"{m['f1']:9.{digits}f} {o['support']:9d}")
    return "\n".join(lines)


# ============================================================== data (same source as fetch_data.py)
def fetch(raw_dir="data_raw"):
    os.makedirs(raw_dir, exist_ok=True)
    for name in list(FILES.values()) + ["label.json"]:
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                open(path, "wb").write(r.read())
            print("downloaded", path)
    label2id = json.load(open(os.path.join(raw_dir, "label.json"), encoding="utf-8"))
    id2label = {v: k for k, v in label2id.items()}
    splits = {}
    for split, fname in FILES.items():
        sents = []
        for line in open(os.path.join(raw_dir, fname), encoding="utf-8"):
            if line.strip():
                ex = json.loads(line)
                sents.append({"tokens": ex["tokens"], "tags": [id2label[t] for t in ex["tags"]]})
        splits[split] = sents
    return splits, label2id, id2label


def encode(sents, tok, label2id, max_len):
    """Word-level tags -> subword inputs. Label goes on the FIRST subword of each word, -100 elsewhere."""
    out = []
    for s in sents:
        enc = tok(s["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        wids, labels, first = enc.word_ids(), [], []
        prev = None
        for pos, w in enumerate(wids):
            if w is None or w == prev:
                labels.append(-100)
            else:
                labels.append(label2id[s["tags"][w]])
                first.append((w, pos))
            prev = w
        out.append({"input_ids": enc["input_ids"], "labels": labels, "first": first, "n_words": len(s["tokens"])})
    return out


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return ids, att, lab


@torch.no_grad()
def predict(model, enc, id2label, pad_id, device, bs=64):
    """Word-level predicted tag strings for every sentence (words cut by truncation stay 'O')."""
    model.eval()
    preds = []
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]
        ids, att, _ = collate(chunk, pad_id)
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.argmax(-1).cpu()
        for b, row in zip(chunk, logits):
            tags = ["O"] * b["n_words"]
            for w, pos in b["first"]:
                tags[w] = id2label[int(row[pos])]
            preds.append(tags)
    return preds


def evaluate(model, enc, sents, id2label, pad_id, device):
    pred = predict(model, enc, id2label, pad_id, device)
    gold = [s["tags"] for s in sents]
    return pred, ner_evaluate(gold, pred)


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.1")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=0, help="use only the first N train sentences (live demo)")
    ap.add_argument("--out", default="biobert_run")
    a, _ = ap.parse_known_args()  # tolerant of extra args injected by notebooks

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| args:", vars(a))
    os.makedirs(a.out, exist_ok=True)

    splits, label2id, id2label = fetch()
    if a.max_train:
        splits["train"] = splits["train"][:a.max_train]
    print({k: len(v) for k, v in splits.items()})

    tok = BertTokenizerFast.from_pretrained(a.model, do_lower_case=False)
    enc = {k: encode(v, tok, label2id, a.max_len) for k, v in splits.items()}
    model = BertForTokenClassification.from_pretrained(
        a.model, num_labels=len(label2id), id2label=id2label, label2id=label2id).to(device)

    loader = DataLoader(enc["train"], batch_size=a.batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": a.wd},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    total = len(loader) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log, best_f1 = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, (ids, att, lab) in enumerate(loader, 1):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(f"epoch {ep}/{a.epochs} step {step}/{len(loader)} loss {run_loss / step:.4f}", flush=True)
        _, dev_res = evaluate(model, enc["dev"], splits["dev"], id2label, tok.pad_token_id, device)
        o = dev_res["overall"]
        print(f"== epoch {ep}: train loss {run_loss / len(loader):.4f} | dev P {o['precision']:.4f} "
              f"R {o['recall']:.4f} F1 {o['f1']:.4f} | {time.time() - t0:.0f}s", flush=True)
        log.append({"epoch": ep, "train_loss": run_loss / len(loader), "dev_precision": o["precision"],
                    "dev_recall": o["recall"], "dev_f1": o["f1"], "seconds": time.time() - t0})
        if o["f1"] > best_f1:
            best_f1 = o["f1"]
            model.save_pretrained(os.path.join(a.out, "model"))
            tok.save_pretrained(os.path.join(a.out, "model"))
            print("   saved best model", flush=True)

    json.dump({"args": vars(a), "log": log}, open(os.path.join(a.out, "training_log.json"), "w"), indent=2)

    # final: reload best-on-dev model, score dev and test once
    model = BertForTokenClassification.from_pretrained(os.path.join(a.out, "model")).to(device)
    _, dev_res = evaluate(model, enc["dev"], splits["dev"], id2label, tok.pad_token_id, device)
    test_pred, test_res = evaluate(model, enc["test"], splits["test"], id2label, tok.pad_token_id, device)
    print("\nTest results (entity level, exact span + type match)")
    print(ner_report(test_res))
    o = test_res["overall"]
    print(f"Overall P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

    json.dump({"model": a.model, "lr": a.lr, "batch": a.batch, "epochs": a.epochs, "max_len": a.max_len,
               "dev": dev_res, "test": test_res}, open(os.path.join(a.out, "biobert_metrics.json"), "w"), indent=2)
    json.dump([{"tokens": s["tokens"], "gold": s["tags"], "pred": p} for s, p in zip(splits["test"], test_pred)],
              open(os.path.join(a.out, "biobert_test_predictions.json"), "w"))
    print("saved to", a.out)


if __name__ == "__main__":
    main()
    drive.flush_and_unmount()   # forces all pending writes to reach Drive
    print("Drive flushed")