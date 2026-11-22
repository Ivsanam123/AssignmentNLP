import sys
# ---- CHOOSE THE RUN HERE (change only this word, then run the cell) -------------------------------
RUN = "replace_only"      # "replace_only" | "mask_only" | "rare" (replace + mask) | "baseline"
USE_DRIVE = False         # False = save under /content and auto-download the 3 JSONs (no Drive needed)
# ---------------------------------------------------------------------------------------------------
D = "/content/drive/MyDrive/" if USE_DRIVE else "/content/"
PRESETS = {
    "baseline":     ["--skip_test", "--out", D + "biobert_dev_baseline"],
    "replace_only": ["--replace", "0.7", "--skip_test", "--out", D + "biobert_replace_only"],
    "mask_only":    ["--mask", "0.3", "--skip_test", "--out", D + "biobert_mask_only"],
    "rare":         ["--replace", "0.7", "--mask", "0.3", "--skip_test", "--out", D + "biobert_rare"],
}
sys.argv = ["biobert_ner.py"] + PRESETS[RUN]

import argparse
import gc
import json
import os
import random
import time
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, get_linear_schedule_with_warmup

try:  # only on Google Colab: persist results on Drive
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

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


def entity_inventory(train):
    """type -> list of DISTINCT training entity strings (token lists): uniform sampling over strings."""
    inv = {}
    for s in train:
        for t, a, b in get_entities(s["tags"]):
            inv.setdefault(t, {}).setdefault(" ".join(s["tokens"][a:b]).lower(), s["tokens"][a:b])
    return {t: list(v.values()) for t, v in inv.items()}


def augment(sent, rng, inv, p_rep, p_mask):
    """Training-time rare-entity intervention. Returns (sentence, set of word indices to [MASK])."""
    toks, tags = sent["tokens"], sent["tags"]
    new_t, new_g, pos, masked = [], [], 0, set()
    for t, a, b in get_entities(tags):
        new_t += toks[pos:a]; new_g += tags[pos:a]
        rep = rng.choice(inv[t]) if rng.random() < p_rep else toks[a:b]
        start = len(new_t)
        new_t += rep; new_g += [f"B-{t}"] + [f"I-{t}"] * (len(rep) - 1); pos = b
        if rng.random() < p_mask:
            masked.update(range(start, start + len(rep)))
    return {"tokens": new_t + toks[pos:], "tags": new_g + tags[pos:]}, masked


def encode(sents, tok, label2id, max_len, masks=None):
    """Word-level tags -> subword inputs. Label goes on the FIRST subword of each word, -100 elsewhere.
    masks (optional): per-sentence set of word indices whose sub-words are replaced by [MASK]."""
    out = []
    for si, s in enumerate(sents):
        enc = tok(s["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        wids, labels, first = enc.word_ids(), [], []
        if masks is not None and masks[si]:
            enc["input_ids"] = [tok.mask_token_id if (w is not None and w in masks[si]) else t
                                for t, w in zip(enc["input_ids"], wids)]
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
    ap.add_argument("--out", default="/content/drive/MyDrive/biobert_run" if IN_COLAB else "biobert_run")
    ap.add_argument("--replace", type=float, default=0.0, help="intervention: mention replacement prob")
    ap.add_argument("--mask", type=float, default=0.0, help="intervention: mention [MASK] prob")
    ap.add_argument("--skip_test", action="store_true", help="do not score the test set")
    ap.add_argument("--eval_only", action="store_true", help="score the saved model in --out/model, no training")
    a, _ = ap.parse_known_args()  # tolerant of extra args injected by notebooks
    if IN_COLAB and USE_DRIVE:
        drive.mount("/content/drive")
        assert os.path.isdir("/content/drive/MyDrive"), "Drive not mounted, stop and fix this first"

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

    inv, rng = entity_inventory(splits["train"]), random.Random(a.seed)
    intervene = a.replace > 0 or a.mask > 0
    if intervene:
        print(f"intervention ON: replace={a.replace} mask={a.mask}")

    def make_loader():  # with the intervention every epoch sees a freshly augmented copy of the training set
        data = enc["train"]
        if intervene:
            aug = [augment(s, rng, inv, a.replace, a.mask) for s in splits["train"]]
            data = encode([x[0] for x in aug], tok, label2id, a.max_len, [x[1] for x in aug])
        return DataLoader(data, batch_size=a.batch, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))

    n_epochs = 0 if a.eval_only else a.epochs   # a.epochs stays untouched so the metrics file records the real value
    loader = make_loader()
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": a.wd},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    total = len(loader) * n_epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log, best_f1 = [], -1.0
    for ep in range(1, n_epochs + 1):
        if intervene:
            loader = make_loader()
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
                print(f"epoch {ep}/{n_epochs} step {step}/{len(loader)} loss {run_loss / step:.4f}", flush=True)
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

    if not a.eval_only:
        json.dump({"args": vars(a), "log": log}, open(os.path.join(a.out, "training_log.json"), "w"), indent=2)

    # final: reload best-on-dev model, score dev (and test unless --skip_test)
    metrics_path = os.path.join(a.out, "biobert_metrics.json")
    old_test = None
    if a.eval_only and a.skip_test and os.path.exists(metrics_path):   # keep an existing test result for this same model
        old_test = json.load(open(metrics_path)).get("test")
    model = BertForTokenClassification.from_pretrained(os.path.join(a.out, "model")).to(device)
    dev_pred, dev_res = evaluate(model, enc["dev"], splits["dev"], id2label, tok.pad_token_id, device)
    o = dev_res["overall"]
    print(f"\nBest model on DEV: P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    json.dump(dev_pred, open(os.path.join(a.out, "biobert_dev_pred.json"), "w"))   # one tag list per dev sentence
    test_res = old_test
    if not a.skip_test:   # the single, final test evaluation
        test_pred, test_res = evaluate(model, enc["test"], splits["test"], id2label, tok.pad_token_id, device)
        print("\nTest results (entity level, exact span + type match)")
        print(ner_report(test_res))
        o = test_res["overall"]
        print(f"Overall P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
        json.dump([{"tokens": s["tokens"], "gold": s["tags"], "pred": p} for s, p in zip(splits["test"], test_pred)],
                  open(os.path.join(a.out, "biobert_test_predictions.json"), "w"))
    json.dump({"model": a.model, "lr": a.lr, "batch": a.batch, "epochs": a.epochs, "max_len": a.max_len, "seed": a.seed,
               "replace": a.replace, "mask": a.mask, "dev": dev_res, "test": test_res},
              open(metrics_path, "w"), indent=2)
    print("saved to", a.out)
    if IN_COLAB and not USE_DRIVE:   # no Drive: download the three result files straight to your computer
        from google.colab import files
        for f in ("training_log.json", "biobert_metrics.json", "biobert_dev_pred.json"):
            files.download(os.path.join(a.out, f))

    # cleanup: free GPU memory
    del model, loader, opt, sched, enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    finally:   # runs even if main() crashes, so whatever was written still reaches Drive
        if IN_COLAB and USE_DRIVE:
            try:
                drive.flush_and_unmount()   # forces all pending writes to reach Drive, then unmounts
                print("Drive flushed and unmounted")
            except Exception as e:
                print("Drive flush skipped:", e)
