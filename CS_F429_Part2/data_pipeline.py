"""BC5CDR data pipeline: load raw (tner/bc5cdr) files, validate, print stats, save to JSON.

Usage:
  python data_pipeline.py                 # data_raw/*.json  ->  data/{train,dev,test}.json
  python data_pipeline.py --out some_dir  # write somewhere else

Raw format (data_raw/*.json): one JSON object per line, {"tokens": [...], "tags": [int ids]}.
Label ids come from data_raw/label.json. Works fully offline (no `datasets` package needed).
"""
import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data_raw")


def load_splits(raw_dir=RAW_DIR):
    with open(os.path.join(raw_dir, "label.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    splits = {}
    for name, fname in [("train", "train.json"), ("dev", "valid.json"), ("test", "test.json")]:
        sents = []
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            for line in f:  # one JSON object per line
                if not line.strip():
                    continue
                ex = json.loads(line)
                sents.append({"tokens": ex["tokens"], "tags": [id2label[t] for t in ex["tags"]]})
        splits[name] = sents
    return splits


def validate(sents, name):
    """Check length match and BIO consistency. Returns number of problems."""
    problems = 0
    for i, s in enumerate(sents):
        if len(s["tokens"]) != len(s["tags"]):
            print(f"[{name}] sentence {i}: token/tag length mismatch")
            problems += 1
            continue
        prev = "O"
        for tag in s["tags"]:
            if tag.startswith("I-"):
                ent_type = tag[2:]
                if prev == "O" or prev[2:] != ent_type:
                    print(f"[{name}] sentence {i}: invalid BIO transition {prev} -> {tag}")
                    problems += 1
            prev = tag
    return problems


def stats(sents):
    mentions = Counter(t[2:] for s in sents for t in s["tags"] if t.startswith("B-"))
    n_tok = sum(len(s["tokens"]) for s in sents)
    n_ent_tok = sum(1 for s in sents for t in s["tags"] if t != "O")
    return {
        "sentences": len(sents),
        "tokens": n_tok,
        "avg_tokens_per_sentence": round(n_tok / max(len(sents), 1), 2),
        "entity_tokens": n_ent_tok,
        "pct_entity_tokens": round(100 * n_ent_tok / max(n_tok, 1), 2),
        "sentences_with_entity": sum(any(t != "O" for t in s["tags"]) for s in sents),
        "mentions_total": sum(mentions.values()),
        "mentions_by_type": dict(mentions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    splits = load_splits()
    os.makedirs(args.out, exist_ok=True)
    all_stats = {}
    for name, sents in splits.items():
        n_bad = validate(sents, name)
        print(f"\n== {name} ==  validation problems: {n_bad}")
        all_stats[name] = stats(sents)
        print(json.dumps(all_stats[name], indent=2))
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(sents, f)

    with open(os.path.join(args.out, "dataset_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    # Example for the report: tokens with their BIO tags
    ex = next(s for s in splits["train"] if any(t != "O" for t in s["tags"]))
    print("\nExample sentence:")
    print(" ".join(ex["tokens"]))
    print([(w, t) for w, t in zip(ex["tokens"], ex["tags"]) if t != "O"])


if __name__ == "__main__":
    main()
