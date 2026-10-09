"""BC5CDR data pipeline: load, validate, print stats, save to JSON.

Usage: python data_pipeline.py
Run once on the HPC login node too (it needs internet) so compute nodes can read data/*.json offline.
"""
import json
import os
from collections import Counter

from datasets import load_dataset

# Label map for tner/bc5cdr. VERIFY against the dataset page on Hugging Face.
ID2LABEL = {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"}
OUT_DIR = "data"


def load_splits():
    with open("data_raw/label.json", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    splits = {}
    for name, fname in [("train", "train.json"), ("dev", "valid.json"), ("test", "test.json")]:
        sents = []
        with open(f"data_raw/{fname}", encoding="utf-8") as f:
            for line in f:  # one JSON object per line
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
    return {
        "sentences": len(sents),
        "tokens": sum(len(s["tokens"]) for s in sents),
        "mentions_total": sum(mentions.values()),
        "mentions_by_type": dict(mentions),
    }


def main():
    splits = load_splits()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, sents in splits.items():
        n_bad = validate(sents, name)
        print(f"\n== {name} ==  validation problems: {n_bad}")
        print(json.dumps(stats(sents), indent=2))
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(sents, f)

    # Example for the report: tokens with their BIO tags
    ex = next(s for s in splits["train"] if any(t != "O" for t in s["tags"]))
    print("\nExample sentence:")
    print(" ".join(ex["tokens"]))
    print([(w, t) for w, t in zip(ex["tokens"], ex["tags"]) if t != "O"])


if __name__ == "__main__":
    main()