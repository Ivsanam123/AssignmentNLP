"""CRF baseline for BC5CDR.

Usage:
  python crf_baseline.py            # train with default c1/c2, evaluate on test, then open CLI
  python crf_baseline.py --tune     # grid-search c1/c2 on dev first
  python crf_baseline.py --no-cli   # skip the interactive loop
Requires: pip install sklearn-crfsuite seqeval
"""
import argparse
import itertools
import json
import re
import time

import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def load(name):
    with open(f"data/{name}.json") as f:
        return json.load(f)


def shape(w):
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)  # collapse long runs


def token_feats(w, prefix=""):
    return {
        prefix + "lower": w.lower(),
        prefix + "suf2": w[-2:],
        prefix + "suf3": w[-3:],
        prefix + "pre2": w[:2],
        prefix + "pre3": w[:3],
        prefix + "isupper": w.isupper(),
        prefix + "istitle": w.istitle(),
        prefix + "hasdigit": any(c.isdigit() for c in w),
        prefix + "hashyphen": "-" in w,
        prefix + "shape": shape(w),
    }


def word2features(sent, i):
    f = {"bias": 1.0}
    f.update(token_feats(sent[i]))
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(sent):
            f.update(token_feats(sent[j], prefix=f"{off:+d}:"))
        else:
            f["BOS" if j < 0 else "EOS"] = True
    return f


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def featurize(data):
    return [sent2features(s["tokens"]) for s in data], [s["tags"] for s in data]


def train_crf(X, y, c1=0.1, c2=0.1, iters=100):
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters, all_possible_transitions=True
    )
    crf.fit(X, y)
    return crf


def tune(Xtr, ytr, Xdev, ydev):
    best = (-1, None)
    for c1, c2 in itertools.product([0.01, 0.1, 0.5], [0.01, 0.1, 0.5]):
        crf = train_crf(Xtr, ytr, c1, c2)
        f1 = f1_score(ydev, crf.predict(Xdev))
        print(f"c1={c1} c2={c2} dev F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, (c1, c2))
    print(f"Best: c1,c2 = {best[1]} (dev F1 {best[0]:.4f})")
    return best[1]


def evaluate(crf, data, name):
    X, y = featurize(data)
    pred = crf.predict(X)
    print(f"\n{name} results (entity level, seqeval)")
    print(classification_report(y, pred, digits=4))
    print(f"Overall P={precision_score(y, pred):.4f} R={recall_score(y, pred):.4f} F1={f1_score(y, pred):.4f}")
    return pred


def save_predictions(data, pred, path):
    out = [{"tokens": s["tokens"], "gold": s["tags"], "pred": p} for s, p in zip(data, pred)]
    with open(path, "w") as f:
        json.dump(out, f)


def extract_entities(tokens, tags):
    ents, cur, typ = [], [], None
    for w, t in zip(tokens, tags):
        if t.startswith("B-"):
            if cur:
                ents.append((" ".join(cur), typ))
            cur, typ = [w], t[2:]
        elif t.startswith("I-") and cur:
            cur.append(w)
        else:
            if cur:
                ents.append((" ".join(cur), typ))
            cur, typ = [], None
    if cur:
        ents.append((" ".join(cur), typ))
    return ents


def cli(crf):
    print("\nType a sentence to tag (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        tokens = re.findall(r"\w+|[^\w\s]", text)
        tags = crf.predict_single(sent2features(tokens))
        for ent, typ in extract_entities(tokens, tags):
            print(f"  {typ:9s} {ent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-cli", action="store_true")
    args = ap.parse_args()

    train, dev, test = load("train"), load("dev"), load("test")
    Xtr, ytr = featurize(train)
    c1, c2 = 0.1, 0.01
    if args.tune:
        Xdev, ydev = featurize(dev)
        c1, c2 = tune(Xtr, ytr, Xdev, ydev)

    t0 = time.time()
    crf = train_crf(Xtr, ytr, c1, c2)
    print(f"Trained in {time.time() - t0:.1f}s (c1={c1}, c2={c2})")

    evaluate(crf, dev, "Dev")
    pred = evaluate(crf, test, "Test")
    save_predictions(test, pred, "crf_test_predictions.json")

    if not args.no_cli:
        cli(crf)


if __name__ == "__main__":
    main()