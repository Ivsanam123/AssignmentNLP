"""CRF baseline for BC5CDR (chemical / disease NER).

Usage:
  python crf_baseline.py                 # train best config (c1=0.1, c2=0.01), evaluate dev+test, save model
  python crf_baseline.py --verbose       # also show the CRFsuite L-BFGS training log (nice for the demo)
  python crf_baseline.py --tune          # grid-search c1/c2 on dev first (9 combinations), then train best
  python crf_baseline.py --cli           # after training, open the interactive tagging prompt

Outputs (all relative to this file):
  models/crf_bc5cdr.crfsuite    trained model, used by ner_cli.py
  crf_test_predictions.json     per-sentence gold / predicted tags on the test set
  results/crf_metrics.json      dev + test precision / recall / F1 (overall and per type)

Requires: pip install sklearn-crfsuite        (no seqeval needed; see ner_metrics.py)
"""
import argparse
import itertools
import json
import os
import re
import time

import sklearn_crfsuite

import ner_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "crf_bc5cdr.crfsuite")
PRED_PATH = os.path.join(HERE, "crf_test_predictions.json")
METRICS_PATH = os.path.join(HERE, "results", "crf_metrics.json")

BEST_C1, BEST_C2, MAX_ITER = 0.1, 0.01, 100  # chosen on dev F1 (see CRF_paramater_tuning_results.txt)


# ----------------------------------------------------------------------------- data
def load(name):
    with open(os.path.join(HERE, "data", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------- features
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
    for off in (-2, -1, 1, 2):  # +-2 token context window
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


# --------------------------------------------------------------------------- model
def train_crf(X, y, c1=BEST_C1, c2=BEST_C2, iters=MAX_ITER, model_filename=None, verbose=False):
    kwargs = dict(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters,
                  all_possible_transitions=True, verbose=verbose)
    if model_filename:
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        kwargs["model_filename"] = model_filename
    crf = sklearn_crfsuite.CRF(**kwargs)
    crf.fit(X, y)
    return crf


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python crf_baseline.py` first (takes ~40 s).")
    return sklearn_crfsuite.CRF(model_filename=path)


def f1_of(gold, pred):
    return ner_metrics.evaluate(gold, pred)["overall"]["f1"]


def tune(Xtr, ytr, Xdev, ydev):
    best = (-1, None)
    for c1, c2 in itertools.product([0.01, 0.1, 0.5], [0.01, 0.1, 0.5]):
        crf = train_crf(Xtr, ytr, c1, c2)
        f1 = f1_of(ydev, crf.predict(Xdev))
        print(f"c1={c1} c2={c2} dev F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, (c1, c2))
    print(f"Best: c1,c2 = {best[1]} (dev F1 {best[0]:.4f})")
    return best[1]


def evaluate(crf, data, name):
    X, y = featurize(data)
    pred = crf.predict(X)
    res = ner_metrics.evaluate(y, pred)
    print(f"\n{name} results (entity level, exact span + type match)")
    print(ner_metrics.report(res))
    o = res["overall"]
    print(f"Overall P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    return pred, res


def save_predictions(data, pred, path=PRED_PATH):
    out = [{"tokens": s["tokens"], "gold": s["tags"], "pred": p} for s, p in zip(data, pred)]
    with open(path, "w") as f:
        json.dump(out, f)


# ------------------------------------------------------------------ tagging raw text
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize_with_offsets(text):
    """Same regex tokenisation used for the CLI: words and single punctuation marks."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def tag_text(crf, text):
    """Tag raw text. Returns list of dicts: {text, type, start, end} (character offsets)."""
    toks = tokenize_with_offsets(text)
    if not toks:
        return []
    words = [t[0] for t in toks]
    tags = crf.predict_single(sent2features(words))
    ents = []
    for typ, s, e in ner_metrics.get_entities(tags):
        cs, ce = toks[s][1], toks[e - 1][2]
        if not any(ch.isalnum() for ch in text[cs:ce]):  # ignore punctuation-only "entities"
            continue
        ents.append({"text": text[cs:ce], "type": typ, "start": cs, "end": ce})
    return ents


def extract_entities(tokens, tags):
    """(string, type) pairs from a tokenised sentence; kept for backwards compatibility."""
    return [(" ".join(tokens[s:e]), t) for t, s, e in ner_metrics.get_entities(tags)]


def cli(crf):
    print("\nType a sentence to tag (empty line to quit).")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        for ent in tag_text(crf, text):
            print(f"  {ent['type']:9s} {ent['text']}")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search c1/c2 on dev first")
    ap.add_argument("--verbose", action="store_true", help="show CRFsuite training log")
    ap.add_argument("--cli", action="store_true", help="open interactive prompt after training")
    ap.add_argument("--no-cli", action="store_true", help="(kept for compatibility; CLI is off by default)")
    args = ap.parse_args()

    train, dev, test = load("train"), load("dev"), load("test")
    print(f"Loaded BC5CDR: {len(train)} train / {len(dev)} dev / {len(test)} test sentences")
    t0 = time.time()
    Xtr, ytr = featurize(train)
    print(f"Extracted features for training set in {time.time() - t0:.1f}s")

    c1, c2 = BEST_C1, BEST_C2
    if args.tune:
        Xdev, ydev = featurize(dev)
        c1, c2 = tune(Xtr, ytr, Xdev, ydev)

    t0 = time.time()
    crf = train_crf(Xtr, ytr, c1, c2, model_filename=MODEL_PATH, verbose=args.verbose)
    print(f"\nTrained in {time.time() - t0:.1f}s (c1={c1}, c2={c2}, max_iter={MAX_ITER}); "
          f"model saved to {os.path.relpath(MODEL_PATH, HERE)}")

    _, dev_res = evaluate(crf, dev, "Dev")
    pred, test_res = evaluate(crf, test, "Test")
    save_predictions(test, pred)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"c1": c1, "c2": c2, "max_iterations": MAX_ITER, "dev": dev_res, "test": test_res}, f, indent=2)

    if args.cli:
        cli(crf)


if __name__ == "__main__":
    main()
