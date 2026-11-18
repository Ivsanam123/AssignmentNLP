"""Side-by-side CRF vs BioBERT on the test set, overall and by training frequency.

The frequency breakdown is the point: the Part 2 error analysis showed CRF
recall collapsing from 0.971 on frequent strings to 0.498 on unseen ones, and
unseen strings accounting for 78.7% of all errors. This script says how much of
that gap BioBERT closes on its own, which is what the Part 3 contribution has
to improve on.

    python phase_3/biobert/compare_models.py

Reads crf_test_predictions.json and biobert_test_predictions.json from the
Part 2 root; writes phase_3/results/model_comparison.{txt,json}.
"""

import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import ner_metrics  # noqa: E402

OUT_DIR = os.path.join(ROOT, "phase_3", "results")
SYSTEMS = [("CRF", "crf_test_predictions.json"),
           ("BioBERT", "biobert_test_predictions.json")]

FREQ_ORDER = ["unseen (0)", "rare (1-4)", "medium (5-19)", "frequent (20+)"]


def freq_bucket(n):
    if n == 0:
        return "unseen (0)"
    if n < 5:
        return "rare (1-4)"
    if n < 20:
        return "medium (5-19)"
    return "frequent (20+)"


def load(path):
    with open(path) as f:
        return json.load(f)


def train_counts():
    """(lowercased string, type) -> gold mentions of it in training.

    Keyed by type as well as string, matching error_analysis.py, so the buckets
    here line up with the Part 2 report's numbers.
    """
    c = Counter()
    for s in load(os.path.join(ROOT, "data", "train.json")):
        for typ, a, b in ner_metrics.get_entities(s["tags"]):
            c[(" ".join(s["tokens"][a:b]).lower(), typ)] += 1
    return c


def recall_by_freq(pred_file, counts):
    """Exact-match recall per frequency bucket, over gold mentions."""
    data = load(os.path.join(ROOT, pred_file))
    stats = {b: [0, 0] for b in FREQ_ORDER}  # [correct, total]
    for s in data:
        got = set(ner_metrics.get_entities(s["pred"]))
        for ent in ner_metrics.get_entities(s["gold"]):
            typ, a, b = ent
            bucket = freq_bucket(counts[(" ".join(s["tokens"][a:b]).lower(), typ)])
            stats[bucket][1] += 1
            if ent in got:
                stats[bucket][0] += 1
    return {b: {"n": t, "correct": c, "recall": c / t if t else 0.0}
            for b, (c, t) in stats.items()}


def overall(pred_file):
    data = load(os.path.join(ROOT, pred_file))
    return ner_metrics.evaluate([s["gold"] for s in data], [s["pred"] for s in data])


def main():
    missing = [f for _, f in SYSTEMS if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        sys.exit(f"missing prediction file(s): {', '.join(missing)}\n"
                 "Run crf_baseline.py and/or the BioBERT job first.")

    counts = train_counts()
    results = {}
    lines = ["== Test set, entity level, exact span + type match ==", "",
             f"{'system':<10}{'P':>8}{'R':>8}{'F1':>8}{'Chem F1':>10}{'Dis F1':>9}"]

    for name, f in SYSTEMS:
        res = overall(f)
        o = res["overall"]
        results[name] = {"overall": res, "recall_by_freq": recall_by_freq(f, counts)}
        lines.append(f"{name:<10}{o['precision']:>8.4f}{o['recall']:>8.4f}{o['f1']:>8.4f}"
                     f"{res['per_type']['Chemical']['f1']:>10.4f}"
                     f"{res['per_type']['Disease']['f1']:>9.4f}")

    d = results["BioBERT"]["overall"]["overall"]["f1"] - results["CRF"]["overall"]["overall"]["f1"]
    lines += ["", f"BioBERT - CRF  =  {d:+.4f} F1", "",
              "== Recall by training frequency of the exact entity string ==", "",
              f"{'bucket':<18}{'n':>7}{'CRF':>9}{'BioBERT':>10}{'delta':>9}"]

    for b in FREQ_ORDER:
        c = results["CRF"]["recall_by_freq"][b]
        bb = results["BioBERT"]["recall_by_freq"][b]
        lines.append(f"{b:<18}{c['n']:>7}{c['recall']:>9.3f}{bb['recall']:>10.3f}"
                     f"{bb['recall'] - c['recall']:>+9.3f}")

    text = "\n".join(lines)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "model_comparison.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR}/model_comparison.{{txt,json}}")


if __name__ == "__main__":
    main()
