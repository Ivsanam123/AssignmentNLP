"""Inconsistent annotations in BC5CDR (ma'am's suggestion 2).

For every string that is annotated as an entity somewhere in TRAIN, count how often that exact token sequence
(case-insensitive) is
  - annotated exactly as an entity (with its type),
  - left completely unannotated (all O),
  - inside a longer/other annotated span (partial overlap).
A string is "inconsistent" when it occurs at least MIN_OCC times and the minority label share is at least MIN_MINORITY.
Type conflicts (Chemical vs Disease) are counted the same way. Flags are computed from TRAIN only.

Usage: python inconsistency_analysis.py [MIN_OCC] [MIN_MINORITY]
Writes: results/inconsistent_strings.csv, results/inconsistency_report.txt
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

from abbrev_lib import spans

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n):
    return json.load(open(os.path.join(HERE, "data", f"{n}.json"), encoding="utf-8"))


def occurrences(data, vocab_strings, maxn):
    """For each sentence, find occurrences of known strings. Returns dict string -> Counter(status) and sentence ids."""
    stat = defaultdict(Counter)
    where = defaultdict(lambda: defaultdict(list))   # string -> status -> [sentence idx]
    for si, s in enumerate(data):
        toks = [t.lower() for t in s["tokens"]]
        sp = spans(s["tags"])
        by_span = {(a, b): t for a, b, t in sp}
        n = len(toks)
        for i in range(n):
            for L in range(1, min(maxn, n - i) + 1):
                key = " ".join(toks[i:i + L])
                if key not in vocab_strings:
                    continue
                if (i, i + L) in by_span:
                    st = "ent:" + by_span[(i, i + L)]
                else:
                    cover = [t for (a, b, t) in sp if a < i + L and i < b]
                    st = "partial" if cover else "unannotated"
                stat[key][st] += 1
                where[key][st].append(si)
    return stat, where


def analyse(train, min_occ=5, min_minor=0.2):
    vocab, maxn = set(), 1
    for s in train:
        for a, b, t in spans(s["tags"]):
            vocab.add(" ".join(w.lower() for w in s["tokens"][a:b]))
            maxn = max(maxn, b - a)
    stat, where = occurrences(train, vocab, min(maxn, 6))
    rows = []
    for k, c in stat.items():
        tot = sum(c.values())
        n_c, n_d = c["ent:Chemical"], c["ent:Disease"]
        n_ent = n_c + n_d
        n_un, n_part = c["unannotated"], c["partial"]
        if tot < min_occ:
            continue
        # the two kinds of inconsistency, each as a minority share of the exact occurrences
        # (partial overlaps are listed but not used for flagging: they are often legitimate longer entities)
        exact = n_ent + n_un
        ann_rate = n_ent / exact if exact else 0.0
        presence_minor = min(ann_rate, 1 - ann_rate)
        type_minor = min(n_c, n_d) / n_ent if n_ent else 0.0
        kinds = []
        if exact >= min_occ and presence_minor >= min_minor:
            kinds.append("annotated-or-not")
        if n_ent >= min_occ and type_minor >= min_minor:
            kinds.append("type-conflict")
        if kinds:
            rows.append({"string": k, "occurrences": tot, "chemical": n_c, "disease": n_d, "unannotated": n_un,
                         "partial": n_part, "annotated_rate": round(ann_rate, 3), "kind": "+".join(kinds)})
    rows.sort(key=lambda r: (-r["occurrences"], r["string"]))
    return rows, where


def main():
    min_occ = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    min_minor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    train, dev, test = load("train"), load("dev"), load("test")
    rows, where = analyse(train, min_occ, min_minor)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "inconsistent_strings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    flagged = {r["string"] for r in rows}
    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    P(f"== Inconsistent strings (train only; >= {min_occ} exact occurrences, minority share >= {min_minor:.0%}) ==")
    P(f"  flagged strings: {len(rows)}")
    kinds = Counter(r["kind"] for r in rows)
    P("  by kind:", dict(kinds))
    P(f"\n{'string':28s} {'occ':>4s} {'Chem':>5s} {'Dis':>4s} {'unann':>5s} {'part':>4s} {'ann%':>5s}  kind")
    for r in rows[:40]:
        P(f"{r['string'][:28]:28s} {r['occurrences']:4d} {r['chemical']:5d} {r['disease']:4d} {r['unannotated']:5d} "
          f"{r['partial']:4d} {100*r['annotated_rate']:5.0f}  {r['kind']}")
    # how much training data is affected
    hit_sent = set()
    for k in flagged:
        for st, ids in where[k].items():
            hit_sent.update(ids)
    P(f"\n  train sentences containing a flagged string: {len(hit_sent)} / {len(train)} ({100*len(hit_sent)/len(train):.1f}%)")
    # does the inconsistency carry over to dev/test?  (annotation rate of the same strings in each split)
    vocab = flagged
    maxn = max(len(k.split()) for k in flagged)
    P("\n== Same strings in dev / test: annotated rate vs train ==")
    for name, data in (("train", train), ("dev", dev), ("test", test)):
        stat, _ = occurrences(data, vocab, maxn)
        ent = sum(c["ent:Chemical"] + c["ent:Disease"] for c in stat.values())
        un = sum(c["unannotated"] for c in stat.values())
        P(f"  {name}: exact occurrences {ent + un}, annotated {ent} ({100*ent/(ent+un):.1f}%)")
    # drift of individual strings between train and test
    st_tr, _ = occurrences(train, vocab, maxn)
    st_te, _ = occurrences(test, vocab, maxn)
    drift = []
    for k in flagged:
        a, b = st_tr[k], st_te[k]
        ta, tb = a["ent:Chemical"] + a["ent:Disease"] + a["unannotated"], b["ent:Chemical"] + b["ent:Disease"] + b["unannotated"]
        if ta >= 5 and tb >= 5:
            ra = (a["ent:Chemical"] + a["ent:Disease"]) / ta
            rb = (b["ent:Chemical"] + b["ent:Disease"]) / tb
            drift.append((abs(ra - rb), k, ra, rb, ta, tb))
    drift.sort(reverse=True)
    P("\n  largest train->test shifts in annotated rate (>=5 occurrences in both):")
    for d, k, ra, rb, ta, tb in drift[:12]:
        P(f"    {k:26s} train {100*ra:3.0f}% ({ta})   test {100*rb:3.0f}% ({tb})")
    open(os.path.join(HERE, "results", "inconsistency_report.txt"), "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
