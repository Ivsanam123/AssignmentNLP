"""Abbreviation table for BC5CDR (ma'am's suggestion 1).

Usage: python abbrev_analysis.py
Writes: results/abbreviation_table.csv, results/abbreviation_report.txt
Definition pairs are found from text only (all three splits). Entity types come from TRAIN gold labels only.
"""
import csv
import json
import os
from collections import Counter, defaultdict

from abbrev_lib import find_pairs, spans

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n):
    return json.load(open(os.path.join(HERE, "data", f"{n}.json"), encoding="utf-8"))


def main():
    splits = {n: load(n) for n in ("train", "dev", "test")}
    tr = splits["train"]
    # token-level type counts for every surface string, from train gold
    str_type = defaultdict(Counter)       # lower string -> Counter(type)
    str_occ = Counter()                   # lower single-token occurrences (annotated or not)
    for s in tr:
        sp = {(a, b): t for a, b, t in spans(s["tags"])}
        for (a, b), t in sp.items():
            str_type[" ".join(s["tokens"][a:b]).lower()][t] += 1
        for w in s["tokens"]:
            str_occ[w.lower()] += 1

    pairs = defaultdict(lambda: {"n": Counter(), "long_forms": Counter()})
    n_sent_with = Counter()
    for name, data in splits.items():
        for s in data:
            ps = find_pairs(s["tokens"])
            if ps:
                n_sent_with[name] += 1
            for p in ps:
                d = pairs[p["short"]]
                d["n"][name] += 1
                d["long_forms"][p["long"].lower()] += 1

    rows = []
    for short, d in pairs.items():
        lf = d["long_forms"].most_common(1)[0][0]
        st = str_type.get(short.lower(), Counter())
        lt = str_type.get(lf, Counter())
        n_total = sum(d["n"].values())
        # short-form entity share in train: annotated / all occurrences
        occ = str_occ.get(short.lower(), 0)
        rows.append({
            "short": short, "long_form": lf, "defs_train": d["n"]["train"], "defs_dev": d["n"]["dev"],
            "defs_test": d["n"]["test"], "defs_total": n_total,
            "short_type_train": "/".join(f"{t}:{c}" for t, c in st.most_common()) or "-",
            "long_type_train": "/".join(f"{t}:{c}" for t, c in lt.most_common()) or "-",
            "short_occurrences_train": occ,
            "short_annotated_train": sum(st.values()),
        })
    rows.sort(key=lambda r: (-r["defs_total"], r["short"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "abbreviation_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    P("== Abbreviation definitions found by Schwartz-Hearst ('long form ( SHORT )') ==")
    for n in ("train", "dev", "test"):
        P(f"  {n}: {n_sent_with[n]} / {len(splits[n])} sentences contain a definition")
    P(f"  distinct short forms: {len(rows)}")
    with_long_ent = sum(1 for r in rows if r["long_type_train"] != "-")
    with_short_ent = sum(1 for r in rows if r["short_type_train"] != "-")
    P(f"  short forms whose long form is an annotated train entity: {with_long_ent} ({100*with_long_ent/len(rows):.1f}%)")
    P(f"  short forms that are themselves annotated in train: {with_short_ent} ({100*with_short_ent/len(rows):.1f}%)")
    # how often is the short form's type the same as its long form's type (semantic transfer)
    agree = dis = 0
    for r in rows:
        if r["short_type_train"] != "-" and r["long_type_train"] != "-":
            a = r["short_type_train"].split("/")[0].split(":")[0]
            b = r["long_type_train"].split("/")[0].split(":")[0]
            agree += a == b
            dis += a != b
    P(f"  where both are annotated in train: type agrees {agree}, differs {dis}")
    P("\n== Top 40 abbreviations by definition count ==")
    P(f"{'short':8s} {'long form':45s} {'defs':>4s}  short type (train)      long type (train)")
    for r in rows[:40]:
        P(f"{r['short']:8s} {r['long_form'][:45]:45s} {r['defs_total']:4d}  {r['short_type_train']:22s}  {r['long_type_train']}")

    P("\n== Short strings annotated as BOTH Chemical and Disease in train ==")
    both = [(w, c) for w, c in str_type.items() if len(c) == 2 and len(w) <= 8]
    both.sort(key=lambda x: -sum(x[1].values()))
    for w, c in both[:15]:
        P(f"  {w:10s} {dict(c)}")
    open(os.path.join(HERE, "results", "abbreviation_report.txt"), "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
