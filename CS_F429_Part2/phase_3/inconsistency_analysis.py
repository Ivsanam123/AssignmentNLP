
"""Inconsistent annotations in BC5CDR (ma'am's suggestion 2).

Phrase-level analysis:
For every string annotated as an entity somewhere in TRAIN, count occurrences as:
  - exact entity annotation (with type),
  - completely unannotated,
  - partial overlap with another entity.

Word-level analysis:
Analyse individual tokens while preserving capitalization.
Tokens inside multiword entities are counted as part of that entity.

Flags are computed from TRAIN only.

Usage: python inconsistency_analysis.py [MIN_OCC] [MIN_MINORITY]

Writes:
  results/inconsistent_strings.csv
  results/inconsistency_report.txt
  results/inconsistent_words.csv
  results/all_word_contexts.txt
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

from abbrev_lib import spans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(n):
    return json.load(
        open(os.path.join(ROOT, "data", f"{n}.json"), encoding="utf-8")
    )

def occurrences(data, vocab_strings, maxn):
    """Find known phrases and classify their annotation status."""
    stat = defaultdict(Counter)
    where = defaultdict(lambda: defaultdict(list))

    for si, s in enumerate(data):
        toks = s["tokens"]
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
                    cover = [
                        t for (a, b, t) in sp
                        if a < i + L and i < b
                    ]
                    st = "partial" if cover else "unannotated"

                stat[key][st] += 1
                where[key][st].append(si)

    return stat, where


def analyse(train, min_occ=5, min_minor=0.2):
    vocab, maxn = set(), 1

    for s in train:
        for a, b, t in spans(s["tags"]):
            vocab.add(" ".join(s["tokens"][a:b]))
            maxn = max(maxn, b - a)

    stat, where = occurrences(train, vocab, min(maxn, 6))
    rows = []

    for k, c in stat.items():
        tot = sum(c.values())
        n_c = c["ent:Chemical"]
        n_d = c["ent:Disease"]
        n_ent = n_c + n_d
        n_un = c["unannotated"]
        n_part = c["partial"]

        if tot < min_occ:
            continue

        # Partial overlaps are reported but not used for flagging.
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
            rows.append({
                "string": k,
                "occurrences": tot,
                "chemical": n_c,
                "disease": n_d,
                "unannotated": n_un,
                "partial": n_part,
                "annotated_rate": round(ann_rate, 3),
                "kind": "+".join(kinds)
            })

    rows.sort(key=lambda r: (-r["occurrences"], r["string"]))
    return rows, where


def analyse_words(train, min_occ=5, min_minor=0.2):
    """Analyse individual tokens, preserving original capitalization.

    A token inside a multiword entity receives that entity's type.
    The context report retains the full sentence and entity boundaries.
    """
    stat = defaultdict(Counter)
    contexts = defaultdict(list)

    for si, s in enumerate(train):
        tokens = s["tokens"]
        entity_spans = spans(s["tags"])

        token_labels = {}

        for a, b, entity_type in entity_spans:
            for i in range(a, b):
                token_labels[i] = entity_type

        for i, token in enumerate(tokens):
            if i in token_labels:
                status = "ent:" + token_labels[i]
            else:
                status = "unannotated"

            stat[token][status] += 1

            contexts[token].append({
                "sentence_id": si,
                "token_index": i,
                "status": status,
                "sentence": " ".join(tokens),
                "entity": [
                    {
                        "start": a,
                        "end": b,
                        "type": entity_type,
                        "text": " ".join(tokens[a:b])
                    }
                    for a, b, entity_type in entity_spans
                    if a <= i < b
                ]
            })

    rows = []

    for token, counts in stat.items():
        total = sum(counts.values())
        chemical = counts["ent:Chemical"]
        disease = counts["ent:Disease"]
        unannotated = counts["unannotated"]
        annotated = chemical + disease

        if total < min_occ:
            continue

        annotation_rate = annotated / total if total else 0.0
        presence_minor = min(annotation_rate, 1 - annotation_rate)
        type_minor = min(chemical, disease) / annotated if annotated else 0.0

        kinds = []

        if presence_minor >= min_minor:
            kinds.append("annotated-or-not")

        if annotated >= min_occ and type_minor >= min_minor:
            kinds.append("type-conflict")

        if kinds:
            rows.append({
                "token": token,
                "occurrences": total,
                "chemical": chemical,
                "disease": disease,
                "unannotated": unannotated,
                "annotated_rate": round(annotation_rate, 3),
                "kind": "+".join(kinds)
            })

    rows.sort(
        key=lambda r: (
            -r["occurrences"],
            r["token"].lower(),
            r["token"]
        )
    )

    return rows, contexts


def main():
    min_occ = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    min_minor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

    train, dev, test = load("train"), load("dev"), load("test")

    rows, where = analyse(train, min_occ, min_minor)
    word_rows, word_contexts = analyse_words(train, min_occ, min_minor)

    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save original phrase-level CSV.
    phrase_fields = [
        "string", "occurrences", "chemical", "disease",
        "unannotated", "partial", "annotated_rate", "kind"
    ]

    with open(
        os.path.join(results_dir, "inconsistent_strings.csv"),
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=phrase_fields)
        writer.writeheader()
        writer.writerows(rows)

    # Save word-level CSV.
    word_fields = [
        "token", "occurrences", "chemical", "disease",
        "unannotated", "annotated_rate", "kind"
    ]

    with open(
        os.path.join(results_dir, "inconsistent_words.csv"),
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=word_fields)
        writer.writeheader()
        writer.writerows(word_rows)

    # Save full word-level contexts.
    with open(
        os.path.join(results_dir, "all_word_contexts.txt"),
        "w", encoding="utf-8"
    ) as f:
        for row in word_rows:
            token = row["token"]

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TOKEN: {token}\n")
            f.write(f"COUNTS: {row}\n")
            f.write("=" * 80 + "\n")

            for item in word_contexts[token]:
                f.write(
                    f"Sentence ID: {item['sentence_id']} | "
                    f"Token index: {item['token_index']} | "
                    f"Status: {item['status']}\n"
                )

                if item["entity"]:
                    f.write(f"Entity: {item['entity']}\n")

                f.write(f"Sentence: {item['sentence']}\n\n")

    # Original phrase-level report.
    flagged = {r["string"] for r in rows}

    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))

    P(
        f"== Inconsistent strings (train only; >= {min_occ} "
        f"occurrences, minority share >= {min_minor:.0%}) =="
    )
    P(f"  flagged strings: {len(rows)}")

    kinds = Counter(r["kind"] for r in rows)
    P("  by kind:", dict(kinds))

    P(
        f"\n{'string':28s} {'occ':>4s} {'Chem':>5s} "
        f"{'Dis':>4s} {'unann':>5s} {'part':>4s} {'ann%':>5s}  kind"
    )

    for r in rows[:40]:
        P(
            f"{r['string'][:28]:28s} {r['occurrences']:4d} "
            f"{r['chemical']:5d} {r['disease']:4d} "
            f"{r['unannotated']:5d} {r['partial']:4d} "
            f"{100 * r['annotated_rate']:5.0f}  {r['kind']}"
        )

    hit_sent = set()

    for k in flagged:
        for st, ids in where[k].items():
            hit_sent.update(ids)

    P(
        f"\n  train sentences containing a flagged string: "
        f"{len(hit_sent)} / {len(train)} "
        f"({100 * len(hit_sent) / len(train):.1f}%)"
    )

    # Compare phrase-level flagged strings across data splits.
    if flagged:
        maxn = max(len(k.split()) for k in flagged)

        P("\n== Same strings in dev / test: annotated rate vs train ==")

        for name, data in (
            ("train", train),
            ("dev", dev),
            ("test", test)
        ):
            split_stat, _ = occurrences(data, flagged, maxn)

            ent = sum(
                c["ent:Chemical"] + c["ent:Disease"]
                for c in split_stat.values()
            )
            un = sum(c["unannotated"] for c in split_stat.values())
            exact_total = ent + un

            rate = 100 * ent / exact_total if exact_total else 0.0

            P(
                f"  {name}: exact occurrences {exact_total}, "
                f"annotated {ent} ({rate:.1f}%)"
            )

        # Drift of individual strings between train and test.
        st_tr, _ = occurrences(train, flagged, maxn)
        st_te, _ = occurrences(test, flagged, maxn)

        drift = []

        for k in flagged:
            a, b = st_tr[k], st_te[k]

            ta = a["ent:Chemical"] + a["ent:Disease"] + a["unannotated"]
            tb = b["ent:Chemical"] + b["ent:Disease"] + b["unannotated"]

            if ta >= 5 and tb >= 5:
                ra = (
                    a["ent:Chemical"] + a["ent:Disease"]
                ) / ta

                rb = (
                    b["ent:Chemical"] + b["ent:Disease"]
                ) / tb

                drift.append((abs(ra - rb), k, ra, rb, ta, tb))

        drift.sort(reverse=True)

        P(
            "\n  largest train->test shifts in annotated rate "
            "(>=5 occurrences in both):"
        )

        for d, k, ra, rb, ta, tb in drift[:12]:
            P(
                f"    {k:26s} train {100 * ra:3.0f}% ({ta}) "
                f"  test {100 * rb:3.0f}% ({tb})"
            )

    else:
        P("\nNo phrase-level strings met the inconsistency thresholds.")

    with open(
        os.path.join(results_dir, "inconsistency_report.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(out))

    print("\n".join(out))

    print("\n== Word-level analysis ==")
    print(f"Flagged word tokens: {len(word_rows)}")
    print(
        f"Word-level contexts saved for all flagged tokens: "
        f"{os.path.join(results_dir, 'all_word_contexts.txt')}"
    )


if __name__ == "__main__":
    main()
