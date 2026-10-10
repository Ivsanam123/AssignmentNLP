"""Dictionary-lookup baseline: tag any test string that is a known training entity (longest match first).

Usage: python dictionary_baseline.py
A deliberately simple lower bound. It shows how much of the CRF's performance could come from memorising
training entities, and its recall ceiling equals the share of test mentions seen in training.
Writes results/dictionary_baseline.json
"""
import json
import os
from collections import Counter, defaultdict

import ner_metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "data", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def build_lexicon(train):
    counts = defaultdict(Counter)  # lowercased token tuple -> type counts
    for s in train:
        for typ, a, b in M.get_entities(s["tags"]):
            counts[tuple(w.lower() for w in s["tokens"][a:b])][typ] += 1
    lex = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    return lex, max(len(k) for k in lex)


def tag(tokens, lex, max_len):
    low = [w.lower() for w in tokens]
    tags, i = ["O"] * len(tokens), 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            typ = lex.get(tuple(low[i:i + n]))
            if typ:
                tags[i] = f"B-{typ}"
                for j in range(i + 1, i + n):
                    tags[j] = f"I-{typ}"
                i += n
                break
        else:
            i += 1
    return tags


def main():
    train, test = load("train"), load("test")
    lex, max_len = build_lexicon(train)
    gold = [s["tags"] for s in test]
    pred = [tag(s["tokens"], lex, max_len) for s in test]
    res = M.evaluate(gold, pred)
    print(f"Lexicon: {len(lex)} distinct training entity strings (max length {max_len} tokens)")
    print(M.report(res))
    o = res["overall"]
    print(f"Overall P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "dictionary_baseline.json"), "w") as f:
        json.dump({"lexicon_size": len(lex), "test": res}, f, indent=2)


if __name__ == "__main__":
    main()
