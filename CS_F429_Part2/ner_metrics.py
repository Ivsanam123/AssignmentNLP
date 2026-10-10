"""Entity-level NER metrics (no external dependencies).

Reproduces the default behaviour of `seqeval` (CoNLL `conlleval` style chunking):
an entity is a maximal run of B-X, I-X, I-X ... A stray I-X after O (or after a
different type) starts a new entity, exactly as conlleval does. An entity counts
as correct only if BOTH its span and its type match the gold entity exactly.

We wrote this module because `seqeval` fails to build on some machines
(setuptools `install_layout` error), and the demo must never depend on it.
Numbers were verified against seqeval-style output (see README / report).
"""
from collections import defaultdict


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
    prev_prefix, prev_type = "O", ""
    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel closes the last entity
        prefix, typ = _split(tag)
        # does the entity that was open end before position i?
        if cur_type is not None:
            ends = (
                prefix in ("B", "O")
                or (prefix == "I" and typ != cur_type)
            )
            if ends:
                ents.append((cur_type, start, i))
                cur_type, start = None, None
        # does a new entity start at position i?
        if prefix == "B" or (prefix == "I" and cur_type is None and typ):
            cur_type, start = typ, i
        prev_prefix, prev_type = prefix, typ
    return ents


def entity_sets(gold_seqs, pred_seqs):
    """Per-sentence sets of entities, tagged with the sentence index."""
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


def evaluate(gold_seqs, pred_seqs):
    """Return {'overall': {...}, 'per_type': {type: {...}}} with P/R/F1/support."""
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


def report(res, digits=4):
    """Pretty text table similar to seqeval.classification_report."""
    w = max([len(t) for t in res["per_type"]] + [9])
    head = f"{'':>{w}}  precision    recall  f1-score   support"
    lines = [head, ""]
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
