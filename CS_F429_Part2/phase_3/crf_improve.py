"""Four-way CRF comparison: original / +abbreviation features / +inconsistency handling / both.
Usage: python crf_improve.py     (writes results/crf_improve.json, results/crf_improve_report.txt)
All lexicons and flags are built from TRAIN only. Settings chosen on dev; test reported once per final config."""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import crf_baseline as B
import ner_metrics
from abbrev_lib import find_pairs, spans
from inconsistency_analysis import analyse, occurrences

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- abbreviation features
def sent_strings(s):
    out = Counter()
    for a, b, t in spans(s["tags"]):
        out[(" ".join(s["tokens"][a:b]).lower(), t)] += 1
    return out

class Lex:
    def __init__(self, train):
        self.g = defaultdict(Counter)
        for s in train:
            for (k, t), c in sent_strings(s).items():
                self.g[k][t] += c
    def typ(self, k, own=None):
        c = Counter(self.g.get(k, {}))
        if own:
            for (kk, t), n in own.items():
                if kk == k: c[t] -= n
        c = {t: n for t, n in c.items() if n > 0}
        if not c: return "none"
        return max(c, key=c.get) if len(c) == 1 else "mixed"

def abbrev_feats(s, lex, train_mode):
    toks = s["tokens"]; n = len(toks)
    own = sent_strings(s) if train_mode else None     # leave-one-sentence-out for training
    extra = [dict() for _ in range(n)]
    for p in find_pairs(toks):
        long = p["long"].lower(); lt = lex.typ(long, own)
        short = p["short"].lower(); st = lex.typ(short, own)
        lw = toks[p["long_end"] - 1].lower()
        for i in range(p["long_start"], p["long_end"]):
            extra[i].update({"ab_inlong": True, "ab_long_type": lt, "ab_long_last": i == p["long_end"] - 1})
        for i in range(*p["short_pos"]):
            extra[i].update({"ab_short": True, "ab_long_type": lt, "ab_short_lex": st,
                             "ab_long_lastword": lw, "ab_long_suf3": lw[-3:]})
    return extra

def featurize_ab(data, lex, train_mode):
    X = []
    for s in data:
        base = B.sent2features(s["tokens"]); ex = abbrev_feats(s, lex, train_mode)
        for f, e in zip(base, ex): f.update(e)
        X.append(base)
    return X, [s["tags"] for s in data]

# ---------------------------------------------------------------- inconsistency removal
def removal(train, flagged, where, mode):
    """R1: drop every train sentence containing a flagged string. R2: drop only sentences where the flagged string's
    status is the minority one (e.g. 'death' annotated 26x, unannotated 4x -> drop the 4)."""
    drop = set()
    for k in flagged:
        w = where[k]
        if mode == "R1":
            for ids in w.values(): drop.update(ids)
        else:
            cnt = {st: len(ids) for st, ids in w.items() if st in ("unannotated",) or st.startswith("ent:")}
            ent = sum(v for st, v in cnt.items() if st.startswith("ent:"))
            un = cnt.get("unannotated", 0)
            if un and ent:
                if un <= ent: drop.update(w.get("unannotated", []))
                else:
                    for st, ids in w.items():
                        if st.startswith("ent:"): drop.update(ids)
            # type conflicts: drop the minority type
            ents = {st: len(ids) for st, ids in w.items() if st.startswith("ent:")}
            if len(ents) == 2:
                minor = min(ents, key=ents.get); drop.update(w[minor])
    return [s for i, s in enumerate(train) if i not in drop], len(drop)

def sents_with(data, flagged):
    maxn = max(len(k.split()) for k in flagged)
    st, wh = occurrences(data, flagged, maxn)
    ids = set()
    for k in wh:
        for v in wh[k].values(): ids.update(v)
    return ids

def f1(gold, pred): return ner_metrics.evaluate(gold, pred)["overall"]

def run(train, evals, use_ab, name):
    lex = Lex(train)
    if use_ab: Xtr, ytr = featurize_ab(train, lex, True)
    else: Xtr, ytr = B.featurize(train)
    crf = B.train_crf(Xtr, ytr)
    res = {}
    for en, d in evals.items():
        X, y = featurize_ab(d, lex, False) if use_ab else B.featurize(d)
        r = ner_metrics.evaluate(y, crf.predict(X))
        res[en] = {"p": r["overall"]["precision"], "r": r["overall"]["recall"], "f1": r["overall"]["f1"],
                   "per_type": {t: v["f1"] for t, v in r["per_type"].items()}}
    print(name, {k: round(v["f1"], 4) for k, v in res.items()}, flush=True)
    return res

def main():
    train, dev, test = B.load("train"), B.load("dev"), B.load("test")
    out = []; P = lambda *a: (out.append(" ".join(map(str, a))), print(*a, flush=True))
    # choose inconsistency setting on dev (standard dev F1), CRF with original features
    cands = {}
    for minor in (0.2, 0.1):
        rows, where = analyse(train, 5, minor)
        flagged = {r["string"] for r in rows}
        for mode in ("R1", "R2"):
            tr2, nd = removal(train, flagged, where, mode)
            cands[(minor, mode)] = (tr2, nd, flagged)
    P("== Choosing inconsistency handling on dev (original features) ==")
    best, bestf = None, -1
    for key, (tr2, nd, fl) in cands.items():
        r = run(tr2, {"dev": dev}, False, f"minority>={key[0]} {key[1]} (dropped {nd} train sents, {len(fl)} strings)")
        if r["dev"]["f1"] > bestf: best, bestf = key, r["dev"]["f1"]
    P("  chosen:", best)
    tr_inc, nd, flagged = cands[best]
    hd, ht = sents_with(dev, flagged), sents_with(test, flagged)
    clean_dev = [s for i, s in enumerate(dev) if i not in hd]
    clean_test = [s for i, s in enumerate(test) if i not in ht]
    P(f"  clean dev {len(clean_dev)}/{len(dev)}, clean test {len(clean_test)}/{len(test)} sentences (flagged strings removed)")
    evals = {"dev": dev, "test": test, "clean_test": clean_test}
    systems = {"A original CRF": (train, False), "B +abbreviation features": (train, True),
               "C +inconsistency handling": (tr_inc, False), "D both": (tr_inc, True)}
    allres = {}
    P("\n== Four-way comparison ==")
    for n, (tr, ab) in systems.items():
        allres[n] = run(tr, evals, ab, n)
    P(f"\n{'system':30s} {'devF1':>7s} {'testP':>7s} {'testR':>7s} {'testF1':>7s} {'Chem':>7s} {'Dis':>7s} {'cleanTestF1':>11s}")
    for n, r in allres.items():
        t = r["test"]; pt = t["per_type"]
        P(f"{n:30s} {r['dev']['f1']:7.4f} {t['p']:7.4f} {t['r']:7.4f} {t['f1']:7.4f} {pt.get('Chemical',0):7.4f} {pt.get('Disease',0):7.4f} {r['clean_test']['f1']:11.4f}")
    json.dump({"chosen": list(map(str, best)), "results": allres}, open(os.path.join(HERE, "results", "crf_improve.json"), "w"), indent=2)
    open(os.path.join(HERE, "results", "crf_improve_report.txt"), "w").write("\n".join(out))

if __name__ == "__main__":
    main()
