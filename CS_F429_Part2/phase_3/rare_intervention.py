import argparse, json, os, random, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
import numpy as np
import crf_baseline as B
import ner_metrics as NM

OUT = os.path.join(HERE, "results", "rare")
os.makedirs(OUT, exist_ok=True)
FREQ_BINS = [("unseen", 0, 0), ("rare 1-4", 1, 4), ("medium 5-19", 5, 19), ("frequent 20+", 20, 10**9)]


# ------------------------------------------------------------------ helpers
def ents(tags):                      # (type, start, end_exclusive)
    return NM.get_entities(tags)

def skey(tokens, a, b, t):
    return (" ".join(tokens[a:b]).lower(), t)

def entity_counts(train):
    c = Counter()
    for s in train:
        for t, a, b in ents(s["tags"]):
            c[skey(s["tokens"], a, b, t)] += 1
    return c

def inventory(train):
    inv = defaultdict(dict)          # type -> {lowercase string: original tokens}
    for s in train:
        for t, a, b in ents(s["tags"]):
            inv[t].setdefault(" ".join(s["tokens"][a:b]).lower(), s["tokens"][a:b])
    return {t: list(v.values()) for t, v in inv.items()}


# ------------------------------------------------------------------ augmentations
def drop_identity(feats, spans):
    """Remove the 'lower' feature of every token in `spans` (list of token indices), wherever it appears."""
    for i in spans:
        for off in (-2, -1, 0, 1, 2):          # token i is seen from position j=i-off with prefix f"{off:+d}:"
            j = i - off
            if 0 <= j < len(feats):
                feats[j].pop("lower" if off == 0 else f"{off:+d}:lower", None)
    return feats

def aug_idrop(train, p, rng):
    out = []
    for s in train:
        sp = [(a, b) for t, a, b in ents(s["tags"]) if rng.random() < p]
        if not sp:
            continue
        f = B.sent2features(s["tokens"])
        out.append((drop_identity(f, [i for a, b in sp for i in range(a, b)]), s["tags"]))
    return out

def aug_replace(train, p, rng, inv):
    out = []
    for s in train:
        toks, tags, new_t, new_g, pos, changed = s["tokens"], s["tags"], [], [], 0, False
        for t, a, b in ents(tags):
            new_t += toks[pos:a]; new_g += tags[pos:a]
            if rng.random() < p:
                rep = rng.choice(inv[t]); changed = True
            else:
                rep = toks[a:b]
            new_t += rep; new_g += [f"B-{t}"] + [f"I-{t}"] * (len(rep) - 1); pos = b
        new_t += toks[pos:]; new_g += tags[pos:]
        if changed:
            out.append((B.sent2features(new_t), new_g))
    return out

def aug_rare(train, counts, thr=4):
    out = []
    for s in train:
        if any(counts[skey(s["tokens"], a, b, t)] <= thr for t, a, b in ents(s["tags"])):
            out.append((B.sent2features(s["tokens"]), s["tags"]))
    return out


# ------------------------------------------------------------------ training / evaluation
def build_training(train, args):
    rng = random.Random(args.seed)
    X, y = B.featurize(train)
    counts, inv, extra = entity_counts(train), inventory(train), []
    for _ in range(args.k):
        if args.idrop > 0:   extra += aug_idrop(train, args.idrop, rng)
        if args.replace > 0: extra += aug_replace(train, args.replace, rng, inv)
    if args.rare:            extra += aug_rare(train, counts) * args.rare
    return X + [e[0] for e in extra], y + [e[1] for e in extra], len(extra)

def bins_recall(gold, pred, toks, counts):
    hit, tot = defaultdict(int), defaultdict(int)
    for g, p, t in zip(gold, pred, toks):
        P = set(ents(p))
        for e in ents(g):
            n = counts.get(skey(t, e[1], e[2], e[0]), 0)
            b = next(nm for nm, lo, hi in FREQ_BINS if lo <= n <= hi)
            tot[b] += 1; hit[b] += e in P
    return {b: hit[b] / tot[b] for b in tot}

def cmd_run(a):
    train, dev = B.load("train"), B.load("dev")
    t0 = time.time(); X, y, n_aug = build_training(train, a)
    crf = B.train_crf(X, y, model_filename=os.path.join(ROOT, "models", f"crf_rare_{a.name}.crfsuite"))
    gold = [s["tags"] for s in dev]
    pred = crf.predict([B.sent2features(s["tokens"]) for s in dev])
    r = NM.evaluate(gold, pred)
    rec = bins_recall(gold, pred, [s["tokens"] for s in dev], entity_counts(train))
    res = dict(name=a.name, args=vars(a), n_train=len(train), n_aug=n_aug, seconds=round(time.time() - t0),
               dev=r["overall"], per_type={t: v["f1"] for t, v in r["per_type"].items()}, recall_by_freq=rec)
    json.dump(res, open(os.path.join(OUT, f"{a.name}.json"), "w"), indent=1)
    json.dump([list(p) for p in pred], open(os.path.join(OUT, f"{a.name}_dev_pred.json"), "w"))
    print(f"{a.name:18s} +{n_aug:5d} aug  dev F1 {r['overall']['f1']:.4f} P {r['overall']['precision']:.4f} "
          f"R {r['overall']['recall']:.4f} | unseen R {rec['unseen']:.3f}  ({res['seconds']}s)", flush=True)

def sent_counts(gold, pred, toks, counts):
    """Per-sentence [tp, n_pred, n_gold, unseen_hit, unseen_n, long_hit, long_n] for paired bootstrap."""
    c = np.zeros((len(gold), 7))
    for i, (g, p, t) in enumerate(zip(gold, pred, toks)):
        P, G = set(ents(p)), ents(g)
        c[i, 0] = len(P & set(G)); c[i, 1] = len(P); c[i, 2] = len(G)
        for e in G:
            if counts.get(skey(t, e[1], e[2], e[0]), 0) == 0:
                c[i, 3] += e in P; c[i, 4] += 1
            if e[2] - e[1] >= 3:
                c[i, 5] += e in P; c[i, 6] += 1
    return c

def f1_of(c):
    tp, npd, ng = c[:3].sum(0) if c.ndim == 2 else c[:3]
    p, r = tp / npd, tp / ng
    return 2 * p * r / (p + r)

def cmd_summarize(a):
    train, dev = B.load("train"), B.load("dev")
    gold, toks, counts = [s["tags"] for s in dev], [s["tokens"] for s in dev], entity_counts(train)
    names = sorted(n[:-5] for n in os.listdir(OUT) if n.endswith(".json") and not n.endswith("_pred.json"))
    names = ["base"] + [n for n in names if n != "base"]
    C = {n: sent_counts(gold, json.load(open(os.path.join(OUT, f"{n}_dev_pred.json"))), toks, counts) for n in names}
    W = np.random.default_rng(0).multinomial(len(dev), np.ones(len(dev)) / len(dev), size=2000)
    def stat(c, w):
        s = w @ c
        p, r = s[0] / s[1], s[0] / s[2]
        return 2 * p * r / (p + r), s[3] / s[4], s[5] / s[6]
    base = np.array([stat(C["base"], w) for w in W])
    lines = ["DEV results (paired sentence bootstrap vs base, 2000 resamples; [95% CI] on the difference)",
             f"{'config':18s} {'F1':>7s} {'dF1 [CI]':>24s} {'unseenR':>8s} {'dUnseen [CI]':>24s} {'3+tokR':>7s} {'P':>6s} {'R':>6s}"]
    for n in names:
        s = stat(C[n], np.ones(len(dev))); bs = np.array([stat(C[n], w) for w in W]); d = bs - base
        ci = lambda k: f"{s[k]-stat(C['base'], np.ones(len(dev)))[k]:+.4f} [{np.percentile(d[:,k],2.5):+.4f},{np.percentile(d[:,k],97.5):+.4f}]"
        tp, npd, ng = C[n][:, :3].sum(0)
        lines.append(f"{n:18s} {s[0]:7.4f} {ci(0) if n!='base' else '-':>24s} {s[1]:8.3f} {ci(1) if n!='base' else '-':>24s} "
                     f"{s[2]:7.3f} {tp/npd:6.3f} {tp/ng:6.3f}")
    txt = "\n".join(lines); print(txt); open(os.path.join(OUT, "dev_summary.txt"), "w").write(txt)

def cmd_final(a):
    """The single test-set evaluation. Run only after the configuration has been chosen on dev."""
    train, test = B.load("train"), B.load("test")
    res = {}
    for name, model in (("original CRF", B.MODEL_PATH), (a.name, os.path.join(ROOT, "models", f"crf_rare_{a.name}.crfsuite"))):
        crf = B.load_model(model)
        gold = [s["tags"] for s in test]
        pred = crf.predict([B.sent2features(s["tokens"]) for s in test])
        r = NM.evaluate(gold, pred); rec = bins_recall(gold, pred, [s["tokens"] for s in test], entity_counts(train))
        res[name] = dict(overall=r["overall"], per_type={t: v for t, v in r["per_type"].items()}, recall_by_freq=rec)
        json.dump([list(p) for p in pred], open(os.path.join(OUT, f"TEST_{name.replace(' ', '_')}_pred.json"), "w"))
        o = r["overall"]; print(f"TEST {name:14s} P {o['precision']:.4f} R {o['recall']:.4f} F1 {o['f1']:.4f} | recall by freq {({k: round(v, 3) for k, v in rec.items()})}")
    json.dump(res, open(os.path.join(OUT, f"final_test_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--name", required=True)
    r.add_argument("--idrop", type=float, default=0.0); r.add_argument("--replace", type=float, default=0.0)
    r.add_argument("--rare", type=int, default=0, help="extra copies of sentences with rare entities")
    r.add_argument("--k", type=int, default=1, help="augmented copies of the training set"); r.add_argument("--seed", type=int, default=0)
    sub.add_parser("summarize")
    f = sub.add_parser("final"); f.add_argument("--name", required=True)
    a = ap.parse_args(); {"run": cmd_run, "summarize": cmd_summarize, "final": cmd_final}[a.cmd](a)
