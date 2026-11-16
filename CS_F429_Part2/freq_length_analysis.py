#!/usr/bin/env python3
"""
Frequency x length analysis of NER errors (CS F429, Part 3, step 1).

Question: how well does a model recognise rare vs common and short vs long
entities, and which of the two factors really matters?

Design rules
  * Test set is NEVER read here. Only train (to define "rarity") and dev (to evaluate).
  * The analysis is separate from the model, so it can be re-run for BioBERT
    with the same bins: train the CRF here (--model crf) or pass predictions
    from any model (--pred_file preds.json).

Usage
  python freq_length_analysis.py --data_dir data --out_dir results_freq_length
  python freq_length_analysis.py --data_dir data --pred_file biobert_dev_preds.json --name biobert

Data format: data/train.json and data/dev.json, each a list of
  {"tokens": [...], "tags": [...]}  (tags = BIO strings or integer ids; key
  may also be "ner_tags"/"labels"). JSON-lines is accepted too.
  A pred_file is a JSON list (one entry per dev sentence) of BIO-tag lists.
"""
import argparse, json, math, os, re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:   # prefer the project's own metric code (seqeval can fail to build on some machines)
    import ner_metrics as _nm
    def get_entities(tags):
        return [(t, s, e - 1) for t, s, e in _nm.get_entities(tags)]   # -> inclusive end
except ImportError:
    _nm = None
    from seqeval.metrics.sequence_labeling import get_entities

# T-NER label map used in Part 2
ID2LABEL = {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"}
FREQ_BINS = [("unseen (0)", 0, 0), ("rare (1-4)", 1, 4), ("medium (5-19)", 5, 19), ("frequent (20+)", 20, 10**9)]
LEN_BINS = [("1 token", 1, 1), ("2 tokens", 2, 2), ("3 tokens", 3, 3), ("4+ tokens", 4, 10**9)]
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- data
def load_split(path):
    """Tolerant loader: list / JSON-lines / {"data": [...]}; tokens under tokens|words,
    tags under tags|ner_tags|labels|bio_tags|bio (strings or int ids)."""
    with open(path) as f:
        txt = f.read().strip()
    rows = json.loads(txt) if txt[0] in "[{" and "\n{" not in txt[:2000] else [json.loads(l) for l in txt.splitlines() if l.strip()]
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))
    toks, tags = [], []
    for r in rows:
        t = r.get("tokens", r.get("words"))
        g = next((r[k] for k in ("tags", "ner_tags", "labels", "bio_tags", "bio") if k in r), None)
        if t is None or g is None:
            raise KeyError(f"unrecognised schema, keys = {list(r)}; edit load_split()")
        g = [ID2LABEL[x] if isinstance(x, int) else x for x in g]
        assert len(t) == len(g), "tokens/tags length mismatch"
        toks.append(t); tags.append(g)
    return toks, tags


# ----------------------------------------------------------------- CRF (features exactly as in Part 2 report)
def word_shape(w):
    s = re.sub("[A-Z]", "X", w); s = re.sub("[a-z]", "x", s); s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)

def tok_feats(w):
    return {"w": w.lower(), "p2": w[:2], "p3": w[:3], "s2": w[-2:], "s3": w[-3:],
            "up": w.isupper(), "ti": w.istitle(), "dg": any(c.isdigit() for c in w),
            "hy": "-" in w, "sh": word_shape(w)}

def sent_feats(tokens):
    base = [tok_feats(w) for w in tokens]
    out = []
    for i in range(len(tokens)):
        f = {"bias": 1.0, **base[i]}
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(tokens):
                f.update({f"{off}:{k}": v for k, v in base[j].items()})
            else:
                f["BOS" if off < 0 else "EOS"] = True
        out.append(f)
    return out

def train_crf(train_toks, train_tags, c1=0.1, c2=0.01):
    import sklearn_crfsuite
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=100,
                               all_possible_transitions=True)
    crf.fit([sent_feats(t) for t in train_toks], train_tags)
    return crf


# ----------------------------------------------------------------- entity bookkeeping
def key(tokens, s, e, typ):
    return (" ".join(tokens[s:e + 1]).lower(), typ)

def is_acronym(tokens, s, e):
    if e != s:
        return False
    w = tokens[s]
    letters = [c for c in w if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)

def build_train_stats(train_toks, train_tags):
    ent_count, vocab = Counter(), Counter()
    for t, g in zip(train_toks, train_tags):
        vocab.update(w.lower() for w in t)
        for typ, s, e in get_entities(g):
            ent_count[key(t, s, e, typ)] += 1
    return ent_count, vocab

def bin_of(x, bins):
    for name, lo, hi in bins:
        if lo <= x <= hi:
            return name

def outcome(g, preds):
    typ, s, e = g
    if g in preds:
        return "correct"
    for (pt, ps, pe) in preds:
        if ps == s and pe == e:
            return "type error"
    over = [(pt, ps, pe) for (pt, ps, pe) in preds if ps <= e and s <= pe]
    if any(pt == typ for pt, _, _ in over):
        return "boundary error"
    return "boundary+type" if over else "missed"

def gold_table(toks, gold, pred, ent_count, vocab):
    rows = []
    for i, (t, g, p) in enumerate(zip(toks, gold, pred)):
        pset = set(get_entities(p))
        for ent in get_entities(g):
            typ, s, e = ent
            k = key(t, s, e, typ)
            n = ent_count.get(k, 0)
            oov = any(vocab.get(w.lower(), 0) == 0 for w in t[s:e + 1])
            rows.append(dict(sent=i, type=typ, length=e - s + 1, train_count=n,
                             freq_bin=bin_of(n, FREQ_BINS), len_bin=bin_of(e - s + 1, LEN_BINS),
                             novelty=("seen string" if n > 0 else
                                      "unseen string, has OOV word" if oov else "unseen string, known words"),
                             acronym=is_acronym(t, s, e), outcome=outcome(ent, pset), string=k[0]))
    df = pd.DataFrame(rows)
    df["correct"] = (df.outcome == "correct").astype(int)
    return df

def pred_table(toks, gold, pred, ent_count):
    rows = []
    for i, (t, g, p) in enumerate(zip(toks, gold, pred)):
        gset = set(get_entities(g))
        for ent in get_entities(p):
            typ, s, e = ent
            n = ent_count.get(key(t, s, e, typ), 0)
            rows.append(dict(sent=i, length=e - s + 1, freq_bin=bin_of(n, FREQ_BINS),
                             len_bin=bin_of(e - s + 1, LEN_BINS), correct=int(ent in gset)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- statistics
def boot_recall(df, col, order, B=1000):
    """Recall per bin with a sentence-level bootstrap 95% CI."""
    sents = df.sent.unique(); S = len(sents); idx = {s: j for j, s in enumerate(sents)}
    n = np.zeros((S, len(order))); c = np.zeros((S, len(order)))
    for r in df.itertuples():
        b = order.index(getattr(r, col)); n[idx[r.sent], b] += 1; c[idx[r.sent], b] += r.correct
    W = RNG.multinomial(S, np.ones(S) / S, size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = (W @ c) / (W @ n)
    lo, hi = np.nanpercentile(rec, [2.5, 97.5], axis=0)
    return n.sum(axis=0), c.sum(axis=0), lo, hi

def summarise(df, col, order, total_err):
    n, c, lo, hi = boot_recall(df, col, order)
    out = pd.DataFrame({"bin": order, "n_gold": n.astype(int), "correct": c.astype(int)})
    out["recall"] = out.correct / out.n_gold
    out["ci_low"], out["ci_high"] = lo, hi
    out["errors"] = out.n_gold - out.correct
    out["pct_of_gold"] = 100 * out.n_gold / out.n_gold.sum()
    out["pct_of_all_errors"] = 100 * out.errors / total_err
    return out

def outcome_mix(df, col, order):
    ct = pd.crosstab(df[col], df.outcome).reindex(order).fillna(0)
    for o in ["correct", "boundary error", "boundary+type", "type error", "missed"]:
        if o not in ct: ct[o] = 0
    ct = ct[["correct", "boundary error", "boundary+type", "type error", "missed"]]
    return (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)

def auc(y, score):
    """Rank-based AUC (probability a correct entity scores above a missed one)."""
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def logit_or(df, B=300):
    """Odds ratios from a logistic regression, CIs by sentence-level bootstrap."""
    from sklearn.linear_model import LogisticRegression
    def design(d):
        return np.c_[(d.train_count == 0).astype(float), np.minimum(d.length, 4) - 1,
                     (d.type == "Disease").astype(float), d.acronym.astype(float)]
    names = ["unseen string (vs seen)", "+1 token of length (1->4+)", "Disease (vs Chemical)", "acronym"]
    def fit(d):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(design(d), d.correct)
        return np.exp(m.coef_[0])
    point = fit(df)
    groups = {s: g for s, g in df.groupby("sent")}; keys = list(groups)
    boots = []
    for _ in range(B):
        samp = RNG.choice(len(keys), len(keys))
        boots.append(fit(pd.concat([groups[keys[j]] for j in samp])))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"factor": names, "odds_ratio": point, "ci_low": lo, "ci_high": hi}), design

# ----------------------------------------------------------------- plots
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#c9c8c2", "#0b0b0b", "#52514e"

def style(ax, ylabel=None):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY); ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED); ax.yaxis.grid(True, color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED)

def bar_recall(ax, t, title, color=BLUE):
    x = np.arange(len(t))
    ax.bar(x, t.recall, color=color, width=0.55)
    ax.errorbar(x, t.recall, yerr=[t.recall - t.ci_low, t.ci_high - t.recall], fmt="none", ecolor=INK, capsize=4, lw=1.2)
    for i, r in enumerate(t.itertuples()):
        ax.text(i, min(r.ci_high + 0.02, 1.02), f"{r.recall:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.text(i, 0.02, f"n={r.n_gold}", ha="center", va="bottom", color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(t.bin, fontsize=9); ax.set_ylim(0, 1.12)
    ax.set_title(title, loc="left", fontsize=11, color=INK); style(ax, "Entity recall")

def make_figures(out, ft, lt, grid, share, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bar_recall(ax[0], ft, "Recall by training frequency"); bar_recall(ax[1], lt, "Recall by entity length")
    fig.suptitle(f"{name}: dev-set recall (bars: 95% bootstrap CI)", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_recall_by_freq_and_length.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8)); w = 0.36
    labels = [b[0] for b in LEN_BINS]; x = np.arange(len(labels))
    for k, (grp, col, off) in enumerate([("seen", BLUE, -w / 2), ("unseen", ORANGE, w / 2)]):
        sub = grid[grid.group == grp].set_index("len_bin").reindex(labels)
        ax.bar(x + off, sub.recall.fillna(0), w, color=col, label=f"{grp} in training")
        for i, r in enumerate(sub.itertuples()):
            if r.n_gold and r.n_gold > 0:
                ax.text(i + off, r.recall + 0.015, f"{r.recall:.2f}\n(n={int(r.n_gold)})", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.32)
    ax.set_title("Length effect once frequency is held fixed", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2); style(ax, "Entity recall")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_length_given_seen_unseen.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8)); x = np.arange(len(share)); w = 0.36
    ax.bar(x - w / 2, share.pct_of_gold, w, color=GREY, label="share of gold mentions")
    ax.bar(x + w / 2, share.pct_of_all_errors, w, color=ORANGE, label="share of all missed/wrong mentions")
    for i, r in enumerate(share.itertuples()):
        ax.text(i - w / 2, r.pct_of_gold + 1, f"{r.pct_of_gold:.0f}%", ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2, r.pct_of_all_errors + 1, f"{r.pct_of_all_errors:.0f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(share.bin, fontsize=9); ax.set_ylim(0, max(share.pct_of_all_errors.max(), share.pct_of_gold.max()) + 12)
    ax.set_title("Where the errors come from (by training frequency)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9); style(ax, "% of mentions")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_error_share_by_frequency.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data"); ap.add_argument("--out_dir", default="results_freq_length")
    ap.add_argument("--train_file", default="train.json"); ap.add_argument("--dev_file", default="dev.json")
    ap.add_argument("--pred_file", default=None, help="dev predictions (JSON list of BIO lists); if absent a CRF is trained")
    ap.add_argument("--name", default="CRF")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tr_t, tr_g = load_split(os.path.join(a.data_dir, a.train_file))
    dv_t, dv_g = load_split(os.path.join(a.data_dir, a.dev_file))
    if a.pred_file:
        pred = json.load(open(a.pred_file)); assert len(pred) == len(dv_t)
        print(f"loaded predictions from {a.pred_file}")
    else:
        pred = None
        try:   # use the real Part 2 model + features, so numbers match crf_baseline.py exactly
            import crf_baseline as cb
            if os.path.exists(cb.MODEL_PATH):
                crf = cb.load_model(); print("using saved Part 2 model", cb.MODEL_PATH)
            else:
                print("training with crf_baseline.train_crf ..."); crf = cb.train_crf(*cb.featurize([{"tokens": t, "tags": g} for t, g in zip(tr_t, tr_g)]))
            pred = crf.predict([cb.sent2features(t) for t in dv_t])
        except ImportError:
            print("crf_baseline.py not found: training a stand-in CRF with built-in features")
            pred = train_crf(tr_t, tr_g).predict([sent_feats(t) for t in dv_t])
    json.dump([list(p) for p in pred], open(f"{a.out_dir}/dev_predictions_{a.name}.json", "w"))
    if _nm:
        o = _nm.evaluate(dv_g, pred)["overall"]
        print(f"CHECK (project ner_metrics): dev P {o['precision']:.4f} R {o['recall']:.4f} F1 {o['f1']:.4f}  <- should be 0.8277 for the Part 2 CRF")

    ent_count, vocab = build_train_stats(tr_t, tr_g)
    gd = gold_table(dv_t, dv_g, pred, ent_count, vocab); pdf = pred_table(dv_t, dv_g, pred, ent_count)
    tp, ng, npred = gd.correct.sum(), len(gd), len(pdf)
    P, R = tp / npred, tp / ng; F1 = 2 * P * R / (P + R)
    total_err = ng - tp
    lines = [f"== {a.name} on DEV (test set not used) ==",
             f"gold mentions {ng}, predicted {npred}, entity-level P {P:.4f}  R {R:.4f}  F1 {F1:.4f}",
             f"dev mentions whose exact string is unseen in train: {100 * (gd.train_count == 0).mean():.1f}%", ""]

    ft = summarise(gd, "freq_bin", [b[0] for b in FREQ_BINS], total_err)
    lt = summarise(gd, "len_bin", [b[0] for b in LEN_BINS], total_err)
    nov = summarise(gd, "novelty", ["seen string", "unseen string, known words", "unseen string, has OOV word"], total_err)
    gd["acr"] = np.where(gd.acronym, "acronym", "not acronym")
    acr = summarise(gd, "acr", ["acronym", "not acronym"], total_err)
    typ = summarise(gd, "type", ["Chemical", "Disease"], total_err)

    # seen/unseen x length grid
    gd["group"] = np.where(gd.train_count > 0, "seen", "unseen"); grid = []
    for grp in ("seen", "unseen"):
        for nme, lo, hi in LEN_BINS:
            d = gd[(gd.group == grp) & (gd.len_bin == nme)]
            grid.append(dict(group=grp, len_bin=nme, n_gold=len(d), recall=d.correct.mean() if len(d) else np.nan))
    grid = pd.DataFrame(grid)
    # frequency x length (4x4) for completeness
    fi, lc = [b[0] for b in FREQ_BINS], [b[0] for b in LEN_BINS]
    fl_rec = gd.pivot_table(index="freq_bin", columns="len_bin", values="correct", aggfunc="mean").reindex(index=fi, columns=lc)
    fl_n = gd.pivot_table(index="freq_bin", columns="len_bin", values="correct", aggfunc="count").reindex(index=fi, columns=lc).fillna(0).astype(int)

    # predicted-side precision
    pp = pdf.groupby("freq_bin").agg(n_pred=("correct", "size"), precision=("correct", "mean")).reindex([b[0] for b in FREQ_BINS])
    pl = pdf.groupby("len_bin").agg(n_pred=("correct", "size"), precision=("correct", "mean")).reindex([b[0] for b in LEN_BINS])

    # which factor explains success? (AUC + odds ratios)
    y = gd.correct.values
    a_len = auc(y, np.minimum(gd.length.values, 4) * -1.0)
    a_frq = auc(y, np.log1p(gd.train_count.values))
    from sklearn.linear_model import LogisticRegression
    orr, design = logit_or(gd)
    X = np.c_[np.log1p(gd.train_count), np.minimum(gd.length, 4)]
    a_both = auc(y, LogisticRegression(C=1e6, max_iter=1000).fit(X, y).decision_function(X))

    # headroom: overall recall if one bin were lifted to the recall of the 'frequent' bin or +10 points
    unseen = ft.iloc[0]
    head = {f"overall recall if unseen recall +10 pts": R + 0.10 * unseen.n_gold / ng,
            "overall recall if unseen recall = recall of frequent strings": R + (ft.iloc[3].recall - unseen.recall) * unseen.n_gold / ng,
            "overall recall if 4+ token recall = 2-3 token recall": R + (lt.iloc[1:3].correct.sum() / lt.iloc[1:3].n_gold.sum() - lt.iloc[3].recall) * lt.iloc[3].n_gold / ng}

    def block(title, df):
        lines.extend([f"-- {title}", df.round(3).to_string(index=False), ""])
    block("Recall by training frequency of the exact (lower-cased, same-type) string", ft)
    block("Recall by entity length", lt)
    block("Recall by novelty of the string (OOV = a word never seen in train text)", nov)
    block("Recall: acronym vs not", acr); block("Recall by type", typ)
    lines += ["-- Recall for seen vs unseen strings at each length", grid.round(3).to_string(index=False), "",
              "-- Frequency x length: recall", fl_rec.round(3).to_string(), "", "-- Frequency x length: number of gold mentions", fl_n.to_string(), "",
              "-- Outcome mix (%) by frequency bin", outcome_mix(gd, "freq_bin", [b[0] for b in FREQ_BINS]).to_string(), "",
              "-- Outcome mix (%) by length bin", outcome_mix(gd, "len_bin", [b[0] for b in LEN_BINS]).to_string(), "",
              "-- Precision of PREDICTED entities by training frequency of predicted string", pp.round(3).to_string(), "",
              "-- Precision of PREDICTED entities by length", pl.round(3).to_string(), "",
              f"-- AUC for 'will this gold entity be recognised?': length only {a_len:.3f} | frequency only {a_frq:.3f} | both {a_both:.3f}",
              "-- Logistic regression odds ratios (>1 helps, <1 hurts; 95% sentence-bootstrap CI)", orr.round(3).to_string(index=False), "",
              "-- Headroom"] + [f"{k}: {v:.4f}" for k, v in head.items()]
    top = (gd[gd.outcome != "correct"].groupby(["string", "type"]).size().sort_values(ascending=False).head(25))
    lines += ["", "-- 25 most frequently missed/wrong gold strings (dev)", top.to_string()]
    txt = "\n".join(lines); print(txt)

    open(f"{a.out_dir}/freq_length_report_{a.name}.txt", "w").write(txt)
    for nme, d in [("freq", ft), ("length", lt), ("novelty", nov), ("acronym", acr), ("seen_unseen_x_length", grid)]:
        d.to_csv(f"{a.out_dir}/table_{nme}_{a.name}.csv", index=False)
    gd.drop(columns=["acr"]).to_csv(f"{a.out_dir}/gold_entities_{a.name}.csv", index=False)
    make_figures(a.out_dir, ft, lt, grid, ft, a.name)
    print(f"\nsaved to {a.out_dir}/")

if __name__ == "__main__":
    main()
