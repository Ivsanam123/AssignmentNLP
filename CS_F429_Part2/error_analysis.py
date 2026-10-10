"""Error analysis of the CRF baseline on the BC5CDR test set.

Usage:  python error_analysis.py          (needs crf_test_predictions.json from crf_baseline.py)

Writes:
  results/error_analysis.json     all numbers
  results/error_examples.md       qualitative examples for each error type
  figures/*.png                   plots used in the report
  report/tables_generated.tex     LaTeX tables generated from the numbers (so nothing is hand-copied)

Error taxonomy (entity level, exact-match scoring):
  gold entity  -> correct | type error (right span, wrong type) | boundary error (overlaps a prediction
                  of the same type but span differs) | boundary+type | missed (no overlapping prediction)
  predicted    -> correct | type error | boundary error | spurious (overlaps no gold entity)
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

import ner_metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG, REP = (os.path.join(HERE, d) for d in ("results", "figures", "report"))
for d in (RES, FIG, REP):
    os.makedirs(d, exist_ok=True)

TYPES = ["Chemical", "Disease"]
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]


def load(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return json.load(f)


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


# --------------------------------------------------------------------------- setup
train = load("data/train.json")
preds = load("crf_test_predictions.json")

surface = lambda toks, s, e: " ".join(toks[s:e])
train_counts = Counter()  # (lowercased string, type) -> count in training gold entities
for sent in train:
    for typ, s, e in M.get_entities(sent["tags"]):
        train_counts[(surface(sent["tokens"], s, e).lower(), typ)] += 1


def freq_bucket(n):
    if n == 0:
        return "unseen (0)"
    if n <= 4:
        return "rare (1-4)"
    if n <= 19:
        return "medium (5-19)"
    return "frequent (20+)"


FREQ_ORDER = ["unseen (0)", "rare (1-4)", "medium (5-19)", "frequent (20+)"]


def len_bucket(n):
    return str(n) if n < 4 else "4+"


LEN_ORDER = ["1", "2", "3", "4+"]


def surface_cat(text):
    toks = text.split()
    raw = text.replace(" ", "")
    if len(toks) == 1 and raw.isupper() and len(raw) >= 2 and raw.isalpha():
        return "acronym (e.g. ACE)"
    if any(c.isdigit() for c in raw):
        return "contains digit"
    if "-" in toks or "-" in raw:
        return "contains hyphen"
    if len(toks) > 1:
        return "multi-word"
    return "single plain word"


# ------------------------------------------------------------ per-entity categorisation
gold_rows, pred_rows = [], []  # one dict per entity
for k, sent in enumerate(preds):
    toks = sent["tokens"]
    G = [(t, s, e) for t, s, e in M.get_entities(sent["gold"])]
    P = [(t, s, e) for t, s, e in M.get_entities(sent["pred"])]
    for (t, s, e) in G:
        if (t, s, e) in P:
            cat, match = "correct", (t, s, e)
        else:
            same_span = [p for p in P if p[1] == s and p[2] == e and p[0] != t]
            ov_same = [p for p in P if overlap((s, e), (p[1], p[2])) and p[0] == t]
            ov_other = [p for p in P if overlap((s, e), (p[1], p[2])) and p[0] != t]
            if same_span:
                cat, match = "type error", same_span[0]
            elif ov_same:
                cat, match = "boundary error", ov_same[0]
            elif ov_other:
                cat, match = "boundary+type", ov_other[0]
            else:
                cat, match = "missed", None
        txt = surface(toks, s, e)
        gold_rows.append(dict(sent=k, type=t, s=s, e=e, text=txt, length=e - s, cat=cat, match=match,
                              train_n=train_counts[(txt.lower(), t)], sc=surface_cat(txt)))
    for (t, s, e) in P:
        if (t, s, e) in G:
            cat = "correct"
        else:
            same_span = [g for g in G if g[1] == s and g[2] == e]
            ov_same = [g for g in G if overlap((s, e), (g[1], g[2])) and g[0] == t]
            ov_other = [g for g in G if overlap((s, e), (g[1], g[2])) and g[0] != t]
            if same_span:
                cat = "type error"
            elif ov_same:
                cat = "boundary error"
            elif ov_other:
                cat = "boundary+type"
            else:
                cat = "spurious"
        txt = surface(toks, s, e)
        pred_rows.append(dict(sent=k, type=t, s=s, e=e, text=txt, length=e - s, cat=cat,
                              train_as_entity=train_counts[(txt.lower(), t)]))

out = {}
n_gold, n_pred = len(gold_rows), len(pred_rows)
out["n_gold"], out["n_pred"] = n_gold, n_pred

# ------------------------------------------------------------- 1. error type breakdown
gc = Counter(r["cat"] for r in gold_rows)
pc = Counter(r["cat"] for r in pred_rows)
out["gold_outcomes"] = {c: gc.get(c, 0) for c in ["correct", "type error", "boundary error", "boundary+type", "missed"]}
out["pred_outcomes"] = {c: pc.get(c, 0) for c in ["correct", "type error", "boundary error", "boundary+type", "spurious"]}
out["gold_outcomes_by_type"] = {t: dict(Counter(r["cat"] for r in gold_rows if r["type"] == t)) for t in TYPES}
out["pred_outcomes_by_type"] = {t: dict(Counter(r["cat"] for r in pred_rows if r["type"] == t)) for t in TYPES}

# lenient (overlap-based) recall/precision: how much of the loss is only about boundaries?
lenient_tp_g = sum(r["cat"] in ("correct", "boundary error", "type error", "boundary+type") for r in gold_rows)
out["lenient"] = {
    "recall_any_overlap": lenient_tp_g / n_gold,
    "recall_overlap_same_type": sum(r["cat"] in ("correct", "boundary error") for r in gold_rows) / n_gold,
    "precision_overlap_same_type": sum(r["cat"] in ("correct", "boundary error") for r in pred_rows) / n_pred,
}

# ---------------------------------------------------------- 2. recall by length / freq
def recall_table(rows, key, order):
    tab = {}
    for b in order:
        sel = [r for r in rows if key(r) == b]
        tab[b] = dict(n=len(sel), correct=sum(r["cat"] == "correct" for r in sel),
                      recall=(sum(r["cat"] == "correct" for r in sel) / len(sel)) if sel else None,
                      missed=sum(r["cat"] == "missed" for r in sel),
                      boundary=sum(r["cat"] in ("boundary error", "boundary+type") for r in sel),
                      type_err=sum(r["cat"] == "type error" for r in sel))
    return tab


out["recall_by_length"] = recall_table(gold_rows, lambda r: len_bucket(r["length"]), LEN_ORDER)
out["recall_by_freq"] = recall_table(gold_rows, lambda r: freq_bucket(r["train_n"]), FREQ_ORDER)
out["recall_by_length_by_type"] = {t: recall_table([r for r in gold_rows if r["type"] == t],
                                                   lambda r: len_bucket(r["length"]), LEN_ORDER) for t in TYPES}
out["recall_by_freq_by_type"] = {t: recall_table([r for r in gold_rows if r["type"] == t],
                                                 lambda r: freq_bucket(r["train_n"]), FREQ_ORDER) for t in TYPES}
out["recall_by_surface"] = recall_table(gold_rows, lambda r: r["sc"],
                                        ["acronym (e.g. ACE)", "contains digit", "contains hyphen", "multi-word",
                                         "single plain word"])
# disentangle length from frequency: seen vs unseen x length
out["recall_length_seen"] = recall_table([r for r in gold_rows if r["train_n"] > 0], lambda r: len_bucket(r["length"]), LEN_ORDER)
out["recall_length_unseen"] = recall_table([r for r in gold_rows if r["train_n"] == 0], lambda r: len_bucket(r["length"]), LEN_ORDER)
out["share_unseen"] = sum(r["train_n"] == 0 for r in gold_rows) / n_gold
out["share_of_misses_unseen"] = (sum(r["train_n"] == 0 for r in gold_rows if r["cat"] != "correct") /
                                 max(sum(r["cat"] != "correct" for r in gold_rows), 1))
out["length_distribution"] = {b: out["recall_by_length"][b]["n"] for b in LEN_ORDER}

# precision by predicted length
plen = {}
for b in LEN_ORDER:
    sel = [r for r in pred_rows if len_bucket(r["length"]) == b]
    plen[b] = dict(n=len(sel), precision=(sum(r["cat"] == "correct" for r in sel) / len(sel)) if sel else None)
out["precision_by_length"] = plen

# --------------------------------------------------- 3. confusion matrices (token and entity)
idx = {l: i for i, l in enumerate(LABELS)}
tok_cm = np.zeros((5, 5), dtype=int)
for sent in preds:
    for g, p in zip(sent["gold"], sent["pred"]):
        tok_cm[idx[g], idx[p]] += 1
out["token_confusion_labels"] = LABELS
out["token_confusion"] = tok_cm.tolist()

ent_labels = ["Chemical", "Disease", "(none)"]
ent_cm = np.zeros((3, 3), dtype=int)  # rows gold, cols pred (any overlapping prediction, preferring same span)
for r in gold_rows:
    gi = ent_labels.index(r["type"])
    if r["cat"] == "correct":
        pi = gi
    elif r["cat"] == "missed":
        pi = 2
    elif r["match"] is not None:
        pi = ent_labels.index(r["match"][0])
    else:
        pi = 2
    ent_cm[gi, pi] += 1
for r in pred_rows:
    if r["cat"] == "spurious":
        ent_cm[2, ent_labels.index(r["type"])] += 1
out["entity_confusion_labels"] = ent_labels
out["entity_confusion"] = ent_cm.tolist()

# ------------------------------------------------ 4. most common misses / spurious / boundary
def top(rows, cat, n=15, key=lambda r: (r["text"].lower(), r["type"])):
    c = Counter(key(r) for r in rows if r["cat"] == cat)
    return [dict(text=k[0], type=k[1], count=v) for k, v in c.most_common(n)]


out["top_missed"] = top(gold_rows, "missed")
out["top_spurious"] = top(pred_rows, "spurious")
bnd = Counter()
for r in gold_rows:
    if r["cat"] == "boundary error" and r["match"]:
        pt, ps, pe = r["match"]
        bnd[(r["text"].lower(), surface(preds[r["sent"]]["tokens"], ps, pe).lower(), r["type"])] += 1
out["top_boundary"] = [dict(gold=k[0], pred=k[1], type=k[2], count=v) for k, v in bnd.most_common(15)]

# boundary direction: predicted shorter / longer / shifted
dirs = Counter()
for r in gold_rows:
    if r["cat"] == "boundary error" and r["match"]:
        _, ps, pe = r["match"]
        dirs["predicted too short (sub-span)" if ps >= r["s"] and pe <= r["e"] else
             "predicted too long (super-span)" if ps <= r["s"] and pe >= r["e"] else "shifted / partial"] += 1
out["boundary_direction"] = dict(dirs)

# annotation consistency: spurious strings that ARE annotated as entities elsewhere in training
spur = [r for r in pred_rows if r["cat"] == "spurious"]
out["spurious_total"] = len(spur)
out["spurious_string_annotated_in_train"] = sum(r["train_as_entity"] > 0 for r in spur)
out["spurious_by_type"] = dict(Counter(r["type"] for r in spur))
# misses whose exact string appears in training as entity of the same type (context-driven misses)
missed = [r for r in gold_rows if r["cat"] == "missed"]
out["missed_total"] = len(missed)
out["missed_string_seen_in_train"] = sum(r["train_n"] > 0 for r in missed)
out["missed_by_type"] = dict(Counter(r["type"] for r in missed))

# annotation-consistency probe: how often is a given string annotated as a standalone entity (exact span)?
def annotation_rate(data, phrase):
    ph, n, total, annotated = phrase.split(), len(phrase.split()), 0, 0
    for sent in data:
        low = [w.lower() for w in sent["tokens"]]
        spans = {(s, e) for _, s, e in M.get_entities(sent["tags"])}
        for i in range(len(low) - n + 1):
            if low[i:i + n] == ph:
                total += 1
                annotated += (i, i + n) in spans
    return total, annotated


test_gold = [{"tokens": s["tokens"], "tags": s["gold"]} for s in preds]
PROBE = ["death", "deaths", "pain", "hypertension", "mi", "stz", "apap"]
out["annotation_probe"] = {}
for ph in PROBE:
    (tt, ta), (et, ea) = annotation_rate(train, ph), annotation_rate(test_gold, ph)
    out["annotation_probe"][ph] = dict(train_occurrences=tt, train_annotated=ta, test_occurrences=et, test_annotated=ea)

# ------------------------------------------------------------------ 5. qualitative examples
rng = random.Random(0)


def render(sent_idx, spans_key):
    toks = preds[sent_idx]["tokens"]
    tags = preds[sent_idx][spans_key]
    out_t, i = [], 0
    ents = {s: (t, e) for t, s, e in M.get_entities(tags)}
    while i < len(toks):
        if i in ents:
            t, e = ents[i]
            out_t.append("[" + " ".join(toks[i:e]) + "]_" + t[:3].upper())
            i = e
        else:
            out_t.append(toks[i])
            i += 1
    return " ".join(out_t)


lines = ["# Qualitative error examples (CRF, BC5CDR test)\n",
         "Notation: `[entity text]_CHE` / `_DIS`. Seed 0 random sample; every example is real, none cherry-picked for effect.\n"]
sections = [("Missed entities (false negatives, no overlapping prediction)", gold_rows, "missed"),
            ("Boundary errors (right type, wrong span)", gold_rows, "boundary error"),
            ("Type errors (right span, wrong type)", gold_rows, "type error"),
            ("Spurious predictions (false positives)", pred_rows, "spurious")]
examples_store = {}
for title, rows, cat in sections:
    pool = [r for r in rows if r["cat"] == cat]
    pick = rng.sample(pool, min(10, len(pool)))
    lines.append(f"\n## {title} (n={len(pool)})\n")
    examples_store[cat] = []
    for r in pick:
        lines.append(f"- **{r['text']}** ({r['type']}, train count={r.get('train_n', r.get('train_as_entity'))})")
        lines.append(f"  - gold: {render(r['sent'], 'gold')}")
        lines.append(f"  - pred: {render(r['sent'], 'pred')}")
        examples_store[cat].append(dict(text=r["text"], type=r["type"], gold=render(r["sent"], "gold"),
                                        pred=render(r["sent"], "pred")))
out["examples"] = examples_store
with open(os.path.join(RES, "error_examples.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

with open(os.path.join(RES, "error_analysis.json"), "w") as f:
    json.dump(out, f, indent=2)

# ------------------------------------------------------------------------ 6. figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"Chemical": "#0072B2", "Disease": "#D55E00", "All": "#555555"}


def grouped_recall(ax, tabs, order, xlabel, title, legend_loc="upper right"):
    w = 0.27
    names = list(tabs)
    for j, name in enumerate(names):
        vals = [tabs[name][b]["recall"] if tabs[name][b]["recall"] is not None else 0 for b in order]
        ns = [tabs[name][b]["n"] for b in order]
        xs = np.arange(len(order)) + (j - (len(names) - 1) / 2) * w
        bars = ax.bar(xs, vals, w, color=C[name], label=name)
        for x, v, n in zip(xs, vals, ns):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{b.replace(' (', chr(10) + '(')}\nn={tabs['All'][b]['n']}" for b in order], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall (exact match)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
grouped_recall(axs[0], {"All": out["recall_by_length"], **{t: out["recall_by_length_by_type"][t] for t in TYPES}},
               LEN_ORDER, "Entity length (tokens)", "Recall by entity length")
grouped_recall(axs[1], {"All": out["recall_by_freq"], **{t: out["recall_by_freq_by_type"][t] for t in TYPES}},
               FREQ_ORDER, "Occurrences of the exact string in training", "Recall by training frequency",
               legend_loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "recall_by_length_frequency.png"))
plt.close()

# error outcomes (stacked share of gold entities)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
order = ["correct", "boundary error", "boundary+type", "type error", "missed"]
colors = ["#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"]
left = 0
for c, col in zip(order, colors):
    v = out["gold_outcomes"][c] / n_gold
    ax.barh([0], [v], left=left, color=col, label=f"{c} ({v:.1%})")
    left += v
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Share of gold test entities (n=%d)" % n_gold)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.38))
ax.set_title("What happens to each gold entity", fontsize=10)
plt.savefig(os.path.join(FIG, "gold_outcomes.png"))
plt.close()

# token-level confusion (row-normalised)
fig, ax = plt.subplots(figsize=(4.6, 4))
norm = tok_cm / tok_cm.sum(axis=1, keepdims=True)
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(LABELS, rotation=40, ha="right"); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{norm[i, j]:.2f}\n({tok_cm[i, j]})", ha="center", va="center", fontsize=6.5,
                color="white" if norm[i, j] > 0.55 else "black")
ax.set_title("Token-level confusion (row-normalised)", fontsize=10)
ax.spines[:].set_visible(False)
plt.savefig(os.path.join(FIG, "token_confusion.png"))
plt.close()

# ---------------------------------------------------------------- 7. LaTeX tables
def pct(x):
    return "--" if x is None else f"{x:.3f}"


def tex_table(caption, label, header, rows, colspec):
    s = ["\\begin{table}[h]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


tex = []
go = out["gold_outcomes"]
po = out["pred_outcomes"]
tex.append(tex_table(
    "Outcome of every gold entity and every predicted entity on the test set (exact-match taxonomy).",
    "tab:outcomes", ["Outcome", "Gold entities", "\\% of gold", "Predicted entities", "\\% of predicted"],
    [["Correct (exact span and type)", go["correct"], f"{100*go['correct']/n_gold:.1f}", po["correct"], f"{100*po['correct']/n_pred:.1f}"],
     ["Boundary error (same type)", go["boundary error"], f"{100*go['boundary error']/n_gold:.1f}", po["boundary error"], f"{100*po['boundary error']/n_pred:.1f}"],
     ["Boundary + type error", go["boundary+type"], f"{100*go['boundary+type']/n_gold:.1f}", po["boundary+type"], f"{100*po['boundary+type']/n_pred:.1f}"],
     ["Type error (right span)", go["type error"], f"{100*go['type error']/n_gold:.1f}", po["type error"], f"{100*po['type error']/n_pred:.1f}"],
     ["Missed / spurious (no overlap)", go["missed"], f"{100*go['missed']/n_gold:.1f}", po["spurious"], f"{100*po['spurious']/n_pred:.1f}"],
     ["\\midrule Total", n_gold, "100.0", n_pred, "100.0"]], "lrrrr"))


def rt(tab, order, name):
    return [[b, tab[b]["n"], tab[b]["correct"], pct(tab[b]["recall"]), tab[b]["missed"], tab[b]["boundary"], tab[b]["type_err"]] for b in order]


tex.append(tex_table("Recall by entity length (tokens) on the test set.", "tab:length",
                     ["Length", "Gold", "Correct", "Recall", "Missed", "Boundary", "Type"],
                     rt(out["recall_by_length"], LEN_ORDER, "len"), "lrrrrrr"))
tex.append(tex_table("Recall by how often the exact entity string occurs as the same type in the training set.",
                     "tab:freq", ["Train count", "Gold", "Correct", "Recall", "Missed", "Boundary", "Type"],
                     rt(out["recall_by_freq"], FREQ_ORDER, "freq"), "lrrrrrr"))
tex.append(tex_table("Recall by surface form of the gold entity.", "tab:surface",
                     ["Surface form", "Gold", "Correct", "Recall", "Missed", "Boundary", "Type"],
                     rt(out["recall_by_surface"], list(out["recall_by_surface"]), "sf"), "lrrrrrr"))
tex.append(tex_table("Does length matter once we control for frequency? Recall for seen vs.\\ unseen strings.",
                     "tab:lenseen", ["Length", "Seen: n", "Seen: recall", "Unseen: n", "Unseen: recall"],
                     [[b, out["recall_length_seen"][b]["n"], pct(out["recall_length_seen"][b]["recall"]),
                       out["recall_length_unseen"][b]["n"], pct(out["recall_length_unseen"][b]["recall"])] for b in LEN_ORDER],
                     "lrrrr"))
esc = lambda s: s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
tex.append(tex_table("Most frequent missed entities (test set).", "tab:topmissed", ["Entity", "Type", "Count"],
                     [[esc(r["text"]), r["type"], r["count"]] for r in out["top_missed"][:10]], "llr"))
tex.append(tex_table("Most frequent spurious predictions (test set).", "tab:topspur", ["Predicted string", "Type", "Count"],
                     [[esc(r["text"]), r["type"], r["count"]] for r in out["top_spurious"][:10]], "llr"))
tex.append(tex_table("Most frequent boundary errors (test set).", "tab:topbound", ["Gold", "Predicted", "Type", "Count"],
                     [[esc(r["gold"]), esc(r["pred"]), r["type"], r["count"]] for r in out["top_boundary"][:10]], "lllr"))
tex.append(tex_table(
    "Annotation consistency probe: how often a string is annotated as a standalone entity (exact span) in training vs.\\ test.",
    "tab:annprobe", ["String", "Train occurrences", "Train annotated", "Test occurrences", "Test annotated"],
    [[ph, v["train_occurrences"], v["train_annotated"], v["test_occurrences"], v["test_annotated"]]
     for ph, v in out["annotation_probe"].items()], "lrrrr"))
with open(os.path.join(REP, "tables_generated.tex"), "w") as f:
    f.write("% AUTO-GENERATED by error_analysis.py -- do not edit by hand\n" + "\n\n".join(tex) + "\n")
os.makedirs(os.path.join(REP, "tables"), exist_ok=True)  # one file per table, so the report can place each one
for t in tex:
    name = re.search(r"\\label\{tab:(\w+)\}", t).group(1)
    with open(os.path.join(REP, "tables", f"{name}.tex"), "w") as f:
        f.write(t.replace("\\begin{table}[h]", "\\begin{table}[t]") + "\n")

# ---------------------------------------------------------------- 8. console summary
print(f"Gold entities: {n_gold}   Predicted entities: {n_pred}")
print("\nGold outcomes:", out["gold_outcomes"])
print("Pred outcomes:", out["pred_outcomes"])
print("Lenient:", {k: round(v, 4) for k, v in out["lenient"].items()})
for name, tab in [("length", out["recall_by_length"]), ("train frequency", out["recall_by_freq"]),
                  ("surface form", out["recall_by_surface"])]:
    print(f"\nRecall by {name}:")
    for b, v in tab.items():
        print(f"  {b:20s} n={v['n']:5d} recall={pct(v['recall'])} missed={v['missed']} boundary={v['boundary']} type={v['type_err']}")
print("\nLength x seen/unseen:")
for b in LEN_ORDER:
    s, u = out["recall_length_seen"][b], out["recall_length_unseen"][b]
    print(f"  len {b:2s} seen n={s['n']:4d} R={pct(s['recall'])} | unseen n={u['n']:4d} R={pct(u['recall'])}")
print(f"\nUnseen share of gold: {out['share_unseen']:.3f}; share of all errors that are unseen strings: {out['share_of_misses_unseen']:.3f}")
print("Boundary direction:", out["boundary_direction"])
print(f"Spurious: {out['spurious_total']} (annotated in train somewhere: {out['spurious_string_annotated_in_train']}) by type {out['spurious_by_type']}")
print(f"Missed: {out['missed_total']} (string seen in train: {out['missed_string_seen_in_train']}) by type {out['missed_by_type']}")
print("Entity confusion", ent_labels, "\n", ent_cm)
print("\nTop missed:", [(r['text'], r['type'], r['count']) for r in out['top_missed'][:8]])
print("Top spurious:", [(r['text'], r['type'], r['count']) for r in out['top_spurious'][:8]])
print("Top boundary:", [(r['gold'], r['pred'], r['count']) for r in out['top_boundary'][:8]])
