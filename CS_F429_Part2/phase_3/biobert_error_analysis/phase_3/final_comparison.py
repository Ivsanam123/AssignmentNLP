"""Final comparison of the selected systems on TEST (CS F429 Part 3).  Descriptive only: every model here was
fixed and selected on DEV before this script was run; nothing below feeds back into any choice.

Systems:  original CRF | improved CRF (rare-entity intervention, config chosen on dev) | BioBERT baseline
Breakdowns use TRAIN counts of the lower-cased (string, type): unseen 0, rare 1-4, medium 5-19, frequent 20+.

Usage (from CS_F429_Part2/):   python phase_3/final_comparison.py
Writes phase_3/results/final/final_comparison.txt, final_comparison.csv and report/tables/final_comparison.tex
"""
import csv, json, os, random, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
import ner_metrics as M

OUT = os.path.join(HERE, "results", "final")
os.makedirs(OUT, exist_ok=True)
BINS = ("unseen", "rare 1-4", "medium 5-19", "frequent 20+")
LENS = ("1", "2", "3", "4+")
OUTC = ("correct", "boundary", "boundary+type", "type", "missed")

train = json.load(open(os.path.join(ROOT, "data", "train.json"), encoding="utf-8"))
test = json.load(open(os.path.join(ROOT, "data", "test.json"), encoding="utf-8"))
cnt = Counter()
for s in train:
    for t, a, b in M.get_entities(s["tags"]):
        cnt[(" ".join(s["tokens"][a:b]).lower(), t)] += 1


def fbin(n):
    return BINS[0] if n == 0 else BINS[1] if n < 5 else BINS[2] if n < 20 else BINS[3]


def lbin(n):
    return LENS[min(n, 4) - 1]


def load(path):
    d = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    return [x["pred"] if isinstance(x, dict) else x for x in d]


SYSTEMS = {
    "Original CRF": load("phase_3/results/rare/TEST_original_CRF_pred.json"),
    "Improved CRF (rare-entity)": load("phase_3/results/rare/TEST_idrop60_replace70_pred.json"),
    "BioBERT": load("colab_outputs/biobert_run/biobert_test_predictions.json"),
}
for k, p in SYSTEMS.items():
    assert len(p) == len(test) and all(len(a["tokens"]) == len(b) for a, b in zip(test, p)), k


def outcome(g, preds):
    """g=(type,s,e). preds = set of predicted entities of that sentence."""
    if g in preds:
        return "correct"
    ov = [p for p in preds if p[1] < g[2] and g[1] < p[2]]
    if not ov:
        return "missed"
    if any((p[1], p[2]) == (g[1], g[2]) for p in ov):
        return "type"
    return "boundary" if any(p[0] == g[0] for p in ov) else "boundary+type"


def per_sentence(preds):
    rows = []
    for s, p in zip(test, preds):
        gold = M.get_entities(s["tags"])
        ps = set(M.get_entities(p))
        r = {"tp": sum(g in ps for g in gold), "pp": len(ps), "gp": len(gold),
             "tpT": Counter(), "ppT": Counter(), "gpT": Counter(),
             "fb": defaultdict(lambda: [0, 0]), "lb": defaultdict(lambda: [0, 0]), "oc": Counter()}
        for g in gold:
            hit = g in ps
            r["gpT"][g[0]] += 1
            r["tpT"][g[0]] += hit
            fb = fbin(cnt[(" ".join(s["tokens"][g[1]:g[2]]).lower(), g[0])])
            lb = lbin(g[2] - g[1])
            r["fb"][fb][0] += hit; r["fb"][fb][1] += 1
            r["lb"][lb][0] += hit; r["lb"][lb][1] += 1
            r["oc"][outcome(g, ps)] += 1
        for q in ps:
            r["ppT"][q[0]] += 1
        rows.append(r)
    return rows


def prf(tp, pp, gp):
    p = tp / pp if pp else 0.0
    r = tp / gp if gp else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarise(rows, idx=None):
    idx = range(len(rows)) if idx is None else idx
    S = lambda k: sum(rows[i][k] for i in idx)
    res = {"overall": prf(S("tp"), S("pp"), S("gp"))}
    for t in ("Chemical", "Disease"):
        res[t] = prf(sum(rows[i]["tpT"][t] for i in idx), sum(rows[i]["ppT"][t] for i in idx),
                     sum(rows[i]["gpT"][t] for i in idx))
    for key, names in (("fb", BINS), ("lb", LENS)):
        for n in names:
            c = sum(rows[i][key][n][0] for i in idx); m = sum(rows[i][key][n][1] for i in idx)
            res[(key, n)] = (c / m if m else float("nan"), m)
    tot = sum(sum(rows[i]["oc"].values()) for i in idx)
    for o in OUTC:
        res[("oc", o)] = sum(rows[i]["oc"][o] for i in idx) / tot
    return res


R = {k: per_sentence(v) for k, v in SYSTEMS.items()}
S = {k: summarise(v) for k, v in R.items()}

lines = []
P = lines.append
P("FINAL COMPARISON ON TEST (entity-level exact match; models fixed and selected on DEV beforehand)")
P("")
P(f"{'system':28s} {'P':>7s} {'R':>7s} {'F1':>7s} | {'Chem F1':>8s} {'Dis F1':>8s}")
for k in SYSTEMS:
    o = S[k]["overall"]
    P(f"{k:28s} {o[0]:7.4f} {o[1]:7.4f} {o[2]:7.4f} | {S[k]['Chemical'][2]:8.4f} {S[k]['Disease'][2]:8.4f}")
P("")
P("Recall by training frequency of the gold string (n = gold mentions)")
P(f"{'bin':14s} {'n':>6s} " + " ".join(f"{k[:26]:>26s}" for k in SYSTEMS))
for b in BINS:
    n = S["BioBERT"][("fb", b)][1]
    P(f"{b:14s} {n:6d} " + " ".join(f"{S[k][('fb', b)][0]:26.3f}" for k in SYSTEMS))
P("")
P("Recall by entity length in tokens")
P(f"{'length':14s} {'n':>6s} " + " ".join(f"{k[:26]:>26s}" for k in SYSTEMS))
for b in LENS:
    n = S["BioBERT"][("lb", b)][1]
    P(f"{b:14s} {n:6d} " + " ".join(f"{S[k][('lb', b)][0]:26.3f}" for k in SYSTEMS))
P("")
P("Outcome of every gold mention (share of all gold mentions)")
P(f"{'outcome':14s} " + " ".join(f"{k[:26]:>26s}" for k in SYSTEMS))
for o in OUTC:
    P(f"{o:14s} " + " ".join(f"{S[k][('oc', o)]:26.3f}" for k in SYSTEMS))

# paired bootstrap over test sentences
random.seed(0)
pairs = (("Improved CRF (rare-entity)", "Original CRF"), ("BioBERT", "Original CRF"), ("BioBERT", "Improved CRF (rare-entity)"))
B = {p: {"f1": [], "unseen": [], "rare": []} for p in pairs}
n = len(test)
NB = 300
for _ in range(NB):
    idx = [random.randrange(n) for _ in range(n)]
    cache = {k: summarise(R[k], idx) for k in SYSTEMS}
    for a, b in pairs:
        B[(a, b)]["f1"].append(cache[a]["overall"][2] - cache[b]["overall"][2])
        B[(a, b)]["unseen"].append(cache[a][("fb", BINS[0])][0] - cache[b][("fb", BINS[0])][0])
        B[(a, b)]["rare"].append(cache[a][("fb", BINS[1])][0] - cache[b][("fb", BINS[1])][0])
P("")
P("Paired bootstrap over test sentences (300 resamples): difference, 95% CI")
ci = lambda x: (sorted(x)[int(0.025 * len(x))], sorted(x)[int(0.975 * len(x)) - 1])
for (a, b), d in B.items():
    base = S[a]["overall"][2] - S[b]["overall"][2]
    cu = ci(d["unseen"]); cf = ci(d["f1"])
    P(f"  {a} minus {b}:  F1 {base:+.4f} [{cf[0]:+.4f}, {cf[1]:+.4f}]   unseen recall "
      f"{S[a][('fb', BINS[0])][0] - S[b][('fb', BINS[0])][0]:+.3f} [{cu[0]:+.3f}, {cu[1]:+.3f}]")
open(os.path.join(OUT, "final_comparison.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

with open(os.path.join(OUT, "final_comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "P", "R", "F1", "Chem_F1", "Dis_F1"] + [f"recall_{b}" for b in BINS] + [f"recall_len{b}" for b in LENS])
    for k in SYSTEMS:
        o = S[k]["overall"]
        w.writerow([k, *(round(x, 4) for x in o), round(S[k]["Chemical"][2], 4), round(S[k]["Disease"][2], 4)]
                   + [round(S[k][("fb", b)][0], 4) for b in BINS] + [round(S[k][("lb", b)][0], 4) for b in LENS])

tex = [r"\begin{tabular}{lrrrrrr}", r"\hline",
       r"System & P & R & F1 & Unseen R & Rare R & Frequent R \\", r"\hline"]
for k in SYSTEMS:
    o = S[k]["overall"]
    tex.append(f"{k} & {o[0]:.3f} & {o[1]:.3f} & {o[2]:.3f} & {S[k][('fb', BINS[0])][0]:.3f} & "
               f"{S[k][('fb', BINS[1])][0]:.3f} & {S[k][('fb', BINS[3])][0]:.3f} \\\\")
tex += [r"\hline", r"\end{tabular}"]
os.makedirs(os.path.join(ROOT, "report", "tables"), exist_ok=True)
open(os.path.join(ROOT, "report", "tables", "final_comparison.tex"), "w").write("\n".join(tex) + "\n")
