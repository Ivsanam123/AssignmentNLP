"""What did the rare-entity intervention change?  DEV only (original CRF vs improved CRF, same sentences).
Counts gold mentions gained / lost and false positives added / removed, by training frequency of the string.
Usage (from CS_F429_Part2/):  python phase_3/improved_crf_error_analysis.py
Writes phase_3/results/final/improved_crf_error_analysis.txt"""
import json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [ROOT]
import ner_metrics as M
R = lambda p: json.load(open(os.path.join(ROOT, p), encoding="utf-8"))
dev, tr = R("data/dev.json"), R("data/train.json")
O, I = R("phase_3/results/rare/base_dev_pred.json"), R("phase_3/results/rare/idrop60_replace70_dev_pred.json")
cnt = Counter()
for s in tr:
    for t, a, b in M.get_entities(s["tags"]): cnt[(" ".join(s["tokens"][a:b]).lower(), t)] += 1
B = ["unseen", "rare 1-4", "medium 5-19", "frequent 20+"]
bk = lambda n: B[0] if n == 0 else B[1] if n < 5 else B[2] if n < 20 else B[3]
gain, lost, gs, ls, fpn, fpg, fps = (Counter() for _ in range(7))
for s, o, i in zip(dev, O, I):
    g, po, pi = set(M.get_entities(s["tags"])), set(M.get_entities(o)), set(M.get_entities(i))
    st = lambda e: " ".join(s["tokens"][e[1]:e[2]]).lower()
    for e in g:
        b = bk(cnt[(st(e), e[0])])
        if e in pi and e not in po: gain[b] += 1; gs[(st(e), e[0])] += 1
        if e in po and e not in pi: lost[b] += 1; ls[(st(e), e[0])] += 1
    for e in pi - g - po: fpn[bk(cnt[(st(e), e[0])])] += 1; fps[(st(e), e[0])] += 1
    for e in po - g - pi: fpg[bk(cnt[(st(e), e[0])])] += 1
L = ["DEV: gold mentions gained / lost by the intervention, by training frequency of the string",
     f"{'bin':13s}{'gained':>8s}{'lost':>7s}{'net':>6s}"]
L += [f"{b:13s}{gain[b]:8d}{lost[b]:7d}{gain[b]-lost[b]:6d}" for b in B]
L += [f"total gained {sum(gain.values())}, lost {sum(lost.values())}", "",
      "False positives added / removed, by training frequency of the predicted string"]
L += [f"{b:13s} added {fpn[b]:4d}  removed {fpg[b]:4d}" for b in B]
L += ["", "Most recovered strings (count of mentions): " + str(gs.most_common(12)),
      "Most newly lost strings: " + str(ls.most_common(8)),
      "Most frequent new false positives: " + str(fps.most_common(12))]
os.makedirs(os.path.join(HERE, "results", "final"), exist_ok=True)
open(os.path.join(HERE, "results", "final", "improved_crf_error_analysis.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
