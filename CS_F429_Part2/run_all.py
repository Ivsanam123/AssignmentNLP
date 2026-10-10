"""Run the whole Part 2 pipeline end to end and CHECK the results against the official numbers.

Usage:  python run_all.py            # data -> train CRF -> baselines -> error analysis -> checks (~1 min)
        python run_all.py --demo     # ... and finish with the CLI demo
Cross-platform (Windows / macOS / Linux). Stops at the first failing step.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = {"f1": 0.8076, "precision": 0.8642, "recall": 0.7579}  # tuned CRF, c1=0.1, c2=0.01, test set


def run(*args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    run("data_pipeline.py")
    run("crf_baseline.py")
    run("dictionary_baseline.py")
    run("error_analysis.py")
    run(os.path.join("report", "make_tables.py"))

    with open(os.path.join(HERE, "results", "crf_metrics.json")) as f:
        test = json.load(f)["test"]["overall"]
    ok = all(abs(test[k] - v) < 5e-5 for k, v in OFFICIAL.items())
    print("\n" + "=" * 60)
    print(f"Test P={test['precision']:.4f} R={test['recall']:.4f} F1={test['f1']:.4f}   "
          f"(official: P={OFFICIAL['precision']} R={OFFICIAL['recall']} F1={OFFICIAL['f1']})")
    print("CHECK PASSED: reproduces the official baseline." if ok else "CHECK FAILED: numbers differ from the official baseline!")
    if not ok:
        sys.exit(1)
    if "--demo" in sys.argv:
        run("ner_cli.py", "--demo")


if __name__ == "__main__":
    main()
