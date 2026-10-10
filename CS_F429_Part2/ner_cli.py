"""Command-line demo: extract CHEMICAL and DISEASE mentions from biomedical text with the CRF.

Usage:
  python ner_cli.py --demo                              # run on built-in biomedical sentences (no typing)
  python ner_cli.py --text "Cisplatin caused nephrotoxicity."
  python ner_cli.py --file abstract.txt                 # tag a whole text file (sentence-split automatically)
  echo "Aspirin can cause gastric ulcers." | python ner_cli.py
  python ner_cli.py                                     # interactive prompt (empty line or Ctrl-D to quit)
  python ner_cli.py --json --text "..."                 # machine-readable output
  python ner_cli.py --retrain                           # retrain first (~25 s), then tag

The trained model lives in models/crf_bc5cdr.crfsuite (created by `python crf_baseline.py`).
If it is missing, this script trains it automatically.
"""
import argparse
import json
import os
import re
import sys
import time

import crf_baseline as cb

COLORS = {"Chemical": "\033[92m", "Disease": "\033[91m"}  # green, red
RESET = "\033[0m"

DEMO_SENTENCES = [
    "Naloxone reverses the antihypertensive effect of clonidine.",
    "Famotidine-associated delirium was observed in elderly patients with renal failure.",
    "Patients treated with cisplatin developed nephrotoxicity and hearing loss.",
    "Long-term use of ibuprofen has been linked to gastric ulcers and kidney injury.",
    "Methotrexate-induced hepatotoxicity resolved after the dose was reduced.",
    "Lithium therapy may cause tremor, hypothyroidism and, rarely, seizures.",
    "Haloperidol was given to control agitation in a patient with Parkinson's disease.",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text):
    """Split text into (sentence, char_offset) pairs."""
    out, pos = [], 0
    for m in SENT_SPLIT.finditer(text):
        out.append((text[pos:m.start()], pos))
        pos = m.end()
    out.append((text[pos:], pos))
    return [(s, o) for s, o in out if s.strip()]


def tag_document(crf, text):
    ents = []
    for sent, off in split_sentences(text):
        for e in cb.tag_text(crf, sent):
            e = dict(e)
            e["start"] += off
            e["end"] += off
            ents.append(e)
    return ents


def mark(text, ents, color):
    """Return text with entities highlighted: ANSI colours or inline [TYPE] labels."""
    out, pos = [], 0
    for e in sorted(ents, key=lambda x: x["start"]):
        out.append(text[pos:e["start"]])
        piece = text[e["start"]:e["end"]]
        if color:
            out.append(f"{COLORS.get(e['type'], '')}{piece}{RESET}\033[1m[{e['type'].upper()}]{RESET}")
        else:
            out.append(f"{piece}[{e['type'].upper()}]")
        pos = e["end"]
    out.append(text[pos:])
    return "".join(out)


def show(text, ents, color):
    print(f"\nText   : {text}")
    print(f"Marked : {mark(text, ents, color)}")
    if not ents:
        print("Entities: (none found)")
        return
    print("Entities:")
    for e in ents:
        print(f"  {e['type'].upper():9s} {e['text']:<32s} chars {e['start']}-{e['end']}")
    n_c = sum(e["type"] == "Chemical" for e in ents)
    print(f"  -> {n_c} chemical(s), {len(ents) - n_c} disease(s)")


def get_model(retrain):
    if retrain or not os.path.exists(cb.MODEL_PATH):
        print("Training CRF on BC5CDR train split (c1=0.1, c2=0.01) ...", flush=True)
        t0 = time.time()
        train = cb.load("train")
        X, y = cb.featurize(train)
        crf = cb.train_crf(X, y, model_filename=cb.MODEL_PATH)
        print(f"Done in {time.time() - t0:.1f}s. Model saved to {os.path.relpath(cb.MODEL_PATH, cb.HERE)}\n")
        return crf
    return cb.load_model()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="text to tag")
    ap.add_argument("--file", help="path of a UTF-8 text file to tag")
    ap.add_argument("--demo", action="store_true", help="tag built-in demo sentences")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--retrain", action="store_true", help="retrain the model before tagging")
    ap.add_argument("--color", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    color = args.color == "on" or (args.color == "auto" and sys.stdout.isatty() and os.name != "nt")
    crf = get_model(args.retrain)

    explicit = args.text is not None or args.file is not None or args.demo
    texts = []
    if args.text is not None:
        texts.append(args.text)
    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as f:
                texts.append(f.read())
        except (OSError, UnicodeDecodeError) as exc:
            sys.exit(f"Could not read {args.file}: {exc}")
    if args.demo:
        texts.extend(DEMO_SENTENCES)

    if not explicit and not sys.stdin.isatty():  # piped input
        piped = sys.stdin.read()
        if piped.strip():
            texts.append(piped)
            explicit = True

    texts = [t.strip() for t in texts if t and t.strip()]  # strip so character offsets match the shown text
    if explicit and not texts:
        print("No text to tag (the input was empty).")
        return
    if texts:
        results = []
        for t in texts:
            ents = tag_document(crf, t)
            results.append({"text": t, "entities": ents})
            if not args.json:
                show(t, ents, color)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # interactive
    print("CRF biomedical NER (BC5CDR: Chemical / Disease). Type a sentence; empty line to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        show(line, tag_document(crf, line), color)


if __name__ == "__main__":
    main()
