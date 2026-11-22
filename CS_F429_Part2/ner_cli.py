"""Command-line demo: extract CHEMICAL and DISEASE mentions from biomedical text.

Models:  crf (Part 2)  |  improved_crf (Part 3 augmentation)  |  biobert (fine-tuned)

Usage:
  python ner_cli.py --demo                                  # built-in sentences, original CRF
  python ner_cli.py --demo --compare                        # same sentences, all models side by side
  python ner_cli.py --model biobert --text "Cisplatin caused nephrotoxicity."
  python ner_cli.py --compare --text "Mesna protects against cystitis."
  python ner_cli.py --file abstract.txt                     # tag a whole text file (sentence-split automatically)
  echo "Aspirin can cause gastric ulcers." | python ner_cli.py
  python ner_cli.py                                         # interactive prompt (empty line or Ctrl-D to quit)
  python ner_cli.py --json --text "..."                     # machine-readable output
  python ner_cli.py --retrain                               # retrain the CRF first (~25 s), then tag

Model files:
  CRF           models/crf_bc5cdr.crfsuite   (created by `python crf_baseline.py`; trained automatically if missing)
  improved CRF  --improved_crf PATH          (default models/crf_improved.crfsuite)
  BioBERT       --biobert_dir DIR            (default models/biobert; the folder with config.json + model.safetensors)
With --compare, a model whose files are missing is skipped with a note instead of stopping the demo.
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
LABELS = {"crf": "CRF", "improved_crf": "Improved CRF", "biobert": "BioBERT"}
ALL_MODELS = ["crf", "improved_crf", "biobert"]
DEFAULT_IMPROVED = os.path.join(cb.HERE, "models", "crf_improved.crfsuite")
DEFAULT_BIOBERT = os.path.join(cb.HERE, "models", "biobert")

DEMO_SENTENCES = [
    "Naloxone reverses the antihypertensive effect of clonidine.",
    "Famotidine-associated delirium was observed in elderly patients with renal failure.",
    "Patients treated with cisplatin developed nephrotoxicity and hearing loss.",
    "Long-term use of ibuprofen has been linked to gastric ulcers and kidney injury.",
    "Methotrexate-induced hepatotoxicity resolved after the dose was reduced.",
    "Lithium therapy may cause tremor, hypothyroidism and, rarely, seizures.",
    "Haloperidol was given to control agitation in a patient with Parkinson's disease.",
    "Mesna was given to prevent ifosfamide-induced hemorrhagic cystitis.",
    "Ouabain increased the force of cardiac contraction.",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)   # word / punctuation tokens, like the BC5CDR word-level data


def split_sentences(text):
    """Split text into (sentence, char_offset) pairs."""
    out, pos = [], 0
    for m in SENT_SPLIT.finditer(text):
        out.append((text[pos:m.start()], pos))
        pos = m.end()
    out.append((text[pos:], pos))
    return [(s, o) for s, o in out if s.strip()]


# ------------------------------------------------------------------ BioBERT tagger
def bio_to_entities(tags, spans, sent):
    """BIO tags per word + (start, end) char span per word -> [{type, text, start, end}]."""
    ents, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        p, ty = (t.split("-", 1) + [""])[:2] if t != "O" else ("O", "")
        if cur is not None and (p in ("B", "O") or (p == "I" and ty != cur[0])):
            a, b = spans[cur[1]][0], spans[i - 1][1]
            ents.append({"type": cur[0], "text": sent[a:b], "start": a, "end": b})
            cur = None
        if p == "B" or (p == "I" and cur is None and ty):
            cur = (ty, i)
    return ents


class BioBertTagger:
    def __init__(self, model_dir):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"BioBERT folder not found: {model_dir}")
        import torch
        from transformers import BertForTokenClassification, BertTokenizerFast
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = BertTokenizerFast.from_pretrained(model_dir, do_lower_case=False)
        self.model = BertForTokenClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def __call__(self, sent):
        spans = [(m.start(), m.end()) for m in WORD_RE.finditer(sent)]
        if not spans:
            return []
        words = [sent[a:b] for a, b in spans]
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
        wids = enc.word_ids()
        with self.torch.no_grad():
            pred = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0].argmax(-1).cpu().tolist()
        tags, prev = ["O"] * len(words), None
        for pos, w in enumerate(wids):          # a word takes the label of its first sub-word
            if w is not None and w != prev:
                tags[w] = self.id2label[pred[pos]]
            prev = w
        return bio_to_entities(tags, spans, sent)


# ------------------------------------------------------------------ model loading
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


def make_tagger(name, args):
    """Return a function  sentence -> [entity dicts]  for the named model (raises if its files are missing)."""
    if name == "crf":
        crf = get_model(args.retrain)
        return lambda s: cb.tag_text(crf, s)
    if name == "improved_crf":
        if not os.path.exists(args.improved_crf):
            raise FileNotFoundError(f"improved CRF model not found: {args.improved_crf} (set --improved_crf)")
        import sklearn_crfsuite
        crf = sklearn_crfsuite.CRF(model_filename=args.improved_crf)
        return lambda s: cb.tag_text(crf, s)
    if name == "biobert":
        return BioBertTagger(args.biobert_dir)
    raise ValueError(name)


def tag_document(tag_fn, text):
    ents = []
    for sent, off in split_sentences(text):
        for e in tag_fn(sent):
            e = dict(e)
            e["start"] += off
            e["end"] += off
            ents.append(e)
    return ents


# ------------------------------------------------------------------ display
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


def show_compare(text, by_model, color):
    print(f"\nText: {text}")
    w = max(len(m) for m in by_model)
    for name, ents in by_model.items():
        found = "; ".join(f"{e['text']} [{e['type']}]" for e in sorted(ents, key=lambda x: x["start"])) or "(none)"
        print(f"  {name:<{w}} : {mark(text, ents, color)}")
        print(f"  {'':<{w}}   found: {found}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="text to tag")
    ap.add_argument("--file", help="path of a UTF-8 text file to tag")
    ap.add_argument("--demo", action="store_true", help="tag built-in demo sentences")
    ap.add_argument("--model", choices=ALL_MODELS, default="crf", help="which model tags the text (default: crf)")
    ap.add_argument("--compare", action="store_true", help="run all models and show their output side by side")
    ap.add_argument("--improved_crf", default=DEFAULT_IMPROVED, help="path of the improved CRF .crfsuite model")
    ap.add_argument("--biobert_dir", default=DEFAULT_BIOBERT, help="folder of the fine-tuned BioBERT model")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--retrain", action="store_true", help="retrain the CRF before tagging")
    ap.add_argument("--color", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    color = args.color == "on" or (args.color == "auto" and sys.stdout.isatty() and os.name != "nt")

    names = ALL_MODELS if args.compare else [args.model]
    taggers = {}
    for n in names:
        try:
            taggers[LABELS[n]] = make_tagger(n, args)
        except Exception as exc:
            if not args.compare:
                sys.exit(f"Could not load {LABELS[n]}: {exc}")
            print(f"[skipping {LABELS[n]}: {exc}]")
    if not taggers:
        sys.exit("No model could be loaded.")
    if args.compare and len(taggers) > 1:
        print("Models:", ", ".join(taggers), "\n")

    def run(t):
        return {name: tag_document(fn, t) for name, fn in taggers.items()}

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
            by_model = run(t)
            if args.json:
                if args.compare:
                    results.append({"text": t, "models": by_model})
                else:
                    results.append({"text": t, "entities": next(iter(by_model.values()))})
            elif args.compare:
                show_compare(t, by_model, color)
            else:
                show(t, next(iter(by_model.values())), color)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # interactive
    print(f"Biomedical NER (BC5CDR: Chemical / Disease) with {', '.join(taggers)}. Type a sentence; empty line to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        by_model = run(line)
        if args.compare:
            show_compare(line, by_model, color)
        else:
            show(line, next(iter(by_model.values())), color)


if __name__ == "__main__":
    main()