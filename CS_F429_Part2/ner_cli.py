import argparse
import json
import os
import re
import sys
import time

import crf_baseline as cb

IMPROVED_PATH = os.path.join(cb.HERE, "models", "crf_rare_idrop60_replace70.crfsuite")
BIOBERT_DIR = os.path.join(cb.HERE, "models", "biobert")

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
    "Mesna did not prevent ifosfamide-induced hemorrhagic cystitis in this patient.",       # rare / unseen-style names
    "Ouabain-induced arrhythmia was reversed by lidocaine in the isolated heart.",
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


class CRFTagger:
    def __init__(self, crf, name):
        self.crf, self.name = crf, name

    def tag(self, sentence):
        return cb.tag_text(self.crf, sentence)


class BioBertTagger:
    """Word-level BioBERT tagger: same word tokenisation as the CRF, label on the first sub-word of each word."""

    def __init__(self, path, name="BioBERT"):
        if not os.path.exists(os.path.join(path, "model.safetensors")) and not os.path.exists(
                os.path.join(path, "pytorch_model.bin")):
            raise FileNotFoundError(
                f"No BioBERT weights in {path} (model.safetensors missing). The weights are not stored in git: "
                "copy the 'model' folder written by biobert_ner.py (Colab) into models/biobert/.")
        try:
            import torch
            from transformers import BertForTokenClassification, BertTokenizerFast
        except ImportError as exc:
            raise FileNotFoundError(f"BioBERT needs the 'torch' and 'transformers' packages ({exc}). "
                                    "Install them with: pip install torch transformers") from exc
        self.torch, self.name = torch, name
        self.tok = BertTokenizerFast.from_pretrained(path, do_lower_case=False)
        self.model = BertForTokenClassification.from_pretrained(path).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def tag(self, sentence):
        toks = cb.tokenize_with_offsets(sentence)
        if not toks:
            return []
        words = [t[0] for t in toks]
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0].argmax(-1).cpu()
        tags, prev = ["O"] * len(words), None
        for pos, w in enumerate(enc.word_ids()):
            if w is not None and w != prev:
                tags[w] = self.id2label[int(logits[pos])]
            prev = w
        ents = []
        for typ, s, e in cb.ner_metrics.get_entities(tags):
            cs, ce = toks[s][1], toks[e - 1][2]
            if any(ch.isalnum() for ch in sentence[cs:ce]):
                ents.append({"text": sentence[cs:ce], "type": typ, "start": cs, "end": ce})
        return ents


def tag_document(tagger, text):
    ents = []
    for sent, off in split_sentences(text):
        for e in tagger.tag(sent):
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


def show(text, ents, color, label=None, first=True):
    if first:
        print(f"\nText   : {text}")
    print(f"Marked : {mark(text, ents, color)}" if label is None else f"{label:<13s}: {mark(text, ents, color)}")
    if label is not None:
        return                      # compare mode: one line per model
    if not ents:
        print("Entities: (none found)")
        return
    print("Entities:")
    for e in ents:
        print(f"  {e['type'].upper():9s} {e['text']:<32s} chars {e['start']}-{e['end']}")
    n_c = sum(e["type"] == "Chemical" for e in ents)
    print(f"  -> {n_c} chemical(s), {len(ents) - n_c} disease(s)")


def get_crf(retrain):
    if retrain or not os.path.exists(cb.MODEL_PATH):
        print("Training CRF on BC5CDR train split (c1=0.1, c2=0.01) ...", flush=True)
        t0 = time.time()
        train = cb.load("train")
        X, y = cb.featurize(train)
        crf = cb.train_crf(X, y, model_filename=cb.MODEL_PATH)
        print(f"Done in {time.time() - t0:.1f}s. Model saved to {os.path.relpath(cb.MODEL_PATH, cb.HERE)}\n")
        return crf
    return cb.load_model()


def load_taggers(which, retrain, biobert_dir):
    """which: 'crf' | 'crf-improved' | 'biobert' | 'all' (skips models that are not available)."""
    out = []
    if which in ("crf", "all"):
        out.append(CRFTagger(get_crf(retrain), "CRF"))
    if which in ("crf-improved", "all"):
        if os.path.exists(IMPROVED_PATH):
            out.append(CRFTagger(cb.load_model(IMPROVED_PATH), "CRF improved"))
        elif which != "all":
            sys.exit(f"Improved CRF not found: {IMPROVED_PATH}\nCreate it with: python phase_3/rare_intervention.py "
                     "run --name idrop60_replace70 --idrop 0.6 --replace 0.7")
    if which in ("biobert", "all"):
        try:
            out.append(BioBertTagger(biobert_dir))
        except FileNotFoundError as exc:
            if which != "all":
                sys.exit(str(exc))
            print(f"[BioBERT skipped] {exc}\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="text to tag")
    ap.add_argument("--file", help="path of a UTF-8 text file to tag")
    ap.add_argument("--demo", action="store_true", help="tag built-in demo sentences")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--retrain", action="store_true", help="retrain the model before tagging")
    ap.add_argument("--color", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--model", choices=["crf", "crf-improved", "biobert"], default="crf", help="which model to use")
    ap.add_argument("--biobert-dir", default=BIOBERT_DIR, help="folder with the saved BioBERT checkpoint")
    ap.add_argument("--compare", action="store_true", help="tag with every available model and show them together")
    args = ap.parse_args()

    color = args.color == "on" or (args.color == "auto" and sys.stdout.isatty() and os.name != "nt")
    taggers = load_taggers('all' if args.compare else args.model, args.retrain, args.biobert_dir)

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
            if args.compare:
                per = {tg.name: tag_document(tg, t) for tg in taggers}
                results.append({"text": t, "entities": per})
                if not args.json:
                    for i, (nm, ents) in enumerate(per.items()):
                        show(t, ents, color, label=nm, first=(i == 0))
            else:
                ents = tag_document(taggers[0], t)
                results.append({"text": t, "model": taggers[0].name, "entities": ents})
                if not args.json:
                    show(t, ents, color)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # interactive
    names = ", ".join(tg.name for tg in taggers)
    print(f"Biomedical NER ({names}; BC5CDR: Chemical / Disease). Type a sentence; empty line to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if args.compare:
            for i, tg in enumerate(taggers):
                show(line, tag_document(tg, line), color, label=tg.name, first=(i == 0))
        else:
            show(line, tag_document(taggers[0], line), color)


if __name__ == "__main__":
    main()
