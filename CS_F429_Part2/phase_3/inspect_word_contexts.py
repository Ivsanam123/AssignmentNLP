
import json
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def get_entities(tags):
    entities = []
    start = None
    entity_type = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((start, i, entity_type))
            start = i
            entity_type = tag[2:]

        elif tag.startswith("I-"):
            if start is None or entity_type != tag[2:]:
                if start is not None:
                    entities.append((start, i, entity_type))
                start = i
                entity_type = tag[2:]

        elif start is not None:
            entities.append((start, i, entity_type))
            start = None
            entity_type = None

    if start is not None:
        entities.append((start, len(tags), entity_type))

    return entities


def main():
    data_path = os.path.join(ROOT, "data", "train.json")
    candidate_path = os.path.join(HERE, "results", "word_inconsistency.csv")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    with open(candidate_path, encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    candidate_words = {row["word"] for row in candidates}
    contexts = defaultdict(list)

    for si, sentence in enumerate(data):
        tokens = sentence["tokens"]
        entities = get_entities(sentence["tags"])

        for i, token in enumerate(tokens):
            word = token.lower()

            if word not in candidate_words:
                continue

            covering = [
                (a, b, t)
                for a, b, t in entities
                if a <= i < b
            ]

            if not covering:
                status = "O"
            elif (
                len(covering) == 1
                and covering[0][0] == i
                and covering[0][1] == i + 1
            ):
                status = covering[0][2]
            else:
                status = "PARTIAL"

            start = max(0, i - 10)
            end = min(len(tokens), i + 11)

            contexts[word].append({
                "word": token,
                "sentence_id": si,
                "status": status,
                "context": " ".join(tokens[start:end])
            })

    output_dir = os.path.join(HERE, "results")
    txt_path = os.path.join(output_dir, "all_word_contexts.txt")
    csv_path = os.path.join(output_dir, "all_word_contexts.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        for word in sorted(contexts):
            f.write(f"\n{'=' * 75}\n")
            f.write(f"WORD: {word.upper()}\n")
            f.write(f"TOTAL OCCURRENCES: {len(contexts[word])}\n")
            f.write(f"{'=' * 75}\n")

            for item in contexts[word]:
                f.write(
                    f"\nSentence {item['sentence_id']} "
                    f"| Status: {item['status']}\n"
                )
                f.write(f"Context: {item['context']}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "sentence_id", "status", "context"]
        )
        writer.writeheader()

        for word in sorted(contexts):
            for item in contexts[word]:
                writer.writerow(item)

    print("Analysis completed.")
    print("Candidate words:", len(contexts))
    print("Total occurrences:", sum(len(v) for v in contexts.values()))
    print("Text report:", txt_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
