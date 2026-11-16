
import json
import csv
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def load_train():
    path = os.path.join(ROOT, "data", "train.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_entities(tags):
    entities = []
    start = None
    current_type = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((start, i, current_type))
            start = i
            current_type = tag[2:]

        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if start is None or current_type != entity_type:
                if start is not None:
                    entities.append((start, i, current_type))
                start = i
                current_type = entity_type

        else:
            if start is not None:
                entities.append((start, i, current_type))
                start = None
                current_type = None

    if start is not None:
        entities.append((start, len(tags), current_type))

    return entities


def analyse_words(train, min_occ=5, min_minor=0.10):
    stats = defaultdict(Counter)

    for sentence in train:
        tokens = sentence["tokens"]
        entities = get_entities(sentence["tags"])

        for i, token in enumerate(tokens):
            word = token.lower()

            covering = [
                (a, b, entity_type)
                for a, b, entity_type in entities
                if a <= i < b
            ]

            if not covering:
                status = "O"
            elif len(covering) == 1 and covering[0][0] == i and covering[0][1] == i + 1:
                status = covering[0][2]
            else:
                status = "partial"

            stats[word][status] += 1

    rows = []

    for word, counts in stats.items():
        total = sum(counts.values())

        if total < min_occ:
            continue

        relevant = counts["Chemical"] + counts["Disease"] + counts["O"]

        if relevant == 0:
            continue

        distribution = [
            counts["Chemical"],
            counts["Disease"],
            counts["O"]
        ]

        nonzero = [x for x in distribution if x > 0]

        if len(nonzero) < 2:
            continue

        minority_share = min(nonzero) / relevant

        if minority_share < min_minor:
            continue

        rows.append({
            "word": word,
            "total": total,
            "chemical": counts["Chemical"],
            "disease": counts["Disease"],
            "unannotated": counts["O"],
            "partial": counts["partial"],
            "minority_share": round(minority_share, 4)
        })

    rows.sort(
        key=lambda r: (-r["minority_share"], -r["total"], r["word"])
    )

    return rows


def main():
    train = load_train()

    rows = analyse_words(train)

    output_dir = os.path.join(HERE, "results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, "word_inconsistency.csv"
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "word", "total", "chemical", "disease",
            "unannotated", "partial", "minority_share"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("Total suspicious words:", len(rows))
    print("\nTop 30 candidates:")

    for row in rows[:30]:
        print(row)

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
