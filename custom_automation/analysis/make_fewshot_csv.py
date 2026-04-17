"""
Prepend few-shot examples to each prompt in a CSV for attribution.

Usage:
  python make_fewshot_csv.py ../prompts/prompts_mquake.csv
  python make_fewshot_csv.py ../prompts/prompts_max_hops_3to5.csv

Outputs: <input>_fewshot.csv with the same slugs but prompts wrapped in
the few-shot Q/A format so attribution sees the full context.
"""

import csv
import sys

FEW_SHOT_PREFIX = (
    "Q: What is the capital of France?\nA: Paris\n\n"
    "Q: The currency of Japan is\nA: the yen\n\n"
    "Q: Who wrote Romeo and Juliet?\nA: William Shakespeare\n\n"
    "Q: The largest planet in the solar system is\nA: Jupiter\n\n"
    "Q: What is the official language of the country where the Eiffel Tower is located?\nA: French\n\n"
)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv_path:
        print("Usage: python make_fewshot_csv.py <prompts.csv>")
        sys.exit(1)

    out_path = csv_path.replace(".csv", "_fewshot.csv")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prompt = row["prompt"].strip()
            row["prompt"] = FEW_SHOT_PREFIX + f"Q: {prompt}\nA:"
            writer.writerow(row)

    print(f"Wrote {len(rows)} few-shot prompts to {out_path}")


if __name__ == "__main__":
    main()
