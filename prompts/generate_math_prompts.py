"""
Generate ~100 addition prompts for circuit tracing on Gemma-2-2B.

Motivation
----------
This set is built to stress-test a known limitation of our grouping pipeline:
on arithmetic prompts like "19 + 23 =", the LLM-generated supernodes may be too
vague/boring (e.g. "numbers", "say a number", "addition") to capture the actual
addition mechanism the model uses internally. To see *where* the grouping gets
vague, the prompts span structured arithmetic categories (carry / no-carry,
round numbers, ties, repeated digits) at two magnitudes.

Composition (default 100 total)
  - 50 two-digit + two-digit   (operands 10..99)
  - 50 three-digit + three-digit (operands 100..999)

Each half is spread across structured categories so we can later correlate
supernode vagueness with problem type (e.g. does a carry produce a richer graph
than a no-carry sum?).

Unlike generate_wikipedia_prompts.py this needs no model — the ground truth is
exact arithmetic. Whether the model actually predicts the sum can be read off
the target logit in each attribution graph afterwards.

Usage:
    python prompts/generate_math_prompts.py
    python prompts/generate_math_prompts.py --n-per-half 50 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = OUT_DIR / "prompts_math.csv"
GROUND_TRUTH_FILE = OUT_DIR / "ground_truth_math.csv"

TRANSCODER_SET = "gemma"
# Prompt format. NO spaces around the operators, and NO trailing space.
# Rationale (measured on Gemma-2-2b):
#   "13 + 13 ="  -> top token is a SPACE (p~0.78); attribution would target
#                   whitespace/formatting, not arithmetic.
#   "13 + 13 = " -> predicts the digit (p~0.86) BUT attribute-batch calls
#                   prompt.strip(), which deletes the trailing space and reverts
#                   to the broken case.
#   "13+13="     -> predicts the first answer digit directly (p~0.6-0.83 across
#                   2/3-digit, carries, ties) AND is unchanged by .strip().
# So the tightly-packed form is the only one that both targets the arithmetic
# and survives the pipeline's strip().
PROMPT_TEMPLATE = "{a}+{b}="


# ---------------------------------------------------------------------------
# Carry analysis — how many column carries does a + b trigger?
# ---------------------------------------------------------------------------

def num_carries(a: int, b: int) -> int:
    """Count the carries produced when adding a and b column by column."""
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        col = (a % 10) + (b % 10) + carry
        carry = 1 if col >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def is_round_10(n: int) -> bool:
    return n % 10 == 0


def is_round_100(n: int) -> bool:
    return n % 100 == 0


def is_repdigit(n: int) -> bool:
    s = str(n)
    return len(set(s)) == 1


# ---------------------------------------------------------------------------
# Category predicates — each maps (a, b) -> bool. A pair is eligible for a
# category if the predicate holds. Categories are sampled to hit a target
# count so the final set has balanced structural coverage.
# ---------------------------------------------------------------------------

def two_digit_categories() -> list[tuple[str, callable]]:
    return [
        ("no_carry",        lambda a, b: num_carries(a, b) == 0),
        ("one_carry",       lambda a, b: num_carries(a, b) == 1),
        ("two_carry",       lambda a, b: num_carries(a, b) == 2),  # cascades into 3 digits
        ("round_tens",      lambda a, b: is_round_10(a) and is_round_10(b)),
        ("tie",             lambda a, b: a == b),
        ("repeated_digit",  lambda a, b: is_repdigit(a) and is_repdigit(b)),
    ]


def three_digit_categories() -> list[tuple[str, callable]]:
    return [
        ("no_carry",        lambda a, b: num_carries(a, b) == 0),
        ("one_carry",       lambda a, b: num_carries(a, b) == 1),
        ("two_carry",       lambda a, b: num_carries(a, b) == 2),
        ("three_carry",     lambda a, b: num_carries(a, b) == 3),  # cascades into 4 digits
        ("round_hundreds",  lambda a, b: is_round_100(a) and is_round_100(b)),
        ("round_tens",      lambda a, b: is_round_10(a) and is_round_10(b)
                                          and not (is_round_100(a) and is_round_100(b))),
        ("tie",             lambda a, b: a == b),
        ("repeated_digit",  lambda a, b: is_repdigit(a) and is_repdigit(b)),
    ]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def allocate(n_total: int, n_buckets: int) -> list[int]:
    """Split n_total as evenly as possible across n_buckets (earlier buckets +1)."""
    base, extra = divmod(n_total, n_buckets)
    return [base + (1 if i < extra else 0) for i in range(n_buckets)]


def sample_half(
    lo: int,
    hi: int,
    categories: list[tuple[str, callable]],
    n_target: int,
    rng: random.Random,
    seen: set[tuple[int, int]],
) -> list[dict]:
    """Sample n_target structured pairs in [lo, hi], balanced across categories.

    Pairs are unordered (a <= b) and globally deduplicated via `seen`. If a
    category is too sparse to fill its quota, the shortfall rolls into a final
    uniform top-up so the half always reaches n_target.
    """
    quotas = allocate(n_target, len(categories))
    selected: list[dict] = []

    for (cat_name, pred), quota in zip(categories, quotas):
        # All eligible unordered pairs for this category.
        pool = [
            (a, b)
            for a in range(lo, hi + 1)
            for b in range(a, hi + 1)
            if pred(a, b)
        ]
        rng.shuffle(pool)
        taken = 0
        for a, b in pool:
            if taken >= quota:
                break
            if (a, b) in seen:
                continue
            seen.add((a, b))
            selected.append(_make_row(a, b, cat_name))
            taken += 1

    # Uniform top-up to cover any category shortfalls.
    while len(selected) < n_target:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        selected.append(_make_row(a, b, f"topup_carry{num_carries(a, b)}"))

    return selected


def _make_row(a: int, b: int, category: str) -> dict:
    width = max(len(str(a)), len(str(b)))
    prompt = PROMPT_TEMPLATE.format(a=a, b=b)
    answer = a + b
    return {
        "slug": f"add-{width}d-{a}-{b}",
        "prompt": prompt,
        "a": a,
        "b": b,
        "answer": answer,
        "operand_digits": width,
        "category": category,
        "num_carries": num_carries(a, b),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-half", type=int, default=50,
                        help="Prompts per magnitude (two-digit and three-digit). Default 50 -> 100 total.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[tuple[int, int]] = set()

    two_digit = sample_half(10, 99, two_digit_categories(), args.n_per_half, rng, seen)
    three_digit = sample_half(100, 999, three_digit_categories(), args.n_per_half, rng, seen)
    rows = two_digit + three_digit

    # --- prompts CSV (for attribute-batch) ---
    with PROMPTS_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slug", "prompt", "transcoder_set"])
        for r in rows:
            w.writerow([r["slug"], r["prompt"], TRANSCODER_SET])

    # --- ground truth CSV (for analysis) — matches the standard schema ---
    with GROUND_TRUTH_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slug", "prompt", "intermediate_concept", "correct_answer", "hop_type", "notes"])
        for r in rows:
            w.writerow([
                r["slug"],
                r["prompt"],
                "N/A",
                r["answer"],
                "addition",
                f"operand_digits={r['operand_digits']} category={r['category']} "
                f"num_carries={r['num_carries']} ({r['a']}+{r['b']}={r['answer']})",
            ])

    # --- summary ---
    print(f"Wrote {len(rows)} prompts -> {PROMPTS_FILE}")
    print(f"Wrote {len(rows)} ground-truth rows -> {GROUND_TRUTH_FILE}")
    for label, half in (("two-digit", two_digit), ("three-digit", three_digit)):
        from collections import Counter
        cats = Counter(r["category"] for r in half)
        print(f"\n  {label} ({len(half)}):")
        for cat, n in sorted(cats.items()):
            print(f"    {cat:18s} {n}")
    print("\nSamples:")
    for r in (two_digit[:3] + three_digit[:3]):
        print(f"  [{r['category']:14s}] {r['prompt']} {r['answer']}")


if __name__ == "__main__":
    main()
