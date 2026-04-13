"""
generate_mquake_prompts.py — Build prompts_mquake.csv and ground_truth_mquake.csv
from the MQuAKE-Remastered CF3k dataset.

Selection strategy: 10 cases per hop count (1-hop through 5-hop where available).
MQuAKE CF3k contains 2-hop and 3-hop cases; 4-hop may appear in smaller numbers.

prompts_mquake.csv        → slug, prompt, transcoder_set  (full questions only)
ground_truth_mquake.csv   → slug, prompt, intermediate_concept, correct_answer,
                             hop_type, num_hops, notes
                             (full questions + sub-hop rows for each case)

Usage:
    python generate_mquake_prompts.py
    python generate_mquake_prompts.py --per_hop 20
    python generate_mquake_prompts.py --per_hop 10 --include_subhops false
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPTS_CSV = PROMPTS_DIR / "prompts_mquake.csv"
GROUND_TRUTH_CSV = PROMPTS_DIR / "ground_truth_mquake.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_cloze(text: str) -> str:
    """Strip whitespace and trailing punctuation from a cloze prompt."""
    return text.strip().rstrip("?.,;:")


def make_slug(case_id: int, suffix: str = "") -> str:
    return f"mquake-{case_id}{suffix}"


def build_rows(case: dict, include_subhops: bool) -> tuple[dict, list[dict]]:
    """
    Return (prompt_row, ground_truth_rows) for one MQuAKE case.

    prompt_row       — one row for prompts_mquake.csv (full question only)
    ground_truth_rows — full question row + optional sub-hop rows
    """
    case_id: int = case["case_id"]
    full_question: str = case["questions"][0].strip()
    final_answer: str = case["answer"].strip()
    hops: list[dict] = case["single_hops"]
    num_hops: int = len(hops)

    intermediate_answers = [h["answer"].strip() for h in hops[:-1]]
    intermediate_concept = " | ".join(intermediate_answers) if intermediate_answers else "N/A"
    chain_str = " → ".join(h["answer"].strip() for h in hops)

    full_slug = make_slug(case_id)

    # prompts_mquake.csv: full question only, 3 columns
    prompt_row = {"slug": full_slug, "prompt": full_question, "transcoder_set": "gemma"}

    # ground_truth_mquake.csv: full question row
    ground_truth_rows = [
        {
            "slug": full_slug,
            "prompt": full_question,
            "intermediate_concept": intermediate_concept,
            "correct_answer": final_answer,
            "hop_type": f"{num_hops}-hop",
            "num_hops": num_hops,
            "notes": chain_str,
        },
    ]

    if include_subhops:
        for k, hop in enumerate(hops, start=1):
            cloze = clean_cloze(hop.get("cloze", "") or hop.get("question", ""))
            if not cloze:
                continue
            ground_truth_rows.append(
                {
                    "slug": make_slug(case_id, f"-h{k}"),
                    "prompt": cloze,
                    "intermediate_concept": "N/A",
                    "correct_answer": hop["answer"].strip(),
                    "hop_type": f"1-hop (hop {k} of {num_hops})",
                    "num_hops": 1,
                    "notes": f"sub-question {k}/{num_hops} of case {case_id}",
                }
            )

    return prompt_row, ground_truth_rows


def select_balanced(all_cases: list[dict], per_hop: int) -> list[dict]:
    """
    Return up to `per_hop` cases for each distinct hop count.
    Logs how many cases are available at each hop level.
    """
    by_hops: dict[int, list[dict]] = defaultdict(list)
    for case in all_cases:
        n = len(case["single_hops"])
        by_hops[n].append(case)

    selected: list[dict] = []
    for n_hops in sorted(by_hops):
        available = len(by_hops[n_hops])
        chosen = by_hops[n_hops][:per_hop]
        print(f"  {n_hops}-hop: {available} available, using {len(chosen)}")
        selected.extend(chosen)

    hop_counts = sorted(by_hops)
    missing = [h for h in range(1, 6) if h not in hop_counts]
    if missing:
        print(f"  Note: hop counts not in CF3k: {missing}")

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(per_hop: int, include_subhops: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        raise SystemExit(
            "huggingface_hub and pyarrow are required.\n"
            "Install with:  pip install huggingface_hub pyarrow"
        )

    # Load CF3k parquet directly — avoids the datasets schema-cast bug that
    # occurs when load_dataset tries to prepare the CF6334 split.
    print("Listing MQuAKE-Remastered repo files …")
    all_files = list(list_repo_files("henryzhongsc/MQuAKE-Remastered", repo_type="dataset"))
    cf3k_parquets = sorted(f for f in all_files if "CF3k" in f and f.endswith(".parquet"))
    if not cf3k_parquets:
        raise SystemExit("No CF3k parquet files found. Available:\n" + "\n".join(all_files))

    print(f"  Downloading {len(cf3k_parquets)} parquet file(s): {cf3k_parquets}")
    tables = []
    for fname in cf3k_parquets:
        local = hf_hub_download(
            repo_id="henryzhongsc/MQuAKE-Remastered",
            filename=fname,
            repo_type="dataset",
        )
        tables.append(pq.read_table(local))

    all_cases: list[dict] = pa.concat_tables(tables).to_pylist()
    print(f"  {len(all_cases)} total cases in CF3k")

    required = {"case_id", "questions", "answer", "single_hops"}
    missing_fields = required - set(all_cases[0].keys())
    if missing_fields:
        raise SystemExit(f"Unexpected schema — missing: {missing_fields}")

    # Select up to per_hop cases per hop count
    print(f"\nSelecting up to {per_hop} cases per hop count:")
    cases = select_balanced(all_cases, per_hop)
    print(f"  Total selected: {len(cases)} cases\n")

    all_prompt_rows: list[dict] = []
    all_ground_truth_rows: list[dict] = []

    for case in cases:
        p_row, gt_rows = build_rows(case, include_subhops)
        all_prompt_rows.append(p_row)
        all_ground_truth_rows.extend(gt_rows)

    # ---- prompts_mquake.csv: full questions only, 3 columns ----
    with open(PROMPTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "prompt", "transcoder_set"])
        w.writeheader()
        w.writerows(all_prompt_rows)
    print(f"Wrote {len(all_prompt_rows)} rows → {PROMPTS_CSV}")

    # ---- ground_truth_mquake.csv: full + sub-hop rows ----
    gt_fields = ["slug", "prompt", "intermediate_concept", "correct_answer",
                 "hop_type", "num_hops", "notes"]
    with open(GROUND_TRUTH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_fields)
        w.writeheader()
        w.writerows(all_ground_truth_rows)
    print(f"Wrote {len(all_ground_truth_rows)} rows → {GROUND_TRUTH_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced MQuAKE prompt CSVs.")
    parser.add_argument(
        "--per_hop", type=int, default=10,
        help="Cases to include per hop count (default=10)",
    )
    parser.add_argument(
        "--include_subhops", type=lambda x: x.lower() != "false", default=True,
        help="Include sub-hop rows in ground_truth_mquake.csv (default=true)",
    )
    args = parser.parse_args()
    run(args.per_hop, args.include_subhops)
