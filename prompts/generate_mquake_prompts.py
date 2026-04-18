"""
generate_mquake_prompts.py — Build prompts_mquake.csv and ground_truth_mquake.csv
from the MQuAKE-Remastered CF3k dataset.

Selection strategy: up to --per_hop cases per hop count, filtered to cases where
Gemma 2 2B can answer the question correctly. This ensures every prompt we spend
GPU time attributing is one where the model actually gets the right answer.

Prompts are wrapped in chat format so Gemma sees the same context during attribution:
  Human: <question> Answer in one word.
  Assistant:

prompts_mquake.csv      → slug, prompt, transcoder_set  (wrapped prompts, passing cases only)
ground_truth_mquake.csv → slug, prompt, intermediate_concept, correct_answer,
                           hop_type, num_hops, notes
                           (full questions + sub-hop rows for each passing case)

Usage:
    python generate_mquake_prompts.py
    python generate_mquake_prompts.py --per_hop 20
    python generate_mquake_prompts.py --per_hop 10 --no_filter       # skip Gemma check
    python generate_mquake_prompts.py --per_hop 10 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPTS_CSV = PROMPTS_DIR / "prompts_mquake.csv"
GROUND_TRUTH_CSV = PROMPTS_DIR / "ground_truth_mquake.csv"

# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------

def wrap_prompt(question: str) -> str:
    """Wrap a raw question in the chat format Gemma sees during attribution."""
    q = question.strip().rstrip("?").strip()
    return f"Human: {q}? Answer in one word.\nAssistant:"


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.,;:!?\"'()\[\]]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answers_match(generated: str, correct: str) -> bool:
    """Return True if generated text contains or is contained by correct answer."""
    g = _normalize(generated)
    c = _normalize(correct)
    if not g or len(g) < 2:
        return False
    return c in g or g in c


def case_answers_match(generated: str, case: dict) -> bool:
    """Check generated text against answer and all answer_alias entries."""
    if answers_match(generated, case["answer"]):
        return True
    for alias in (case.get("answer_alias") or []):
        if alias and answers_match(generated, str(alias)):
            return True
    return False


# ---------------------------------------------------------------------------
# Selection strategy
# ---------------------------------------------------------------------------

CANDIDATES_PER_HOP = {2: 80, 3: 300, 4: 300}

# Max cases allowed to share the same first-hop entity (prevents clustering like Beatles x11)
MAX_CHAIN_DUPLICATES = 3


def select_balanced(all_cases: list[dict], targets: dict[int, int]) -> list[dict]:
    """
    Feed CANDIDATES_PER_HOP[n] candidates per hop count to Gemma for filtering.
    Shuffles each bucket randomly so no single entity dominates.
    Single-word answers are still preferred within the shuffled bucket.
    """
    import random

    by_hops: dict[int, list[dict]] = defaultdict(list)
    for case in all_cases:
        hops = case.get("single_hops") or []
        answer = (case.get("answer") or "").strip()
        question = (case.get("questions") or [""])[0].strip()
        if not hops or not answer or not question:
            continue
        by_hops[len(hops)].append(case)

    selected: list[dict] = []
    for n_hops in sorted(by_hops):
        target = targets.get(n_hops, 0)
        if target == 0:
            continue
        bucket = by_hops[n_hops]
        random.shuffle(bucket)
        # Within shuffled bucket, still prefer single-word answers
        bucket.sort(key=lambda c: len((c.get("answer") or "").split()))
        limit = min(CANDIDATES_PER_HOP.get(n_hops, 100), len(bucket))
        chosen = bucket[:limit]
        print(f"  {n_hops}-hop: {len(bucket)} available, sending {len(chosen)} to Gemma (target={target})")
        selected.extend(chosen)

    missing = [h for h in sorted(targets) if h not in by_hops]
    if missing:
        print(f"  WARNING: hop counts not in CF3k: {missing}")

    return selected


# ---------------------------------------------------------------------------
# Gemma filter
# ---------------------------------------------------------------------------

TOP_K = 3              # correct answer must appear in top-K next-token predictions
MIN_PROB_TOP23 = 0.10  # min probability required when answer is in top-2 or top-3 (not top-1)

SANITY_PROMPT = "The capital of France is"
SANITY_EXPECTED = "Paris"


def _run_sanity_check(model, tokenizer) -> None:
    """Verify probability computation on a known easy prompt before filtering."""
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(wrap_prompt(SANITY_PROMPT), return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = probs.topk(5)
    topk = [(tokenizer.decode([tid]).strip(), p.item()) for tid, p in zip(topk_ids, topk_probs)]
    match = any(answers_match(tok, SANITY_EXPECTED) for tok, _ in topk[:3])
    print(f"  Sanity check: '{SANITY_PROMPT}'")
    print(f"    top5: {[(t, f'{p:.3f}') for t, p in topk]}")
    print(f"    '{SANITY_EXPECTED}' in top-3: {match}  — {'OK' if match else 'WARNING: prob lookup may be broken'}")
    print()


def filter_by_gemma(cases: list[dict], device: str) -> list[dict]:
    """
    Run a single forward pass per case through Gemma 2 2B.

    Pass logic:
      - Top-1 match: always pass (model picked the right answer, any prob is fine)
      - Top-2 or top-3 match: pass only if the matched token's prob >= MIN_PROB_TOP23

    Probabilities are read directly from the top-K results so there is no
    tokenization mismatch between the decoded token and the prob lookup.
    """
    import gc
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "google/gemma-2-2b"
    print(f"\nLoading {model_name} for answer filtering ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    _run_sanity_check(model, tokenizer)

    passed: list[dict] = []
    n = len(cases)
    for i, case in enumerate(cases):
        question = case["questions"][0].strip()
        prompt = wrap_prompt(question)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_ids = probs.topk(TOP_K)
        topk = [(tokenizer.decode([tid]).strip(), p.item()) for tid, p in zip(topk_ids, topk_probs)]

        all_answers = [case["answer"]] + [str(a) for a in (case.get("answer_alias") or []) if a]

        # Find the first top-K rank where the token matches the answer.
        # Require the token to be at least 3 chars to avoid 'The'/'It' matching
        # aliases that happen to contain those words (e.g. 'the City of Edinburgh').
        matched_rank = None   # 0-indexed: 0 = top-1
        matched_prob = 0.0
        for rank, (tok, p) in enumerate(topk):
            if len(tok) >= 3 and any(answers_match(tok, ans) for ans in all_answers):
                matched_rank = rank
                matched_prob = p
                break

        if matched_rank == 0:
            passes = True        # top-1 correct: always pass
        elif matched_rank is not None:
            passes = matched_prob >= MIN_PROB_TOP23   # top-2/3: need sufficient prob
        else:
            passes = False

        top1_token = topk[0][0]
        correct = case["answer"]
        topk_tokens = [t for t, _ in topk]
        if passes:
            case["_answer_prob"] = round(matched_prob, 4)
            case["_answer_rank"] = (matched_rank or 0) + 1
            case["_top1_token"] = top1_token
            passed.append(case)
            status = "PASS"
        else:
            status = "FAIL"

        rank_str = f"rank={matched_rank + 1}" if matched_rank is not None else "not in top3"
        print(
            f"  [{i+1}/{n}] {status}  top3={topk_tokens}  "
            f"answer_prob={matched_prob:.3f}  {rank_str}  expected='{correct}'  ({case['case_id']})"
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_hops: dict[int, int] = defaultdict(int)
    for case in passed:
        by_hops[len(case["single_hops"])] += 1
    breakdown = ", ".join(f"{h}-hop: {by_hops[h]}" for h in sorted(by_hops))
    print(f"\n  Passed: {len(passed)}/{n} cases  [{breakdown}]")
    return passed


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def clean_cloze(text: str) -> str:
    return text.strip().rstrip("?.,;:")


def make_slug(case_id: int, suffix: str = "") -> str:
    return f"mquake-{case_id}{suffix}"


def build_rows(case: dict, include_subhops: bool) -> tuple[dict, list[dict]]:
    """
    Return (prompt_row, ground_truth_rows) for one MQuAKE case.

    The prompt stored in prompts_mquake.csv is the wrapped chat-format string
    so attribution sees the same context as the Gemma filter check.
    """
    case_id: int = case["case_id"]
    raw_question: str = case["questions"][0].strip()
    wrapped_prompt: str = wrap_prompt(raw_question)
    final_answer: str = case["answer"].strip()
    hops: list[dict] = case["single_hops"]
    num_hops: int = len(hops)

    intermediate_answers = [h["answer"].strip() for h in hops[:-1]]
    intermediate_concept = " | ".join(intermediate_answers) if intermediate_answers else "N/A"
    chain_str = " → ".join(h["answer"].strip() for h in hops)

    full_slug = make_slug(case_id)

    prompt_row = {"slug": full_slug, "prompt": wrapped_prompt, "transcoder_set": "gemma"}

    ground_truth_rows = [
        {
            "slug": full_slug,
            "prompt": raw_question,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cap_by_hop(cases: list[dict], targets: dict[int, int]) -> list[dict]:
    """After Gemma filtering, deduplicate by first-hop entity then cap to target."""
    by_hops: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        by_hops[len(case["single_hops"])].append(case)
    result = []
    for n in sorted(by_hops):
        target = targets.get(n, len(by_hops[n]))
        # Cap by first-hop entity to prevent clustering (e.g. Beatles x11, football x3)
        chain_counts: dict[str, int] = {}
        deduped = []
        for case in by_hops[n]:
            key = (case["single_hops"][0].get("answer") or "").strip().lower()
            if chain_counts.get(key, 0) < MAX_CHAIN_DUPLICATES:
                chain_counts[key] = chain_counts.get(key, 0) + 1
                deduped.append(case)
        kept = deduped[:target]
        dropped_chain = len(by_hops[n]) - len(deduped)
        dropped_cap = len(deduped) - len(kept)
        print(f"  {n}-hop: {len(kept)} kept  (chain dedup dropped {dropped_chain}, cap dropped {dropped_cap})")
        result.extend(kept)
    return result


def parse_targets(s: str) -> dict[int, int]:
    """Parse '2:20,3:15,4:15' into {2: 20, 3: 15, 4: 15}."""
    result = {}
    for part in s.split(","):
        hops, count = part.strip().split(":")
        result[int(hops)] = int(count)
    return result


def run(targets: dict[int, int], include_subhops: bool,
        filter_by_model: bool, device: str) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        raise SystemExit(
            "huggingface_hub and pyarrow are required.\n"
            "Install with:  pip install huggingface_hub pyarrow"
        )

    # Load CF3k parquet directly — avoids the datasets schema-cast bug
    print("Listing MQuAKE-Remastered repo files ...")
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
        raise SystemExit(f"Unexpected schema — missing fields: {missing_fields}")

    print(f"\nTargets: { {k: targets[k] for k in sorted(targets)} }")
    cases = select_balanced(all_cases, targets)
    print(f"  Total candidates for Gemma: {len(cases)}")

    if filter_by_model:
        cases = filter_by_gemma(cases, device)
    else:
        print("\n  Skipping Gemma filter (--no_filter)")

    if not cases:
        raise SystemExit("No cases passed the filter. Increase CANDIDATE_MULTIPLIER or check dataset.")

    print(f"\nCapping to targets after filtering:")
    cases = cap_by_hop(cases, targets)

    all_prompt_rows: list[dict] = []
    all_ground_truth_rows: list[dict] = []

    for case in cases:
        p_row, gt_rows = build_rows(case, include_subhops)
        all_prompt_rows.append(p_row)
        all_ground_truth_rows.extend(gt_rows)

    # ---- prompts_mquake.csv ----
    with open(PROMPTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "prompt", "transcoder_set"])
        w.writeheader()
        w.writerows(all_prompt_rows)
    print(f"\nWrote {len(all_prompt_rows)} rows → {PROMPTS_CSV}")

    # ---- ground_truth_mquake.csv ----
    gt_fields = ["slug", "prompt", "intermediate_concept", "correct_answer",
                 "hop_type", "num_hops", "notes"]
    with open(GROUND_TRUTH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_fields)
        w.writeheader()
        w.writerows(all_ground_truth_rows)
    print(f"Wrote {len(all_ground_truth_rows)} rows → {GROUND_TRUTH_CSV}")


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description="Generate Gemma-filtered MQuAKE prompt CSVs.")
    parser.add_argument(
        "--targets", type=str, default="2:20,3:15,4:15",
        help="Target cases per hop count, format '2:20,3:15,4:15' (default: 2:20,3:15,4:15)",
    )
    parser.add_argument(
        "--include_subhops", type=lambda x: x.lower() != "false", default=True,
        help="Include sub-hop rows in ground_truth_mquake.csv (default=true)",
    )
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Skip the Gemma answer-correctness filter",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for Gemma inference (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run(
        targets=parse_targets(args.targets),
        include_subhops=args.include_subhops,
        filter_by_model=not args.no_filter,
        device=args.device,
    )
