"""
Test whether the models can actually answer multi-hop prompts before
spending GPU time on full attribution.

Two approaches:
  1. Gemma 2 2B (base) with few-shot prompting
  2. Gemma 3 4B IT (instruction-tuned) with chat formatting

Usage — generate answers:
  python test_model_answers.py --csv ../prompts/prompts_max_hops.csv
  python test_model_answers.py --csv ../prompts/prompts_max_hops.csv --model gemma3-4b-it
  python test_model_answers.py --csv ../prompts/prompts_max_hops.csv --model both
  python test_model_answers.py --csv ../prompts/prompts_max_hops.csv --start-row 22  # 3-hop+

Usage — analyze results:
  python test_model_answers.py analyze --answers ../prompts/prompts_max_hops_answers.csv \
                                       --ground-truth ../prompts/ground_truth_max_hops.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Few-shot examples for base model (1-hop facts the model definitely knows)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """\
Q: What is the capital of France?
A: Paris

Q: The currency of Japan is
A: the yen

Q: Who wrote Romeo and Juliet?
A: William Shakespeare

Q: The largest planet in the solar system is
A: Jupiter

Q: What is the official language of the country where the Eiffel Tower is located?
A: French

"""



# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

# Aliases so fuzzy matching can handle common variations
ANSWER_ALIASES = {
    "pound sterling": ["pound", "gbp", "british pound", "pound sterling", "the pound"],
    "us dollar": ["us dollar", "usd", "the us dollar", "the dollar", "dollar",
                  "united states dollar", "the u.s. dollar", "u.s. dollar"],
    "euro": ["euro", "the euro", "eur"],
    "yen": ["yen", "the yen", "jpy", "japanese yen"],
    "yuan (renminbi)": ["yuan", "renminbi", "rmb", "cny", "the yuan"],
    "north america": ["north america"],
    "washington d.c.": ["washington d.c.", "washington dc", "washington, dc",
                        "washington, d.c.", "washington"],
    "american english": ["american english", "english"],
    "english": ["english", "the english language"],
    "elizabeth ii": ["elizabeth ii", "queen elizabeth ii", "queen elizabeth",
                     "elizabeth", "the queen"],
    "donald trump": ["donald trump", "trump", "president trump"],
    "association football": ["association football", "football", "soccer"],
    "jesus christ": ["jesus christ", "jesus", "christ"],
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles for comparison."""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML tags
    text = re.sub(r"[.,;:!?\"'()]+", "", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answers_match(model_answer: str, correct_answer: str) -> bool:
    """Check if model_answer matches correct_answer, with alias expansion."""
    m = normalize(model_answer)
    c = normalize(correct_answer)

    if not m or m == ":" or len(m) < 2:
        return False

    # Direct substring match (either direction)
    if c in m or m in c:
        return True

    # Check aliases: find the alias group for the correct answer
    for canonical, aliases in ANSWER_ALIASES.items():
        c_in_group = c in [normalize(a) for a in aliases] or normalize(canonical) == c
        if c_in_group:
            return any(normalize(a) in m or m in normalize(a) for a in aliases)

    return False


def parse_hop_count(row: dict) -> int | None:
    """Extract numeric hop count from a ground-truth row.

    Checks (in order): num_hops column, hop_type prefix, arrow count in hop_type,
    and finally the notes field.
    """
    # Explicit num_hops column (mquake ground truth has this)
    num_hops = row.get("num_hops", "").strip()
    if num_hops.isdigit():
        return int(num_hops)

    hop_type = row.get("hop_type", "").strip()
    m = re.match(r"(\d+)-hop", hop_type)
    if m:
        return int(m.group(1))

    arrows = hop_type.count("→")
    if arrows > 0:
        return arrows

    notes = row.get("notes", "").strip()
    m = re.search(r"(\d+)-hop", notes)
    if m:
        return int(m.group(1))

    return None


def load_ground_truth(path: str) -> dict:
    """Load ground truth CSV into {slug: {correct_answer, hop_type, ...}}.

    Skips sub-hop rows (slugs containing '-h' followed by digits, e.g. mquake-1001-h1).
    """
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row["slug"].strip()
            # Skip sub-hop decomposition rows (e.g. mquake-1001-h1)
            if re.search(r"-h\d+$", slug):
                continue
            gt[slug] = {
                "correct_answer": row.get("correct_answer", "").strip(),
                "hop_type": row.get("hop_type", "").strip(),
                "num_hops": parse_hop_count(row),
                "intermediate_concept": row.get("intermediate_concept", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
    return gt


def analyze(answers_path: str, gt_path: str, output_path: str | None = None):
    """Score model answers against ground truth and print analysis."""

    gt = load_ground_truth(gt_path)

    with open(answers_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        answer_rows = list(reader)

    # Detect which model columns are present
    model_cols = [c for c in fieldnames if c not in ("slug", "prompt")]
    if not model_cols:
        print("ERROR: No model answer columns found in the answers CSV.")
        return

    # --- Per-row scoring ---
    scored_rows = []
    for row in answer_rows:
        slug = row["slug"].strip()
        if slug not in gt:
            print(f"  WARNING: No ground truth for slug '{slug}', skipping.")
            continue
        correct = gt[slug]["correct_answer"]
        hop_type = gt[slug]["hop_type"]
        hops = gt[slug]["num_hops"]

        scored = {
            "slug": slug,
            "prompt": row.get("prompt", ""),
            "correct_answer": correct,
            "hops": hops,
            "hop_type": hop_type,
            "intermediate": gt[slug]["intermediate_concept"],
        }
        for col in model_cols:
            ans = row.get(col, "").strip()
            is_correct = answers_match(ans, correct)
            scored[col] = ans
            scored[f"{col}_correct"] = is_correct
        scored_rows.append(scored)

    # --- Per-model, per-hop accuracy ---
    # {model_col: {hop_count: [bool, ...]}}
    by_hop: dict[str, dict[int | None, list[bool]]] = {
        col: defaultdict(list) for col in model_cols
    }
    for sr in scored_rows:
        for col in model_cols:
            by_hop[col][sr["hops"]].append(sr[f"{col}_correct"])

    # --- Print detailed results ---
    print()
    print("=" * 90)
    print("  DETAILED RESULTS")
    print("=" * 90)

    for sr in scored_rows:
        markers = []
        for col in model_cols:
            mark = "OK" if sr[f"{col}_correct"] else "X "
            markers.append(f"{col}: [{mark}] {sr[col]}")
        print(f"\n  {sr['slug']}  ({sr['hops']}-hop)")
        print(f"    Prompt:  {sr['prompt']}")
        print(f"    Answer:  {sr['correct_answer']}")
        print(f"    Chain:   {sr['intermediate']}")
        for m in markers:
            print(f"    {m}")

    # --- Print accuracy table ---
    all_hops = sorted(set(sr["hops"] for sr in scored_rows if sr["hops"] is not None))

    print()
    print("=" * 90)
    print("  ACCURACY BY HOP COUNT")
    print("=" * 90)

    # Header
    header = f"  {'Model':<25}"
    for h in all_hops:
        header += f"  {h}-hop"
    header += "   Total"
    print(header)
    print("  " + "-" * (25 + 7 * len(all_hops) + 8))

    for col in model_cols:
        line = f"  {col:<25}"
        total_correct = 0
        total_count = 0
        for h in all_hops:
            results = by_hop[col].get(h, [])
            n_correct = sum(results)
            n_total = len(results)
            total_correct += n_correct
            total_count += n_total
            line += f"  {n_correct:>2}/{n_total:<2} "
        pct = (total_correct / total_count * 100) if total_count else 0
        line += f"  {total_correct:>2}/{total_count} ({pct:.0f}%)"
        print(line)

    # --- Failure analysis ---
    print()
    print("=" * 90)
    print("  FAILURE PATTERNS")
    print("=" * 90)

    for col in model_cols:
        failures = [sr for sr in scored_rows if not sr[f"{col}_correct"]]
        if not failures:
            print(f"\n  {col}: all correct!")
            continue

        # Categorize failures
        intermediate_answer = 0
        empty_or_junk = 0
        wrong_entity = 0
        for sr in failures:
            ans = normalize(sr[col])
            intermediates = [normalize(x) for x in sr["intermediate"].split("|")]
            if not ans or ans == ":" or len(ans) < 2:
                empty_or_junk += 1
            elif any(i in ans or ans in i for i in intermediates if i.strip()):
                intermediate_answer += 1
            else:
                wrong_entity += 1

        total_f = len(failures)
        print(f"\n  {col}: {total_f} failures")
        print(f"    Answered intermediate hop:  {intermediate_answer:>3}  ({intermediate_answer/total_f*100:.0f}%)")
        print(f"    Empty / junk output:        {empty_or_junk:>3}  ({empty_or_junk/total_f*100:.0f}%)")
        print(f"    Wrong entity:               {wrong_entity:>3}  ({wrong_entity/total_f*100:.0f}%)")

    # --- Write scored CSV ---
    out = output_path or answers_path.replace(".csv", "_scored.csv")
    out_fields = ["slug", "prompt", "correct_answer", "hops", "hop_type", "intermediate"]
    for col in model_cols:
        out_fields += [col, f"{col}_correct"]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for sr in scored_rows:
            writer.writerow({k: sr.get(k, "") for k in out_fields})

    print(f"\n  Scored CSV written to: {out}")
    print()


def load_gemma2_base(device: str):
    """Load Gemma 2 2B base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "google/gemma-2-2b"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tokenizer


def load_gemma3_4b_it(device: str):
    """Load Gemma 3 4B instruction-tuned model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "google/gemma-3-4b-it"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tokenizer


def _truncate_response(text: str, stop_markers: list[str] | None = None) -> str:
    """Keep only the model's answer, cutting at the first stop marker.

    For few-shot: stop at "Q:" (next question) or "\\n\\n" (blank line).
    This lets the model give multi-word answers like "Guru Nanak Dev"
    without being cut off at the first newline.
    """
    if stop_markers is None:
        stop_markers = ["\nQ:", "\n\n"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate_base(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Generate with few-shot prompting for the base model."""
    import torch

    full_prompt = FEW_SHOT_EXAMPLES + f"Q: {prompt}\nA:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _truncate_response(text)


def generate_base_completion(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Generate with plain completion (no few-shot) — the prompt ends with 'is'."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _truncate_response(text, stop_markers=["\n\n"])


def generate_it(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """Generate with chat template for instruction-tuned model."""
    import torch

    messages = [
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "Answer with just the name/word, nothing else."
            ),
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _truncate_response(text, stop_markers=["\n\n"])


def main():
    parser = argparse.ArgumentParser(description="Test model answers on multi-hop prompts")
    subparsers = parser.add_subparsers(dest="command")

    # --- generate subcommand (default when no subcommand given) ---
    gen_parser = subparsers.add_parser("generate", help="Generate model answers")
    gen_parser.add_argument("--csv", required=True, help="Path to prompts CSV")
    gen_parser.add_argument(
        "--model",
        choices=["gemma2-base", "gemma3-4b-it", "both"],
        default="both",
        help="Which model(s) to test",
    )
    gen_parser.add_argument("--start-row", type=int, default=1, help="Start from this row (1-indexed, after header)")
    gen_parser.add_argument("--max-rows", type=int, default=None, help="Max rows to process")
    gen_parser.add_argument("--device", default=None, help="Device (default: cuda if available)")
    gen_parser.add_argument("--output", default=None, help="Write results CSV to this path")
    gen_parser.add_argument("--analyze", action="store_true", help="Run analysis after generation")
    gen_parser.add_argument("--ground-truth", default=None, help="Ground truth CSV (required with --analyze)")

    # --- analyze subcommand ---
    ana_parser = subparsers.add_parser("analyze", help="Score answers against ground truth")
    ana_parser.add_argument("--answers", required=True, help="Path to answers CSV (from generate)")
    ana_parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV")
    ana_parser.add_argument("--output", default=None, help="Path for scored output CSV")

    args = parser.parse_args()

    # Default to generate if no subcommand and --csv is present (backward compat)
    if args.command is None:
        # Check for legacy usage: test_model_answers.py --csv ...
        legacy_parser = argparse.ArgumentParser()
        legacy_parser.add_argument("--csv", required=True)
        legacy_parser.add_argument("--model", choices=["gemma2-base", "gemma3-4b-it", "both"], default="both")
        legacy_parser.add_argument("--start-row", type=int, default=1)
        legacy_parser.add_argument("--max-rows", type=int, default=None)
        legacy_parser.add_argument("--device", default=None)
        legacy_parser.add_argument("--output", default=None)
        legacy_parser.add_argument("--analyze", action="store_true")
        legacy_parser.add_argument("--ground-truth", default=None)
        args = legacy_parser.parse_args()
        args.command = "generate"

    if args.command == "analyze":
        analyze(args.answers, args.ground_truth, args.output)
        return

    # --- generate ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows = rows[args.start_row - 1:]
    if args.max_rows:
        rows = rows[:args.max_rows]

    print(f"Testing {len(rows)} prompts on device={args.device}")
    print()

    models = {}
    if args.model in ("gemma2-base", "both"):
        models["gemma2-base"] = load_gemma2_base(args.device)
    if args.model in ("gemma3-4b-it", "both"):
        models["gemma3-4b-it"] = load_gemma3_4b_it(args.device)

    results = []
    for i, row in enumerate(rows):
        slug = row["slug"].strip()
        prompt = row["prompt"].strip()
        print(f"[{i+1}/{len(rows)}] {slug}")
        print(f"  Prompt: {prompt}")

        result = {"slug": slug, "prompt": prompt}

        if "gemma2-base" in models:
            model, tokenizer = models["gemma2-base"]
            ans_fewshot = generate_base(model, tokenizer, prompt)
            ans_completion = generate_base_completion(model, tokenizer, prompt)
            result["gemma2_fewshot"] = ans_fewshot
            result["gemma2_completion"] = ans_completion
            print(f"  Gemma2 few-shot:    {ans_fewshot}")
            print(f"  Gemma2 completion:  {ans_completion}")

        if "gemma3-4b-it" in models:
            model, tokenizer = models["gemma3-4b-it"]
            ans_it = generate_it(model, tokenizer, prompt)
            result["gemma3_it"] = ans_it
            print(f"  Gemma3-4B-IT:       {ans_it}")

        results.append(result)
        print()

    out_path = args.output or args.csv.replace(".csv", "_answers.csv")
    fieldnames = ["slug", "prompt"]
    if "gemma2-base" in models:
        fieldnames += ["gemma2_fewshot", "gemma2_completion"]
    if "gemma3-4b-it" in models:
        fieldnames += ["gemma3_it"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {out_path}")

    if args.analyze:
        if not args.ground_truth:
            print("ERROR: --ground-truth is required with --analyze")
            sys.exit(1)
        print()
        analyze(out_path, args.ground_truth)


if __name__ == "__main__":
    main()
