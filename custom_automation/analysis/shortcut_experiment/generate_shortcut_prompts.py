"""
generate_shortcut_prompts.py — Generate shortcut prompt variants for cases
where the model answered correctly but skipped intermediate reasoning hops.

For each shortcut candidate, GPT rewrites the original question by substituting
a missed intermediate entity directly — removing the indirection.

Example:
  Original:  "What continent is the country in where the director of the
              performer of Back in the U.S.S.R. is a citizen?"
  Shortcut:  "What continent is the country in where The Beatles is from?"
  Shortcut:  "What continent is the country in where Brian Epstein is from?"

Reads:
  analysis/results/hop_analysis.csv       — identifies shortcut candidates
  prompts/ground_truth_mquake.csv         — original prompts and hop chains

Writes:
  analysis/shortcut_experiment/prompts/prompts_shortcut.csv
  analysis/shortcut_experiment/prompts/ground_truth_shortcut.csv

Usage:
    python analysis/shortcut_experiment/generate_shortcut_prompts.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PACKAGE_DIR.parent
HOP_CSV = PACKAGE_DIR / "analysis" / "results" / "hop_analysis.csv"
GROUND_TRUTH_CSV = REPO_ROOT / "prompts" / "ground_truth_mquake.csv"
OUT_DIR = Path(__file__).resolve().parent / "prompts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts_shortcut.csv"
GT_OUT = OUT_DIR / "ground_truth_shortcut.csv"

MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """\
You rewrite multi-hop factual questions into simpler "shortcut" variants.

Given an original question and one specific intermediate entity that was NOT found
in the model's attribution graph, rewrite the question by substituting that entity
directly — removing only the indirect phrasing that leads to it. Keep any later
steps in the chain that WERE found.

Rules:
- Replace only the indirect reference that leads to the missed entity.
- Preserve any remaining chain steps (especially found concepts) in the question.
- The rewritten question must have the same correct final answer.
- Return ONLY valid JSON, no other text.

---
Example A — substituting the first missed concept, keeping later chain intact:

Input:
  original: "What continent is the country in where the director of the performer of Back in the U.S.S.R. is a citizen?"
  chain: Back in the U.S.S.R. → The Beatles → Brian Epstein → United Kingdom → Europe
  correct_answer: Europe
  found_concepts: United Kingdom
  missed_concept: "The Beatles"

Output:
  {{"rewritten_question": "What continent is the country in where the director of The Beatles is a citizen?", "substitution_made": "replaced 'the performer of Back in the U.S.S.R.' with 'The Beatles', kept 'director of ___'"}}

---
Example B — substituting a later missed concept, skipping further into the chain:

Input:
  original: "What continent is the country in where the director of the performer of Back in the U.S.S.R. is a citizen?"
  chain: Back in the U.S.S.R. → The Beatles → Brian Epstein → United Kingdom → Europe
  correct_answer: Europe
  found_concepts: United Kingdom
  missed_concept: "Brian Epstein"

Output:
  {{"rewritten_question": "What continent is the country in where Brian Epstein is a citizen?", "substitution_made": "replaced 'the director of the performer of Back in the U.S.S.R.' with 'Brian Epstein'"}}

---
Example C — one hop found, substitute the single missing one:

Input:
  original: "What city is home to the headquarters of the university where the entity known as President of Albania was educated?"
  chain: President of Albania → Ilir Meta → University of Tirana → Tirana
  correct_answer: Tirana
  found_concepts: University of Tirana
  missed_concept: "Ilir Meta"

Output:
  {{"rewritten_question": "What city is home to the headquarters of the university where Ilir Meta was educated?", "substitution_made": "replaced 'the entity known as President of Albania' with 'Ilir Meta'"}}
"""

USER_TEMPLATE = """\
original: "{original_prompt}"
chain: {chain}
correct_answer: {correct_answer}
found_concepts: {found_concepts}
missed_concept: "{missed_concept}"
"""


def call_gpt(client: OpenAI, original_prompt: str, chain: str,
             correct_answer: str, missed_concept: str,
             found_concepts: list[str]) -> dict | None:
    user_msg = USER_TEMPLATE.format(
        original_prompt=original_prompt,
        chain=chain,
        correct_answer=correct_answer,
        missed_concept=missed_concept,
        found_concepts=", ".join(found_concepts) if found_concepts else "none",
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        # strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group(0))
            print(f"    GPT raw response: {text!r}", file=sys.stderr)
            raise
    except Exception as e:
        print(f"    GPT error: {e}", file=sys.stderr)
        return None


def wrap_prompt(question: str) -> str:
    q = question.strip().rstrip("?").strip()
    return f"Human: {q}? Answer in one word.\nAssistant:"


def _parse_list(val: str) -> list[str]:
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def load_shortcut_candidates(hop_csv: Path) -> list[dict]:
    candidates = []
    with open(hop_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                num_hops = int(row.get("num_hops") or 0)
                model_correct = row.get("model_correct", "").lower() == "true"
                n_concepts = int(row.get("n_intermediate_concepts") or 0)
                n_found = int(row.get("n_hops_found") or 0)
            except (ValueError, TypeError):
                continue
            if model_correct and num_hops >= 3 and n_concepts > 0 and n_found < n_concepts:
                row["_concepts_missed"] = _parse_list(row.get("concepts_missed", "[]"))
                row["_concepts_found"] = _parse_list(row.get("concepts_found", "[]"))
                candidates.append(row)
    return candidates


def load_ground_truth(gt_csv: Path) -> dict[str, dict]:
    by_slug: dict[str, dict] = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = row["slug"].strip()
            by_slug[slug] = row
    return by_slug


def main() -> None:
    for path in [HOP_CSV, GROUND_TRUTH_CSV]:
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            sys.exit(1)

    client = OpenAI()

    print("Loading shortcut candidates from hop_analysis.csv ...")
    candidates = load_shortcut_candidates(HOP_CSV)
    print(f"  {len(candidates)} candidates found\n")

    print("Loading original prompts from ground_truth_mquake.csv ...")
    gt_by_slug = load_ground_truth(GROUND_TRUTH_CSV)

    prompt_rows: list[dict] = []
    gt_rows: list[dict] = []
    total_variants = 0

    for cand in candidates:
        slug = cand["slug"]
        gt = gt_by_slug.get(slug)
        if gt is None:
            print(f"  SKIP {slug} — not found in ground truth CSV")
            continue

        original_prompt = gt["prompt"].strip()
        chain = cand.get("notes", "").strip()
        correct_answer = cand["correct_answer"].strip()
        missed = cand["_concepts_missed"]
        found = cand["_concepts_found"]
        num_hops = cand["num_hops"]

        print(f"[{slug}] {num_hops}-hop | found: {found} | missed: {missed}")
        print(f"  Chain: {chain}")

        for i, missed_concept in enumerate(missed, start=1):
            print(f"  → Generating shortcut for missed concept: '{missed_concept}' ...")
            result = call_gpt(client, original_prompt, chain, correct_answer, missed_concept, found)
            if result is None:
                print(f"    SKIP (GPT failed)")
                continue

            rewritten = result.get("rewritten_question", "").strip()
            substitution = result.get("substitution_made", "")
            if not rewritten:
                print(f"    SKIP (empty rewrite)")
                continue

            sc_slug = f"{slug}-sc{i}"
            wrapped = wrap_prompt(rewritten)

            print(f"    {sc_slug}: {rewritten}")
            print(f"    ({substitution})")

            prompt_rows.append({
                "slug": sc_slug,
                "prompt": wrapped,
                "transcoder_set": "gemma",
            })
            gt_rows.append({
                "slug": sc_slug,
                "prompt": rewritten,
                "intermediate_concept": missed_concept,
                "correct_answer": correct_answer,
                "hop_type": f"shortcut (from {slug})",
                "num_hops": 1,
                "notes": f"Shortcut: '{missed_concept}' substituted directly | original chain: {chain}",
            })
            total_variants += 1

        print()

    if not prompt_rows:
        print("No shortcut prompts generated.", file=sys.stderr)
        sys.exit(1)

    # Write prompts CSV
    with open(PROMPTS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "prompt", "transcoder_set"])
        w.writeheader()
        w.writerows(prompt_rows)

    # Write ground truth CSV
    gt_fields = ["slug", "prompt", "intermediate_concept", "correct_answer",
                 "hop_type", "num_hops", "notes"]
    with open(GT_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_fields)
        w.writeheader()
        w.writerows(gt_rows)

    print("=" * 60)
    print(f"  Generated {total_variants} shortcut prompt(s) from {len(candidates)} candidate(s)")
    print(f"  Prompts CSV     -> {PROMPTS_OUT}")
    print(f"  Ground truth    -> {GT_OUT}")
    print()
    print("NEXT STEPS — run these manually:")
    print()
    print("  1. Attribute:")
    print(f"     circuit-tracer attribute-batch \\")
    print(f"       --csv {PROMPTS_OUT} \\")
    print(f"       --graph_file_dir ./test_graphs")
    print()
    print("  2. Group and push to viewer (from custom_automation/):")
    print(f"     GROUPING_VARIANTS=a2 ./batch_reproduce.sh \\")
    print(f"       {PROMPTS_OUT.relative_to(PACKAGE_DIR)}")
    print()
    print("  3. Analyze hops (from custom_automation/):")
    print(f"     python analysis/analyze_hops.py \\")
    print(f"       --ground_truth {GT_OUT} \\")
    print(f"       --variants a2")
    print("=" * 60)


if __name__ == "__main__":
    main()
