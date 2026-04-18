"""
generate_shortcut_prompts.py — Test whether the model uses shortcuts on missed hops.

For each case where:
  - The model answered correctly (predicts D)
  - At least one intermediate concept was NOT found in the attribution graph

We generate a "skip" variant: rewrite the question to remove that missed
intermediate from the chain, still asking for the same final answer D.

Rationale:
  If a concept was found in the attribution graph it is almost certainly
  load-bearing — the graph shows the model used it.  The interesting test
  is for concepts that were NOT found: the model reached D without going
  through that intermediate, which suggests a shortcut exists.

  We confirm by generating a simpler question that explicitly skips the
  missed hop.  If the model still gets D → shortcut confirmed.
  If the model gets D wrong → the intermediate was implicitly needed.

General pattern:
  Chain A → B → C → D
  B not found in graph → skip B → prompt becomes A → C → D
  C not found in graph → skip C → prompt becomes A → B → D
  B and C both not found → skip both → prompt becomes A → D

Example:
  Chain:   A → John Lennon → Yoko Ono
  John Lennon NOT found in graph, model still predicted Yoko Ono.
  Skip:    "The spouse of Imagine is"  ← removes the John Lennon hop
  If still correct → confirmed shortcut (A → Yoko Ono directly).

Only missed concepts get a skip variant. If a concept WAS found in the
attribution graph it is considered load-bearing and is not tested.

Reads:
  analysis/results/hop_analysis.csv       — correctness + missed concepts per case
  prompts/ground_truth_mquake.csv         — original prompts and full hop chains

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

MODEL = "gpt-5.4"

# ---------------------------------------------------------------------------
# GPT prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You rewrite multi-hop factual questions to skip ALL listed missed intermediate steps.

Given an original question, its reasoning chain, found concepts (load-bearing — keep
them implicit in the rewrite), and missed concepts (skip all of them at once),
produce ONE rewritten question that removes every missed hop while keeping the same
final answer.

The skip logic:
  Chain A → B → C → D, B not found → prompt asks A → C → D
  Chain A → B → C → D, C not found → prompt asks A → B → D
  Chain A → B → C → D, B and C both not found → prompt asks A → D

Critical rules:
- Do NOT name found concepts explicitly if doing so makes the answer trivially obvious.
  Keep found concepts implicit via a relational phrase ("the country associated with X").
- Remove ALL phrasing that resolves the skipped concepts.
- The final answer must remain correct and non-obvious from the question alone.
- The rewrite must be natural English.
- Return ONLY valid JSON with keys "rewritten_question" and "skip_description".

---
Example A — 3-hop, no found concepts, skip all intermediates:

Input:
  original: "What faith is linked to the child of the individual who established the Sasanian Empire?"
  chain: Ardashir I → Shapur I → Zoroastrianism
  correct_answer: Zoroastrianism
  found_concepts: []
  skip_concepts: ["Ardashir I", "Shapur I"]

Output:
  {{"rewritten_question": "What faith is linked to the Sasanian Empire?", "skip_description": "removed founder+child hops; no found concepts so Sasanian Empire (from original question) becomes the direct anchor"}}

---
Example B — 4-hop, skip Beatles+Epstein, keep UK implicit (answer=London):

Input:
  original: "What city is the capital of the country of citizenship of the director of the performer of Nowhere Man?"
  chain: Nowhere Man → The Beatles → Brian Epstein → United Kingdom → London
  correct_answer: London
  found_concepts: ["United Kingdom"]
  skip_concepts: ["The Beatles", "Brian Epstein"]

Output:
  {{"rewritten_question": "What city is the capital of the country Nowhere Man is associated with?", "skip_description": "removed performer+director hops; UK kept implicit so London isn't trivially obvious"}}

---
Example C — 4-hop, skip Junkers+Hugo Junkers, keep Germany implicit (answer=Europe):

Input:
  original: "What continent contains the country of citizenship of the founder of the developer of Ju 88?"
  chain: Ju 88 → Junkers → Hugo Junkers → Germany → Europe
  correct_answer: Europe
  found_concepts: ["Germany"]
  skip_concepts: ["Junkers", "Hugo Junkers"]

Output:
  {{"rewritten_question": "What continent is the country that developed Ju 88 in?", "skip_description": "removed developer+founder hops; Germany kept implicit"}}
"""

USER_TEMPLATE = """\
original: "{original_prompt}"
chain: {chain}
correct_answer: {correct_answer}
found_concepts: {found_concepts}
skip_concepts: {skip_concepts}
"""


def call_gpt(client: OpenAI, original_prompt: str, chain: str,
             correct_answer: str, found_concepts: list[str],
             skip_concepts: list[str]) -> dict | None:
    user_msg = USER_TEMPLATE.format(
        original_prompt=original_prompt,
        chain=chain,
        correct_answer=correct_answer,
        found_concepts=json.dumps(found_concepts),
        skip_concepts=json.dumps(skip_concepts),
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
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


def load_candidates(hop_csv: Path) -> list[dict]:
    """
    Select cases where:
      - num_hops >= 2
      - model is correct
      - at least one intermediate concept was NOT found in the attribution graph

    Found concepts are considered load-bearing (the graph shows the model used
    them).  Only missed concepts are worth testing — these are intermediates the
    model skipped over, suggesting a direct shortcut to D.
    """
    candidates = []
    with open(hop_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                model_correct = row.get("model_correct", "").lower() == "true"
                n_found = int(row.get("n_hops_found") or 0)
                n_intermediate = int(row.get("n_intermediate_concepts") or 0)
                num_hops = int(row.get("num_hops") or 0)
            except (ValueError, TypeError):
                continue
            n_missed = n_intermediate - n_found
            if model_correct and n_missed > 0 and num_hops >= 2:
                row["_concepts_missed"] = _parse_list(row.get("concepts_missed", "[]"))
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

    print("Loading candidates from hop_analysis.csv ...")
    candidates = load_candidates(HOP_CSV)
    print(f"  {len(candidates)} candidates (3-hop+, correct, at least one intermediate NOT found)\n")

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
        found = _parse_list(cand.get("concepts_found", "[]"))
        num_hops = cand.get("num_hops", "?")

        print(f"[{slug}] {num_hops}-hop | missed: {missed} | found: {found}")
        print(f"  Chain: {chain}")
        print(f"  → Generating skip for all missed concepts ...")

        result = call_gpt(client, original_prompt, chain, correct_answer, found, missed)
        if result is None:
            print(f"    SKIP (GPT failed)")
            print()
            continue

        rewritten = result.get("rewritten_question", "").strip()
        description = result.get("skip_description", "")
        if not rewritten:
            print(f"    SKIP (empty rewrite)")
            print()
            continue

        skip_slug = f"{slug}-skip"
        wrapped = wrap_prompt(rewritten)

        print(f"    {skip_slug}: {rewritten}")
        print(f"    ({description})")

        prompt_rows.append({
            "slug": skip_slug,
            "prompt": wrapped,
            "transcoder_set": "gemma",
        })
        gt_rows.append({
            "slug": skip_slug,
            "prompt": rewritten,
            "intermediate_concept": ", ".join(missed),
            "correct_answer": correct_answer,
            "hop_type": f"skip-test (from {slug})",
            "num_hops": max(int(num_hops) - len(missed), 1) if str(num_hops).isdigit() else 1,
            "notes": (
                f"Skipped {missed} (NOT found in graph) | "
                f"kept {found} (found, load-bearing) | "
                f"original chain: {chain} | "
                f"if still correct → shortcut confirmed | "
                f"if wrong → missed concepts were implicitly needed"
            ),
        })
        total_variants += 1

        print()

    if not prompt_rows:
        print("No skip prompts generated.", file=sys.stderr)
        sys.exit(1)

    with open(PROMPTS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "prompt", "transcoder_set"])
        w.writeheader()
        w.writerows(prompt_rows)

    gt_fields = ["slug", "prompt", "intermediate_concept", "correct_answer",
                 "hop_type", "num_hops", "notes"]
    with open(GT_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_fields)
        w.writeheader()
        w.writerows(gt_rows)

    print("=" * 60)
    print(f"  Generated {total_variants} skip variant(s) from {len(candidates)} candidate(s)")
    print(f"  Prompts CSV  -> {PROMPTS_OUT}")
    print(f"  Ground truth -> {GT_OUT}")
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
    print("  3. Analyze (from custom_automation/):")
    print(f"     python analysis/analyze_hops.py \\")
    print(f"       --ground_truth {GT_OUT} \\")
    print(f"       --variants a2")
    print("=" * 60)


if __name__ == "__main__":
    main()
