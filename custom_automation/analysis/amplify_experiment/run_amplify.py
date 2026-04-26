"""
run_amplify.py — Amplify answer-token features to correct wrong predictions.

Candidates: rows in hop_analysis.csv where
  - model_correct = False
  - correct answer token appears in all_supernodes (the model already has an
    internal representation of the answer, just doesn't emit it)

For each candidate:
  1. Load the graph and feature_groups artifact.
  2. Find transcoder features whose supernode group contains the correct answer.
  3. Run baseline forward pass, then amplify those features by AMPLIFY_FACTOR.
  4. Check whether the model now outputs the correct answer.

Writes:
  analysis/amplify_experiment/amplify_results.csv
  analysis/amplify_experiment/amplify_results.md

Usage:
    python analysis/amplify_experiment/run_amplify.py
    python analysis/amplify_experiment/run_amplify.py --variants a2
    python analysis/amplify_experiment/run_amplify.py --dry_run
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
HOP_CSV = PACKAGE_DIR / "analysis" / "results" / "hop_analysis.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "amplify_results.csv"
OUT_MD = OUT_DIR / "amplify_results.md"

DESCRIPTION_VARIANT = "v2"
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"

# Multiply baseline activation by this factor. 5x is a strong but not extreme boost.
AMPLIFY_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def _parse_list(val: str) -> list[str]:
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def load_candidates(hop_csv: Path, variants: list[str]) -> list[dict]:
    """
    Filter hop_analysis.csv for rows where:
      - correct answer token appears (case-insensitive) in all_supernodes
      - variant is in the requested list
    Includes both correct and wrong baseline predictions.
    """
    candidates = []
    with open(hop_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant", "").strip() not in variants:
                continue
            correct_answer = row.get("correct_answer", "").strip().lower()
            if not correct_answer:
                continue
            supernodes = _parse_list(row.get("all_supernodes", "[]"))
            supernodes_lower = [s.lower() for s in supernodes
                                if not s.startswith("Emb:") and not s.startswith("Output:")]
            if any(correct_answer in s for s in supernodes_lower):
                row["_supernodes"] = supernodes
                candidates.append(row)
    return candidates


# ---------------------------------------------------------------------------
# Graph / feature loading (mirrors run_interventions.py)
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_feature_groups(slug: str, variant: str) -> dict[str, str] | None:
    path = (ARTIFACTS_DIR / f"{slug}-{DESCRIPTION_VARIANT}-{variant}"
            / f"feature_groups_{DESCRIPTION_VARIANT}_{variant}.json")
    if not path.exists():
        path = ARTIFACTS_DIR / slug / f"feature_groups_{DESCRIPTION_VARIANT}_{variant}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_answer_features(graph: dict, correct_answer: str,
                         feature_groups: dict[str, str] | None) -> list[dict]:
    """
    Find transcoder features whose supernode group contains the correct answer token.
    Returns empty list if no feature_groups or no group matches.
    """
    if not feature_groups:
        return []

    node_lookup: dict[str, dict] = {str(n.get("node_id", "")): n for n in graph.get("nodes", [])}
    answer_lower = correct_answer.lower()

    matching_groups = {
        g for g in set(feature_groups.values())
        if g != "Ungrouped"
        and answer_lower in g.lower()
        and not g.startswith("Emb:")
        and not g.startswith("Output:")
    }
    if not matching_groups:
        return []

    print(f"    '{correct_answer}' → {sorted(matching_groups)}")
    results = []
    for fid, gname in feature_groups.items():
        if gname not in matching_groups:
            continue
        node = node_lookup.get(fid)
        if node is None or node.get("feature_type") != "cross layer transcoder":
            continue
        parts = fid.split("_")
        if len(parts) < 3:
            continue
        try:
            layer, feature_idx, ctx_idx = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        results.append({
            "node_id": fid,
            "layer": layer,
            "feature_idx": feature_idx,
            "ctx_idx": ctx_idx,
            "clerp": node.get("clerp", ""),
            "influence": float(node.get("influence", 0.0)),
        })
    return results


# ---------------------------------------------------------------------------
# Steering helpers (mirrors run_interventions.py)
# ---------------------------------------------------------------------------

def build_amplify_interventions(answer_features: list[dict],
                                activation_cache: torch.Tensor) -> list[tuple]:
    tuples = []
    for f in answer_features:
        layer, ctx_idx, feature_idx = f["layer"], f["ctx_idx"], f["feature_idx"]
        if (layer >= activation_cache.shape[0] or ctx_idx >= activation_cache.shape[1]
                or feature_idx >= activation_cache.shape[2]):
            continue
        default_act = float(activation_cache[layer, ctx_idx, feature_idx].item())
        tuples.append((layer, ctx_idx, feature_idx, AMPLIFY_FACTOR * default_act))
    return tuples


def next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.squeeze()[-1]


def correct_token_id(tokenizer, correct_answer: str) -> int | None:
    for candidate in (f" {correct_answer}", correct_answer):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


def token_rank(logits_last: torch.Tensor, token_id: int) -> int:
    sorted_ids = torch.argsort(logits_last, descending=True)
    matches = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    return int(matches[0].item()) + 1 if len(matches) else -1


def top_k_tokens(logits_last: torch.Tensor, tokenizer, k: int = 5) -> list[tuple[str, float]]:
    probs = torch.softmax(logits_last, dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i].item()]), float(topk.values[i].item()))
            for i in range(k)]


def fmt_top5(top5: list[tuple[str, float]]) -> str:
    return ", ".join(f"{tok.strip()}({prob:.1%})" for tok, prob in top5)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(model, prompt: str, correct_answer: str,
               answer_features: list[dict]) -> dict:
    tok_id = correct_token_id(model.tokenizer, correct_answer)

    with torch.inference_mode():
        baseline_logits, activation_cache = model.feature_intervention(
            prompt, [], return_activations=True
        )
        amp_tuples = build_amplify_interventions(answer_features, activation_cache)
        amplified_logits, _ = model.feature_intervention(
            prompt, amp_tuples, return_activations=False
        )

    base_last = next_token_logits(baseline_logits)
    amp_last = next_token_logits(amplified_logits)

    base_top5 = top_k_tokens(base_last, model.tokenizer)
    amp_top5 = top_k_tokens(amp_last, model.tokenizer)

    base_rank = token_rank(base_last, tok_id) if tok_id is not None else -1
    amp_rank = token_rank(amp_last, tok_id) if tok_id is not None else -1

    base_probs = torch.softmax(base_last, dim=-1)
    amp_probs = torch.softmax(amp_last, dim=-1)
    base_prob = float(base_probs[tok_id].item()) if tok_id is not None else 0.0
    amp_prob = float(amp_probs[tok_id].item()) if tok_id is not None else 0.0

    rank_change = (amp_rank - base_rank) if base_rank > 0 and amp_rank > 0 else None
    flipped_correct = amp_top5[0][0].strip().lower() == correct_answer.lower() if amp_top5 else False

    return {
        "n_features_amplified": len(amp_tuples),
        "baseline_predicted": base_top5[0][0].strip() if base_top5 else "?",
        "amplified_predicted": amp_top5[0][0].strip() if amp_top5 else "?",
        "flipped_correct": flipped_correct,
        "baseline_rank": base_rank if base_rank > 0 else None,
        "amplified_rank": amp_rank if amp_rank > 0 else None,
        "rank_change": rank_change,
        "baseline_prob": round(base_prob, 6),
        "amplified_prob": round(amp_prob, 6),
        "prob_gain": round(amp_prob - base_prob, 6),
        "baseline_top5": fmt_top5(base_top5),
        "amplified_top5": fmt_top5(amp_top5),
        "skipped": None,
    }


EMPTY_METRICS: dict = {k: None for k in [
    "n_features_amplified", "baseline_predicted", "amplified_predicted",
    "flipped_correct", "baseline_rank", "amplified_rank", "rank_change",
    "baseline_prob", "amplified_prob", "prob_gain",
    "baseline_top5", "amplified_top5", "skipped",
]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(variants: list[str], dry_run: bool = False) -> None:
    if not HOP_CSV.exists():
        print(f"ERROR: not found: {HOP_CSV}", file=sys.stderr)
        sys.exit(1)

    print("Filtering candidates from hop_analysis.csv ...")
    candidates = load_candidates(HOP_CSV, variants)
    print(f"  {len(candidates)} candidates (wrong + answer token in supernodes)\n")

    if not candidates:
        print("No candidates found.", file=sys.stderr)
        sys.exit(1)

    model = None
    if not dry_run:
        print(f"Loading {MODEL_NAME} ...")
        from circuit_tracer import ReplacementModel
        model = ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODER_SET, dtype=torch.bfloat16)
        print("Model loaded.\n")

    results: list[dict] = []

    for cand in candidates:
        slug = cand["slug"].strip()
        variant = cand["variant"].strip()
        correct_answer = cand["correct_answer"].strip()
        predicted = cand.get("predicted", "?").strip()

        graph_slug = f"{slug}-{DESCRIPTION_VARIANT}-{variant}"
        graph_path = TEST_GRAPHS_DIR / f"{graph_slug}.json"
        if not graph_path.exists():
            graph_slug = slug
            graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
        graph = load_graph(graph_path)

        base_row = {
            "slug": slug,
            "variant": variant,
            "model_correct": cand.get("model_correct", "").strip().lower() == "true",
            "correct_answer": correct_answer,
            "predicted": predicted,
            "prompt": "",
        }

        if graph is None:
            print(f"  SKIP {graph_slug} — graph not found")
            results.append({**base_row, **EMPTY_METRICS, "skipped": "graph_missing"})
            continue

        # Pull prompt from the graph metadata — hop_analysis.csv has no prompt column
        prompt = graph.get("metadata", {}).get("prompt", "").replace("<bos>", "").strip()
        base_row["prompt"] = prompt[:80]
        if not prompt:
            print(f"  SKIP {graph_slug} — no prompt in graph metadata")
            results.append({**base_row, **EMPTY_METRICS, "skipped": "no_prompt"})
            continue

        feature_groups = load_feature_groups(slug, variant)
        answer_features = find_answer_features(graph, correct_answer, feature_groups)
        print(f"  {graph_slug}: {len(answer_features)} answer feature(s) for '{correct_answer}'")

        if dry_run:
            results.append({**base_row, **EMPTY_METRICS,
                             "n_features_amplified": len(answer_features), "skipped": "dry_run"})
            continue

        if not answer_features:
            results.append({**base_row, **EMPTY_METRICS, "skipped": "no_answer_features"})
            continue

        metrics = run_single(model, prompt, correct_answer, answer_features)
        results.append({**base_row, **metrics})
        status = "✓ FLIPPED" if metrics["flipped_correct"] else "✗ still wrong"
        print(f"    {status} | {metrics['baseline_predicted']} → {metrics['amplified_predicted']} "
              f"| rank {metrics['baseline_rank']} → {metrics['amplified_rank']}")

    if dry_run:
        print("\nDry-run complete.")
        for r in results:
            print(f"  {r['slug']} ({r['variant']}): {r['n_features_amplified']} answer feature(s)")
        return

    # Write CSV
    fieldnames = list(results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV  -> {OUT_CSV}")

    # Write markdown
    valid = [r for r in results if r.get("baseline_rank") is not None]
    n = len(valid)
    n_flipped = sum(1 for r in valid if r.get("flipped_correct"))

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Amplify Experiment Results\n\n")
        f.write(f"Variants: {', '.join(variants)}  \n")
        f.write(f"Candidates: {len(candidates)}  \n")
        f.write(f"Amplify factor: {AMPLIFY_FACTOR}×  \n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Candidates with answer features found | {n} / {len(candidates)} |\n")
        f.write(f"| Flipped to correct after amplification | {n_flipped} / {n} "
                f"({n_flipped/n:.1%}) |\n" if n else "| Flipped to correct | — |\n")
        if n:
            mean_gain = sum(r["prob_gain"] for r in valid if r.get("prob_gain") is not None) / n
            f.write(f"| Mean correct-answer prob gain | {mean_gain:+.4f} |\n")
        f.write("\n")

        f.write("## Per-Prompt Results\n\n")
        f.write("| Slug | Correct | Predicted | # Amp | Flipped? | Rank Δ | "
                "Prob gain | Baseline top-5 | Amplified top-5 |\n")
        f.write("|------|---------|-----------|-------|----------|--------|"
                "-----------|----------------|----------------|\n")
        for r in valid:
            rc = r.get("rank_change")
            delta = f"{rc:+d}" if rc is not None else "—"
            flipped = "✓" if r.get("flipped_correct") else "✗"
            gain = r.get("prob_gain", 0.0)
            f.write(
                f"| {r['slug']} | {r['correct_answer']} | {r['predicted']} "
                f"| {r['n_features_amplified']} | {flipped} | {delta} "
                f"| {gain:+.4f} | {r.get('baseline_top5','—')} | {r.get('amplified_top5','—')} |\n"
            )
        f.write("\n")

        skipped = [r for r in results if r.get("skipped")]
        if skipped:
            f.write("## Skipped\n\n")
            for r in skipped:
                f.write(f"- {r['slug']} ({r['variant']}): {r['skipped']}\n")

    print(f"Report -> {OUT_MD}")

    print()
    print("=" * 55)
    print(f"  Candidates:              {len(candidates)}")
    print(f"  With answer features:    {n}")
    print(f"  Flipped correct:         {n_flipped} / {n}")
    if n:
        print(f"  Mean prob gain:          {mean_gain:+.4f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="a2")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(variants, dry_run=args.dry_run)
