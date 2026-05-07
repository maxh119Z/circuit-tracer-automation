"""
run_amplify_middlehop.py — Amplify intermediate-hop features and check if p(correct answer) rises.

For each candidate:
  1. Load the graph and feature_groups artifact.
  2. Find transcoder features whose supernode group matches the intermediate concept
     (e.g. "Texas" for a capitals question about a Texas city).
  3. Run baseline forward pass, then amplify those features by AMPLIFY_FACTOR.
  4. Check whether p(correct answer) increases and/or model flips to correct.

Writes:
  analysis/amplify_experiment/amplify_middlehop_results.csv
  analysis/amplify_experiment/amplify_middlehop_results.md

Usage:
    python analysis/amplify_experiment/run_amplify_middlehop.py
    python analysis/amplify_experiment/run_amplify_middlehop.py --variants a2
    python analysis/amplify_experiment/run_amplify_middlehop.py --dry_run
"""

from __future__ import annotations

import argparse
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
OUT_CSV = OUT_DIR / "amplify_middlehop_results.csv"
OUT_MD = OUT_DIR / "amplify_middlehop_results.md"

DESCRIPTION_VARIANT = "v2"
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"

# 2x — lighter than answer-token amplification since middle features are more
# broadly connected and a strong boost may disrupt unrelated computation.
AMPLIFY_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def load_candidates(hop_csv: Path, variants: list[str]) -> list[dict]:
    """Filter hop_analysis.csv for all cases (correct and wrong) where the intermediate hop was found,
    restricted to exactly 1 intermediate hop (2-hop chains) to avoid ambiguity about which hop to amplify."""
    candidates = []
    with open(hop_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant", "").strip() not in variants:
                continue
            if row.get("hop_found", "").lower() != "true":
                continue
            # Filter to 2-hop only; treat empty num_hops (e.g. Capitals) as 2-hop
            try:
                num_hops = int(row.get("num_hops") or 2)
                if num_hops != 2:
                    continue
            except (ValueError, TypeError):
                continue
            candidates.append(row)
    return candidates


# ---------------------------------------------------------------------------
# Graph / feature loading
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


def find_middlehop_features(graph: dict, intermediate_concept: str,
                             feature_groups: dict[str, str] | None) -> list[dict]:
    """
    Find transcoder features whose supernode group name shares significant words
    with the intermediate concept (e.g. 'Texas' matches group 'Texas').
    Uses word overlap with length threshold >5 chars to avoid false positives.
    """
    if not feature_groups:
        return []

    node_lookup: dict[str, dict] = {str(n.get("node_id", "")): n for n in graph.get("nodes", [])}
    concept_words = {w for w in intermediate_concept.lower().split() if len(w) > 5}
    if not concept_words:
        # Short concepts (e.g. "Iowa"): fall back to full substring match
        concept_words = {intermediate_concept.lower()}

    _SUPPRESS_PREFIXES = ("suppress ", "anti-", "anti ", "demote ", "inhibit ", "avoid ", "repress ")

    def _matches(g: str) -> bool:
        if g in ("Ungrouped",) or g.startswith("Output:") or g.startswith("Emb:"):
            return False
        # Never match suppression groups — they have opposite causal role
        if any(g.lower().startswith(p) for p in _SUPPRESS_PREFIXES):
            return False
        g_lower = g.lower()
        g_words = {w for w in g_lower.split() if len(w) > 5}
        if not g_words:
            g_words = {g_lower}
        if concept_words & g_words:
            return True
        # Prefix fallback: first 4 chars match (handles Thai/Thailand, Colombian/Colombia)
        all_g_words = {w for w in g_lower.split() if len(w) >= 4}
        for cw in concept_words:
            if len(cw) >= 4:
                for gw in all_g_words:
                    if cw[:4] == gw[:4]:
                        return True
        return False

    matching_groups = {g for g in set(feature_groups.values()) if _matches(g)}
    if not matching_groups:
        return []

    print(f"    '{intermediate_concept}' → {sorted(matching_groups)}")
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
            "influence": float(node.get("influence", 0.0)),
        })
    return results


# ---------------------------------------------------------------------------
# Steering helpers
# ---------------------------------------------------------------------------

def build_amplify_interventions(features: list[dict],
                                activation_cache: torch.Tensor) -> list[tuple]:
    tuples = []
    for f in features:
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
               intermediate_concept: str, middlehop_features: list[dict]) -> dict:
    # p(correct) is measured on the first subword token of the correct answer.
    # For multi-token answers this is an upper bound on the true sequence probability.
    tok_id = correct_token_id(model.tokenizer, correct_answer)
    n_layers = model.cfg.n_layers

    with torch.inference_mode():
        baseline_logits, activation_cache = model.feature_intervention(
            prompt, [], return_activations=True
        )
        amp_tuples = build_amplify_interventions(middlehop_features, activation_cache)
        amplified_logits, _ = model.feature_intervention(
            prompt, amp_tuples, constrained_layers=range(n_layers), return_activations=False
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
    flipped_to_middlehop = amp_top5[0][0].strip().lower() == intermediate_concept.lower() if amp_top5 else False

    return {
        "n_features_amplified": len(amp_tuples),
        "baseline_predicted": base_top5[0][0].strip() if base_top5 else "?",
        "amplified_predicted": amp_top5[0][0].strip() if amp_top5 else "?",
        "flipped_correct": flipped_correct,
        "flipped_to_middlehop": flipped_to_middlehop,
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
    "flipped_correct", "flipped_to_middlehop", "baseline_rank", "amplified_rank", "rank_change",
    "baseline_prob", "amplified_prob", "prob_gain",
    "baseline_top5", "amplified_top5", "skipped",
]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(variants: list[str], dry_run: bool = False,
        hop_csv: Path | None = None,
        graphs_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        out_csv: Path | None = None,
        out_md: Path | None = None) -> None:
    global TEST_GRAPHS_DIR, ARTIFACTS_DIR, OUT_CSV, OUT_MD
    if graphs_dir is not None:
        TEST_GRAPHS_DIR = graphs_dir
    if artifacts_dir is not None:
        ARTIFACTS_DIR = artifacts_dir
    if out_csv is not None:
        OUT_CSV = out_csv
    if out_md is not None:
        OUT_MD = out_md
    hop_csv = hop_csv or HOP_CSV
    if not hop_csv.exists():
        print(f"ERROR: not found: {hop_csv}", file=sys.stderr)
        sys.exit(1)

    print("Filtering candidates from hop_analysis.csv ...")
    candidates = load_candidates(hop_csv, variants)
    print(f"  {len(candidates)} candidates (2-hop, intermediate hop found in graph)\n")

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
        intermediate_concept = cand.get("intermediate_concept", "").strip()
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
            "intermediate_concept": intermediate_concept,
            "correct_answer": correct_answer,
            "predicted": predicted,
            "prompt": "",
        }

        if graph is None:
            print(f"  SKIP {graph_slug} — graph not found")
            results.append({**base_row, **EMPTY_METRICS, "skipped": "graph_missing"})
            continue

        prompt = graph.get("metadata", {}).get("prompt", "").replace("<bos>", "").strip()
        base_row["prompt"] = prompt[:80]
        if not prompt:
            print(f"  SKIP {graph_slug} — no prompt in graph metadata")
            results.append({**base_row, **EMPTY_METRICS, "skipped": "no_prompt"})
            continue

        feature_groups = load_feature_groups(slug, variant)
        middlehop_features = find_middlehop_features(graph, intermediate_concept, feature_groups)
        print(f"  {graph_slug}: {len(middlehop_features)} middlehop feature(s) for '{intermediate_concept}'")

        if dry_run:
            results.append({**base_row, **EMPTY_METRICS,
                             "n_features_amplified": len(middlehop_features), "skipped": "dry_run"})
            continue

        if not middlehop_features:
            results.append({**base_row, **EMPTY_METRICS, "skipped": "no_middlehop_features"})
            continue

        metrics = run_single(model, prompt, correct_answer, intermediate_concept, middlehop_features)
        results.append({**base_row, **metrics})
        status = "✓ FLIPPED" if metrics["flipped_correct"] else ("→ middlehop" if metrics["flipped_to_middlehop"] else "✗ still wrong")
        print(f"    {status} | {metrics['baseline_predicted']} → {metrics['amplified_predicted']} "
              f"| rank {metrics['baseline_rank']} → {metrics['amplified_rank']} "
              f"| prob gain: {metrics['prob_gain']:+.4f}")

    if dry_run:
        print("\nDry-run complete.")
        for r in results:
            print(f"  {r['slug']} ({r['variant']}): {r['n_features_amplified']} middlehop feature(s)")
        return

    fieldnames = list(results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV  -> {OUT_CSV}")

    valid = [r for r in results if r.get("baseline_rank") is not None]
    n = len(valid)

    was_correct   = [r for r in valid if r.get("model_correct")]
    was_wrong     = [r for r in valid if not r.get("model_correct")]

    # wrong baseline → top-1 after amplification is the correct answer
    n_rescued     = sum(1 for r in was_wrong if r.get("flipped_correct"))
    # correct baseline → top-1 after amplification is still the correct answer
    n_held        = sum(1 for r in was_correct if r.get("flipped_correct"))
    # correct baseline → top-1 after amplification is no longer correct (amplification hurt)
    n_broke       = sum(1 for r in was_correct if not r.get("flipped_correct"))
    n_middlehop   = sum(1 for r in valid if r.get("flipped_to_middlehop"))
    n_improved    = sum(1 for r in valid if (r.get("rank_change") or 0) < 0)

    mean_gain = (sum(r["prob_gain"] for r in valid if r.get("prob_gain") is not None) / n) if n else 0.0

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Amplify Middle-Hop Experiment Results\n\n")
        f.write(f"Variants: {', '.join(variants)}  \n")
        f.write(f"Candidates: {len(candidates)}  \n")
        f.write(f"Amplify factor: {AMPLIFY_FACTOR}×  \n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Candidates with middlehop features found | {n} / {len(candidates)} |\n")
        f.write(f"| Baseline correct | {len(was_correct)} / {n} |\n")
        f.write(f"| Baseline wrong | {len(was_wrong)} / {n} |\n")
        if n:
            f.write(f"| Wrong → correct after amplification | {n_rescued} / {len(was_wrong)} |\n" if was_wrong else "")
            f.write(f"| Correct → still correct after amplification | {n_held} / {len(was_correct)} |\n" if was_correct else "")
            f.write(f"| Correct → broken by amplification | {n_broke} / {len(was_correct)} |\n" if was_correct else "")
            f.write(f"| Flipped to middlehop intermediate (over-amplified) | {n_middlehop} / {n} |\n")
            f.write(f"| Correct-answer rank improved (any baseline) | {n_improved} / {n} ({n_improved/n:.1%}) |\n")
            f.write(f"| Mean correct-answer prob gain | {mean_gain:+.4f} |\n")
        f.write("\n")

        f.write("## Per-Prompt Results\n\n")
        f.write("| Slug | Baseline | Intermediate | Correct | Predicted | # Amp | Amplified top-1 correct? | Rank Δ | "
                "Prob gain | Baseline top-5 | Amplified top-5 |\n")
        f.write("|------|----------|-------------|---------|-----------|-------|--------------------------|--------|"
                "-----------|----------------|----------------|\n")
        for r in valid:
            rc = r.get("rank_change")
            delta = f"{rc:+d}" if rc is not None else "—"
            baseline_label = "correct" if r.get("model_correct") else "wrong"
            amp_correct = "yes" if r.get("flipped_correct") else "no"
            gain = r.get("prob_gain", 0.0)
            f.write(
                f"| {r['slug']} | {baseline_label} | {r['intermediate_concept']} | {r['correct_answer']} "
                f"| {r['predicted']} | {r['n_features_amplified']} | {amp_correct} | {delta} "
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
    print("=" * 60)
    print(f"  Candidates:                        {len(candidates)}")
    print(f"  With middlehop features:           {n}")
    if n:
        print(f"  Baseline correct:                  {len(was_correct)} / {n}")
        print(f"  Baseline wrong:                    {len(was_wrong)} / {n}")
        if was_wrong:
            print(f"  Wrong → correct (amplified):       {n_rescued} / {len(was_wrong)}")
        if was_correct:
            print(f"  Correct → still correct:           {n_held} / {len(was_correct)}")
            print(f"  Correct → broken by amplification: {n_broke} / {len(was_correct)}")
        print(f"  Over-amplified (→ middlehop):      {n_middlehop} / {n}")
        print(f"  Rank improved (any baseline):      {n_improved} / {n}")
        print(f"  Mean prob gain:                    {mean_gain:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="a2")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--hop_csv", type=Path, default=None,
                        help="Path to hop_analysis.csv (default: analysis/results/hop_analysis.csv)")
    parser.add_argument("--graphs_dir", type=Path, default=None,
                        help="Directory containing graph JSONs (default: test_graphs/)")
    parser.add_argument("--artifacts_dir", type=Path, default=None,
                        help="Directory containing feature_groups JSON files (default: artifacts/)")
    parser.add_argument("--out_csv", type=Path, default=None,
                        help="Output CSV path (default: amplify_middlehop_results.csv next to this script)")
    parser.add_argument("--out_md", type=Path, default=None,
                        help="Output Markdown path (default: amplify_middlehop_results.md next to this script)")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(variants, dry_run=args.dry_run, hop_csv=args.hop_csv, graphs_dir=args.graphs_dir,
        artifacts_dir=args.artifacts_dir, out_csv=args.out_csv, out_md=args.out_md)
