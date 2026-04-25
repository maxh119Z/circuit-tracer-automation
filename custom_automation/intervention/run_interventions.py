"""
run_interventions.py — Causal validation via negative multiplicative steering on intermediate-hop features.

For each (slug, grouping variant) in ground_truth.csv this script:
  1. Loads the processed attribution graph and the feature_groups artifact.
  2. Identifies transcoder features belonging to the supernode group matching the
     intermediate-hop concept (e.g. the "Texas" group for a Dallas prompt).
     Falls back to clerp string matching if no groups artifact is found.
  3. Runs a baseline forward pass (return_activations=True) to get each feature's
     default activation value.
  4. Steers those features negatively using scaling_factor * default_activation as
     the intervention value — matching the multiplicative approach from the paper
     and tutorial (NOT zero-ablation, which ignores baseline contributions).
     freeze_attention=True is kept (default) per the paper.
  5. Compares the correct-answer token's rank and probability before and after.
     A significant drop indicates those features causally support the prediction.

Methodology follows:
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  https://github.com/decoderesearch/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb

Outputs:
  - artifacts/intervention_results.csv — one row per (slug, variant)
  - artifacts/intervention_results.md  — human-readable report

Usage:
    python intervention/run_interventions.py
    python intervention/run_interventions.py --variants a0,a3
    python intervention/run_interventions.py --ground_truth ../ground_truth.csv
    python intervention/run_interventions.py --dry_run   # skip model loading
    python intervention/run_interventions.py \
  --ground_truth ../prompts/ground_truth_mquake.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths (mirrors analyze_hops.py)
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GROUND_TRUTH = REPO_ROOT / "prompts" / "ground_truth_capital.csv"
DESCRIPTION_VARIANT = "v2"
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_graph(graph_path: Path) -> dict | None:
    if not graph_path.exists():
        return None
    with open(graph_path, encoding="utf-8") as f:
        return json.load(f)


def load_feature_groups(slug: str, variant: str) -> dict[str, str] | None:
    """Load feature_groups artifact for a slug+variant. Returns {feature_id: group_name} or None."""
    # Multi-variant naming: artifacts/{slug}-v2-a2/feature_groups_v2_a2.json
    path = ARTIFACTS_DIR / f"{slug}-{DESCRIPTION_VARIANT}-{variant}" / f"feature_groups_{DESCRIPTION_VARIANT}_{variant}.json"
    if not path.exists():
        # Single-variant naming: artifacts/{slug}/feature_groups_v2_a2.json
        path = ARTIFACTS_DIR / slug / f"feature_groups_{DESCRIPTION_VARIANT}_{variant}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_hop_features(graph: dict, intermediate_concept: str, feature_groups: dict[str, str] | None) -> list[dict]:
    """
    Return cross-layer transcoder nodes whose supernode group name shares at least one
    meaningful word with the intermediate concept (bidirectional word overlap).
    Returns empty list if no feature_groups are available or no group matches.
    """
    if not feature_groups:
        return []

    node_lookup: dict[str, dict] = {str(n.get("node_id", "")): n for n in graph.get("nodes", [])}
    concept_words = {w for w in intermediate_concept.lower().split() if len(w) > 2}

    def _group_matches(g: str) -> bool:
        if g in ("Ungrouped",) or g.startswith("Output:") or g.startswith("Emb:"):
            return False
        g_lower = g.lower()
        g_words = {w for w in g_lower.split() if len(w) > 2}
        if concept_words & g_words:
            return True
        # Prefix fallback: first 4 chars match (handles Thai/Thailand, Colombian/Colombia)
        for cw in concept_words:
            if len(cw) >= 4:
                for gw in g_words:
                    if len(gw) >= 4 and cw[:4] == gw[:4]:
                        return True
        return False

    matching_groups = {g for g in set(feature_groups.values()) if _group_matches(g)}
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
            "clerp": node.get("clerp", ""),
            "influence": float(node.get("influence", 0.0)),
        })
    return results


# Scaling factor for negative multiplicative steering, per the paper and tutorial.
# -1 suppresses the feature to its negative (strong but not extreme).
# The tutorial uses -2 for even stronger suppression.
SCALING_FACTOR = -2.0


def build_scaled_interventions(hop_features: list[dict], activation_cache: torch.Tensor) -> list[tuple]:
    """
    Build intervention tuples using negative multiplicative steering:
        value = SCALING_FACTOR * default_activation

    This matches the paper's approach (steer negatively rather than zero-ablate),
    accounting for each feature's baseline contribution.
    activation_cache is indexed as activation_cache[layer][ctx_idx, feature_idx].
    """
    # activation_cache shape: [n_layers, seq_len, n_features]
    tuples = []
    for f in hop_features:
        layer, ctx_idx, feature_idx = f["layer"], f["ctx_idx"], f["feature_idx"]
        if layer >= activation_cache.shape[0] or ctx_idx >= activation_cache.shape[1]:
            continue
        default_act = float(activation_cache[layer, ctx_idx, feature_idx].item())
        value = SCALING_FACTOR * default_act
        tuples.append((layer, ctx_idx, feature_idx, value))
    return tuples


# ---------------------------------------------------------------------------
# Token / logit utilities
# ---------------------------------------------------------------------------


def next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Extract last-position logits from a model output tensor.

    feature_intervention() returns the raw model output; for both the
    transformerlens and nnsight backends this is [1, seq_len, vocab_size] or
    similar, so squeezing and indexing the last sequence position gives the
    next-token distribution.
    """
    return logits.squeeze()[-1]


def correct_token_id(tokenizer, correct_answer: str) -> int | None:
    """
    Find the token ID for the first subword of correct_answer.
    Tries with a leading space first (typical for non-initial tokens).
    """
    for candidate in (f" {correct_answer}", correct_answer):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


def token_rank(logits_last: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of token_id in the descending logit order."""
    sorted_ids = torch.argsort(logits_last, descending=True)
    matches = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return -1
    return int(matches[0].item()) + 1


def top_k_tokens(logits_last: torch.Tensor, tokenizer, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k (token_string, probability) pairs."""
    probs = torch.softmax(logits_last, dim=-1)
    topk = torch.topk(probs, k)
    return [
        (tokenizer.decode([topk.indices[i].item()]), float(topk.values[i].item()))
        for i in range(k)
    ]


# ---------------------------------------------------------------------------
# Single-prompt intervention
# ---------------------------------------------------------------------------


def run_single(
    model,
    prompt: str,
    correct_answer: str,
    hop_features: list[dict],
) -> dict:
    """
    Run a baseline forward pass then a negatively-steered intervention.

    Baseline pass uses return_activations=True to get each feature's default
    activation value. Intervention tuples are then built as
    (layer, ctx_idx, feature_idx, SCALING_FACTOR * default_activation),
    matching the multiplicative steering approach from the paper and tutorial.
    freeze_attention=True (default) is kept per the paper.
    """
    tok_id = correct_token_id(model.tokenizer, correct_answer)

    with torch.inference_mode():
        # Baseline: empty interventions, collect activations for scaling
        baseline_logits, activation_cache = model.feature_intervention(
            prompt, [], return_activations=True
        )
        scaled_tuples = build_scaled_interventions(hop_features, activation_cache)
        steered_logits, _ = model.feature_intervention(
            prompt, scaled_tuples, return_activations=False
        )

    base_last = next_token_logits(baseline_logits)
    steered_last = next_token_logits(steered_logits)

    base_probs = torch.softmax(base_last, dim=-1)
    steered_probs = torch.softmax(steered_last, dim=-1)

    base_rank = token_rank(base_last, tok_id) if tok_id is not None else -1
    steered_rank = token_rank(steered_last, tok_id) if tok_id is not None else -1
    base_prob = float(base_probs[tok_id].item()) if tok_id is not None else 0.0
    steered_prob = float(steered_probs[tok_id].item()) if tok_id is not None else 0.0

    base_top5 = top_k_tokens(base_last, model.tokenizer)
    steered_top5 = top_k_tokens(steered_last, model.tokenizer)

    rank_change: int | None = None
    if base_rank > 0 and steered_rank > 0:
        rank_change = steered_rank - base_rank

    # Format top-5 as "token(prob%)" strings for easy reading
    def fmt_top5(top5: list[tuple[str, float]]) -> str:
        return ", ".join(f"{tok.strip()}({prob:.1%})" for tok, prob in top5)

    # Which tokens appear in steered top-5 but not baseline, and vice versa
    base_tokens = {tok.strip() for tok, _ in base_top5}
    steered_tokens = {tok.strip() for tok, _ in steered_top5}

    return {
        "n_features_steered": len(scaled_tuples),
        "correct_token_id": tok_id,
        "baseline_rank": base_rank if base_rank > 0 else None,
        "steered_rank": steered_rank if steered_rank > 0 else None,
        "rank_change": rank_change,
        "baseline_prob": round(base_prob, 6),
        "steered_prob": round(steered_prob, 6),
        "prob_drop": round(base_prob - steered_prob, 6),
        "baseline_top5": fmt_top5(base_top5),
        "steered_top5": fmt_top5(steered_top5),
        "top5_new": ", ".join(steered_tokens - base_tokens) or "—",
        "top5_dropped": ", ".join(base_tokens - steered_tokens) or "—",
        "top1_changed": (
            base_top5[0][0].strip() != steered_top5[0][0].strip()
            if base_top5 and steered_top5
            else False
        ),
        "skipped": None,
    }


# ---------------------------------------------------------------------------
# Empty result row (no hop features found or graph missing)
# ---------------------------------------------------------------------------

EMPTY_METRICS: dict = {
    "n_features_steered": 0,
    "correct_token_id": None,
    "baseline_rank": None,
    "steered_rank": None,
    "rank_change": None,
    "baseline_prob": None,
    "steered_prob": None,
    "prob_drop": None,
    "baseline_top5": None,
    "steered_top5": None,
    "top5_new": None,
    "top5_dropped": None,
    "top1_changed": None,
    "skipped": None,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(ground_truth_path: Path, variants: list[str], dry_run: bool = False) -> None:
    rows = load_ground_truth(ground_truth_path)
    if not rows:
        print("ERROR: ground_truth.csv is empty or missing.", file=sys.stderr)
        sys.exit(1)

    model = None
    if not dry_run:
        print(f"Loading {MODEL_NAME} (transcoder: {TRANSCODER_SET})...")
        from circuit_tracer import ReplacementModel

        model = ReplacementModel.from_pretrained(
            MODEL_NAME, TRANSCODER_SET, dtype=torch.bfloat16
        )
        print("Model loaded.\n")

    results: list[dict] = []

    for row in rows:
        slug = row["slug"].strip()
        prompt = row["prompt"].strip()
        intermediate_concept = row.get("intermediate_concept", "").strip()
        correct_answer = row.get("correct_answer", "").strip()
        hop_type = row.get("hop_type", "").strip()
        num_hops_str = row.get("num_hops", "").strip()

        # Only run on 2-hop cases (one intermediate step, e.g. dallas→texas→austin).
        # CSVs without a num_hops column (capital, anthropic, multihop) are all 2-hop
        # by construction and pass through. Sub-hop rows and 3-hop+ are skipped.
        if num_hops_str and num_hops_str != "2":
            continue
        if not intermediate_concept or intermediate_concept.upper() == "N/A":
            continue

        for variant in variants:
            graph_slug = f"{slug}-{DESCRIPTION_VARIANT}-{variant}"
            graph_path = TEST_GRAPHS_DIR / f"{graph_slug}.json"
            if not graph_path.exists():
                # Single-variant mode: graph was saved as {slug}.json
                graph_slug = slug
                graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
            graph = load_graph(graph_path)

            base_row = {
                "slug": slug,
                "variant": variant,
                "hop_type": hop_type,
                "intermediate_concept": intermediate_concept,
                "correct_answer": correct_answer,
            }

            if graph is None:
                print(f"  SKIP {graph_slug} — graph not found")
                results.append({**base_row, **EMPTY_METRICS, "skipped": "graph_missing"})
                continue

            # Use the prompt stored in the graph metadata — it matches exactly what
            # was passed during attribution (e.g. chat-wrapped for mquake prompts).
            graph_prompt = graph.get("metadata", {}).get("prompt", "").replace("<bos>", "").strip()
            if graph_prompt:
                prompt = graph_prompt

            feature_groups = load_feature_groups(slug, variant)
            hop_features = find_hop_features(graph, intermediate_concept, feature_groups)
            print(
                f"  {graph_slug}: {len(hop_features)} hop feature(s) for '{intermediate_concept}'"
            )

            if dry_run:
                results.append({
                    **base_row,
                    **EMPTY_METRICS,
                    "n_features_steered": len(hop_features),
                    "skipped": "dry_run",
                })
                continue

            if not hop_features:
                results.append({**base_row, **EMPTY_METRICS, "skipped": "no_hop_features"})
                continue

            metrics = run_single(model, prompt, correct_answer, hop_features)
            results.append({**base_row, **metrics})

    if not results:
        print("No results — check that graphs exist in test_graphs/ and have been processed.")
        return

    if dry_run:
        print("\nDry-run complete — model not loaded.")
        for r in results:
            print(f"  {r['slug']} ({r['variant']}): {r['n_features_steered']} hop feature(s)")
        return

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    csv_path = ARTIFACTS_DIR / "intervention_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV  -> {csv_path}")

    # ------------------------------------------------------------------
    # Write Markdown report
    # ------------------------------------------------------------------
    valid = [r for r in results if r.get("baseline_rank") is not None]
    n = len(valid)

    md_path = ARTIFACTS_DIR / "intervention_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Intervention Results\n\n")
        f.write(f"Variants: {', '.join(variants)}  \n")
        f.write(f"Prompts: {len(rows)}  \n")
        f.write(f"Graph runs total: {len(results)}  \n")
        f.write(f"Runs with hop features steered: {n}\n\n")
        f.write(f"Steering: multiplicative, scaling_factor={SCALING_FACTOR} (paper methodology)\n\n")

        if n > 0:
            n_top1_changed = sum(1 for r in valid if r.get("top1_changed"))
            mean_prob_drop = sum(
                r["prob_drop"] for r in valid if r.get("prob_drop") is not None
            ) / n
            n_rank_worsened = sum(
                1 for r in valid if (r.get("rank_change") or 0) > 0
            )

            f.write("## Aggregate Stats\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Runs with hop features steered | {n} |\n")
            f.write(
                f"| Top-1 prediction changed after steering | "
                f"{n_top1_changed} / {n} ({n_top1_changed / n:.1%}) |\n"
            )
            f.write(f"| Mean correct-answer prob drop | {mean_prob_drop:.4f} |\n")
            f.write(
                f"| Rank worsened after steering | "
                f"{n_rank_worsened} / {n} ({n_rank_worsened / n:.1%}) |\n\n"
            )

        for variant in variants:
            vr = [r for r in results if r["variant"] == variant]
            runnable = [r for r in vr if r.get("baseline_rank") is not None]
            skipped = [r for r in vr if r.get("skipped")]

            f.write(f"## Per-Prompt Results — variant {variant}\n\n")

            if skipped:
                f.write(f"Skipped ({len(skipped)}): ")
                f.write(
                    ", ".join(f"{r['slug']} ({r['skipped']})" for r in skipped)
                )
                f.write("\n\n")

            if not runnable:
                f.write("_No intervention results for this variant._\n\n")
                continue

            f.write(
                "| Slug | Intermediate | # Steered | Rank Δ | Prob drop | "
                "Baseline top-5 | Steered top-5 | New in top-5 | Dropped from top-5 |\n"
            )
            f.write(
                "|------|--------------|-----------|--------|-----------|"
                "----------------|---------------|--------------|--------------------|\n"
            )
            for r in runnable:
                rc = r.get("rank_change")
                if rc is None:
                    delta_str = "—"
                elif rc > 0:
                    delta_str = f"+{rc}"
                else:
                    delta_str = str(rc)
                f.write(
                    f"| {r['slug']} "
                    f"| {r['intermediate_concept']} "
                    f"| {r['n_features_steered']} "
                    f"| {delta_str} "
                    f"| {r.get('prob_drop', 0.0):.4f} "
                    f"| {r.get('baseline_top5', '—')} "
                    f"| {r.get('steered_top5', '—')} "
                    f"| {r.get('top5_new', '—')} "
                    f"| {r.get('top5_dropped', '—')} |\n"
                )
            f.write("\n")

    print(f"Report -> {md_path}")

    # Terminal summary
    if n > 0:
        print()
        print("=" * 55)
        print(f"  Graph runs:              {len(results)}")
        print(f"  With hop features:       {n}")
        print(f"  Top-1 changed:           {n_top1_changed} / {n}")
        print(f"  Mean prob drop:          {mean_prob_drop:.4f}")
        print(f"  Rank worsened:           {n_rank_worsened} / {n}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Negatively steer intermediate-hop features and measure causal effect."
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to ground_truth.csv",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="a0,a1,a2,a3",
        help="Comma-separated grouping variants to run (e.g. a0,a3)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip model loading — report which features would be ablated",
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(args.ground_truth, variants, dry_run=args.dry_run)