"""
analyze_hops.py — Intermediate hop detection across multi-hop attribution graphs.

For each prompt in ground_truth.csv, loads its processed graph JSON and checks:
  1. What the model predicted (top logit token).
  2. Whether the model got it right.
  3. Whether intermediate-hop features are present in the graph — detectable
     either in individual node clerp descriptions or in supernode group names.
  4. How strong those intermediate-hop features are (mean influence score).

MQuAKE multi-hop validation (when using ground_truth_mquake.csv):
  For each full multi-hop question, the attribution graph is used to predict
  the longest sub-chain the model would get right (predicted_max_hop).  This
  is compared against the actual deepest sub-question answered correctly
  (ground_truth_max_hop), computed from the sub-question graphs.

Outputs:
  - artifacts/hop_analysis.csv          — one row per (slug, variant)
  - artifacts/hop_analysis.md           — human-readable summary report
  - artifacts/mquake_hop_accuracy.csv   — one row per MQuAKE case (if applicable)

Usage:
    python analyze_hops.py
    python analyze_hops.py --variants a0,a3
    python analyze_hops.py --ground_truth ../prompts/ground_truth_mquake.csv --variants a0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GROUND_TRUTH = REPO_ROOT / "prompts/ground_truth_capital.csv"
DESCRIPTION_VARIANT = "v2"


# ---------------------------------------------------------------------------
# Helpers
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


def get_model_prediction(graph: dict) -> tuple[str, float]:
    """Return (predicted_token, probability) for the top logit node."""
    logit_nodes = [
        n for n in graph.get("nodes", [])
        if n.get("feature_type") == "logit" or n.get("is_target_logit")
    ]
    if not logit_nodes:
        return ("", 0.0)

    logit_nodes.sort(key=lambda n: float(n.get("token_prob", 0.0)), reverse=True)
    top = logit_nodes[0]

    # Extract token from clerp: 'Output " Sacramento" (p=0.277)'
    clerp = top.get("clerp", "")
    match = re.search(r'"([^"]+)"', clerp)
    token = match.group(1).strip() if match else clerp.strip()
    prob = float(top.get("token_prob", 0.0))
    return (token, prob)


def get_target_prediction(graph: dict) -> tuple[str, float]:
    """Return the token the prompt was targeting (is_target_logit=True)."""
    for n in graph.get("nodes", []):
        if n.get("is_target_logit"):
            clerp = n.get("clerp", "")
            match = re.search(r'"([^"]+)"', clerp)
            token = match.group(1).strip() if match else clerp.strip()
            prob = float(n.get("token_prob", 0.0))
            return (token, prob)
    return ("", 0.0)


def get_top_k_predictions(graph: dict, k: int = 5) -> list[tuple[str, float]]:
    """Return the top-k logit tokens sorted by probability (highest first)."""
    logit_nodes = [
        n for n in graph.get("nodes", [])
        if n.get("feature_type") == "logit" or n.get("is_target_logit")
    ]
    logit_nodes.sort(key=lambda n: float(n.get("token_prob", 0.0)), reverse=True)
    result = []
    for n in logit_nodes[:k]:
        clerp = n.get("clerp", "")
        match = re.search(r'"([^"]+)"', clerp)
        token = match.group(1).strip() if match else clerp.strip()
        prob = float(n.get("token_prob", 0.0))
        result.append((token, prob))
    return result


def find_correct_rank(top_k: list[tuple[str, float]], correct_answer: str) -> int | None:
    """Return 1-indexed rank of correct_answer in top_k, or None if not found."""
    for rank, (token, _) in enumerate(top_k, start=1):
        if answer_matches(token, correct_answer):
            return rank
    return None


def rank_to_score(rank: int | None, k: int = 5) -> float:
    """
    Convert a top-k rank to a 0.0–1.0 score.
      rank 1 → 1.0  (top prediction)
      rank 2 → 0.8
      rank 3 → 0.6
      rank 4 → 0.4
      rank 5 → 0.2
      None   → 0.0  (not in top k)
    """
    if rank is None or rank > k:
        return 0.0
    return (k + 1 - rank) / k


def parse_supernodes(graph: dict) -> list[tuple[str, list[str]]]:
    """Parse qParams.supernodes → list of (group_name, [node_ids])."""
    raw = graph.get("qParams", {}).get("supernodes", "[]")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    result = []
    for item in items:
        if isinstance(item, list) and len(item) >= 1:
            result.append((str(item[0]), [str(x) for x in item[1:]]))
    return result


def concept_in_text(concept: str, text: str) -> bool:
    """Case-insensitive whole-word match of concept in text."""
    if not concept or not text:
        return False
    pattern = re.compile(r'\b' + re.escape(concept.lower()) + r'\b')
    return bool(pattern.search(text.lower()))


def detect_intermediate_hop(
    graph: dict,
    intermediate_concept: str,
) -> dict:
    """
    Scan all transcoder feature nodes for mentions of the intermediate concept.
    Also check supernode group names.

    Returns a dict with detection metrics.
    """
    nodes = graph.get("nodes", [])
    transcoder_nodes = [
        n for n in nodes
        if n.get("feature_type") == "cross layer transcoder"
    ]

    matching_nodes = []
    for n in transcoder_nodes:
        clerp = n.get("clerp", "")
        if concept_in_text(intermediate_concept, clerp):
            matching_nodes.append(n)

    matching_influences = [
        float(n.get("influence", 0.0))
        for n in matching_nodes
        if n.get("influence") is not None
    ]

    supernodes = parse_supernodes(graph)
    matching_groups = [
        name for name, _ in supernodes
        if concept_in_text(intermediate_concept, name)
    ]

    # Fraction of transcoder nodes that mention the intermediate concept
    total_transcoder = len(transcoder_nodes)
    frac = len(matching_nodes) / total_transcoder if total_transcoder > 0 else 0.0

    mean_influence = (
        sum(matching_influences) / len(matching_influences)
        if matching_influences else 0.0
    )
    max_influence = max(matching_influences) if matching_influences else 0.0

    return {
        "total_transcoder_nodes": total_transcoder,
        "hop_feature_count": len(matching_nodes),
        "hop_feature_fraction": round(frac, 4),
        "hop_mean_influence": round(mean_influence, 4),
        "hop_max_influence": round(max_influence, 4),
        "hop_found_in_clerp": len(matching_nodes) > 0,
        "hop_groups": matching_groups,
        "hop_found_in_groups": len(matching_groups) > 0,
        "hop_found": len(matching_nodes) > 0 or len(matching_groups) > 0,
    }


def answer_matches(predicted: str, correct: str) -> bool:
    """Loose match: correct_answer is a substring of predicted or vice versa."""
    p = predicted.lower().strip()
    c = correct.lower().strip()
    return c in p or p in c


# ---------------------------------------------------------------------------
# Multi-hop helpers (MQuAKE)
# ---------------------------------------------------------------------------

_MQUAKE_FULL_RE = re.compile(r"^mquake-(\d+)$")
_MQUAKE_HOP_RE = re.compile(r"^mquake-(\d+)-h(\d+)$")


def parse_mquake_slug(slug: str) -> tuple[str, int | None] | None:
    """
    Return (case_id, hop_k) for MQuAKE slugs, or None if not a MQuAKE slug.
    hop_k is None for full-question slugs (mquake-42),
    an integer for sub-question slugs (mquake-42-h2 → hop_k=2).
    """
    m = _MQUAKE_FULL_RE.match(slug)
    if m:
        return (m.group(1), None)
    m = _MQUAKE_HOP_RE.match(slug)
    if m:
        return (m.group(1), int(m.group(2)))
    return None


def detect_all_hops(graph: dict, intermediate_concept_str: str) -> dict:
    """
    For multi-hop questions whose intermediate_concept is ' | '-separated,
    check each intermediate concept individually in the graph.

    predicted_max_hop: highest *consecutive* hop (from hop 1) where the
    concept is detected.  Stops at the first gap.

    Example:
      concepts = ["USA", "Donald Trump"]   (for a 3-hop question)
      If "USA" found but "Donald Trump" not → predicted_max_hop = 1
      If both found → predicted_max_hop = 2
    """
    concepts = [c.strip() for c in intermediate_concept_str.split("|")
                if c.strip() and c.strip().upper() != "N/A"]

    if not concepts:
        return {
            "num_intermediate_hops": 0,
            "predicted_max_hop": None,
            "hops_found_in_graph": "",
        }

    found_hops: list[int] = []
    predicted_max_hop = 0

    for k, concept in enumerate(concepts, start=1):
        result = detect_intermediate_hop(graph, concept)
        if result["hop_found"]:
            found_hops.append(k)
            predicted_max_hop = k   # only advances if consecutive from start
        else:
            break                   # stop at first gap

    return {
        "num_intermediate_hops": len(concepts),
        "predicted_max_hop": predicted_max_hop,
        "hops_found_in_graph": ",".join(str(h) for h in found_hops),
    }


def compute_mquake_validation(results: list[dict]) -> list[dict]:
    """
    Group MQuAKE results by case_id and compute hop-prediction accuracy.

    For each case:
      predicted_max_hop   — from the full-question attribution graph
      ground_truth_max_hop — highest hop k where the model correctly answers
                             sub-question mquake-{id}-hk  (None if not run)
      hop_prediction_correct — True/False/None
    """
    from collections import defaultdict

    by_case: dict[str, dict] = defaultdict(dict)

    for r in results:
        parsed = parse_mquake_slug(r["slug"])
        if parsed is None:
            continue
        case_id, hop_k = parsed
        if hop_k is None:
            by_case[case_id]["full"] = r
        else:
            by_case[case_id].setdefault("hops", {})[hop_k] = r

    case_results: list[dict] = []
    for case_id, case_data in sorted(by_case.items(), key=lambda x: int(x[0])):
        full = case_data.get("full")
        if full is None:
            continue
        hops: dict[int, dict] = case_data.get("hops", {})

        ground_truth_max_hop: int | None = None
        if hops:
            correct_hops = [k for k, r in hops.items() if r.get("model_correct")]
            ground_truth_max_hop = max(correct_hops) if correct_hops else 0

        predicted_max_hop = full.get("predicted_max_hop")
        hop_prediction_correct: bool | None = None
        if predicted_max_hop is not None and ground_truth_max_hop is not None:
            hop_prediction_correct = (predicted_max_hop == ground_truth_max_hop)

        case_results.append({
            "case_id": case_id,
            "full_slug": full["slug"],
            "num_hops": full.get("num_hops", ""),
            "full_model_correct": full.get("model_correct", False),
            "correct_answer": full.get("correct_answer", ""),
            "predicted": full.get("predicted", ""),
            "predicted_max_hop": predicted_max_hop,
            "hops_found_in_graph": full.get("hops_found_in_graph", ""),
            "ground_truth_max_hop": ground_truth_max_hop,
            "hop_prediction_correct": hop_prediction_correct,
            "subhop_results": {
                k: {"correct": r.get("model_correct"), "predicted": r.get("predicted")}
                for k, r in hops.items()
            },
        })

    return case_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(ground_truth_path: Path, variants: list[str]) -> None:
    rows = load_ground_truth(ground_truth_path)
    if not rows:
        print("ERROR: ground_truth.csv is empty or missing.", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = []

    for row in rows:
        slug = row["slug"].strip()
        intermediate_concept = row.get("intermediate_concept", "").strip()
        correct_answer = row.get("correct_answer", "").strip()
        hop_type = row.get("hop_type", "").strip()
        notes = row.get("notes", "").strip()

        # Extra fields from ground_truth (e.g. num_hops from MQuAKE CSV)
        num_hops_str = row.get("num_hops", "").strip()

        for variant in variants:
            graph_slug = f"{slug}-{DESCRIPTION_VARIANT}-{variant}"
            graph_path = TEST_GRAPHS_DIR / f"{graph_slug}.json"
            if not graph_path.exists():
                # Single-variant mode: graph saved as plain slug
                graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
                graph_slug = slug

            graph = load_graph(graph_path)
            if graph is None:
                print(f"  SKIP {slug} — graph not found")
                continue

            top_token, top_prob = get_model_prediction(graph)
            target_token, target_prob = get_target_prediction(graph)

            # Use the target logit token as predicted (it's what the pipeline chose to attribute)
            predicted = target_token if target_token else top_token
            predicted_prob = target_prob if target_token else top_prob

            correct = answer_matches(predicted, correct_answer)

            top_k = get_top_k_predictions(graph, k=5)
            correct_rank = find_correct_rank(top_k, correct_answer)
            score = rank_to_score(correct_rank)

            hop_metrics = detect_intermediate_hop(graph, intermediate_concept)
            all_supernodes = [name for name, _ in parse_supernodes(graph)]

            # Multi-hop fields: populated for MQuAKE full questions (intermediate_concept has '|')
            # and for any row whose intermediate_concept lists multiple pipe-separated concepts.
            is_multihop_row = "|" in intermediate_concept
            if is_multihop_row and parse_mquake_slug(slug) is not None:
                mh = detect_all_hops(graph, intermediate_concept)
            else:
                mh = {"num_intermediate_hops": 0, "predicted_max_hop": None, "hops_found_in_graph": ""}

            results.append({
                "slug": slug,
                "variant": variant,
                "hop_type": hop_type,
                "num_hops": num_hops_str,
                "intermediate_concept": intermediate_concept,
                "correct_answer": correct_answer,
                "predicted": predicted,
                "predicted_prob": round(predicted_prob, 4),
                "model_correct": correct,
                "top_k_rank": correct_rank,
                "rank_score": round(score, 2),
                "notes": notes,
                "all_supernodes": all_supernodes,
                **hop_metrics,
                **mh,
            })

    if not results:
        print("No results -- check that graphs exist in test_graphs/ and have been processed by batch_all_groups.sh.")
        return

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    csv_path = ARTIFACTS_DIR / "hop_analysis.csv"
    # Collect all fieldnames across all result rows (MQuAKE rows add extra fields)
    all_fieldnames: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_fieldnames.append(k)
                seen.add(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in all_fieldnames})
    print(f"CSV -> {csv_path}")

    # ------------------------------------------------------------------
    # MQuAKE multi-hop validation (only when MQuAKE slugs are present)
    # ------------------------------------------------------------------
    mquake_cases = compute_mquake_validation(results)
    if mquake_cases:
        mq_csv_path = ARTIFACTS_DIR / "mquake_hop_accuracy.csv"
        mq_fields = [
            "case_id", "full_slug", "num_hops", "correct_answer", "predicted",
            "full_model_correct", "hops_found_in_graph",
            "predicted_max_hop", "ground_truth_max_hop", "hop_prediction_correct",
        ]
        with open(mq_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=mq_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mquake_cases)
        print(f"MQuAKE CSV -> {mq_csv_path}")

    # ------------------------------------------------------------------
    # Write Markdown report
    # ------------------------------------------------------------------
    md_path = ARTIFACTS_DIR / "hop_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Intermediate Hop Analysis\n\n")
        f.write(f"Variants analysed: {', '.join(variants)}  \n")
        f.write(f"Prompts analysed: {len(rows)}  \n")
        f.write(f"Total graph runs: {len(results)}\n\n")

        # ---- aggregate stats ----
        n = len(results)
        n_correct = sum(1 for r in results if r["model_correct"])
        n_top5    = sum(1 for r in results if r["top_k_rank"] is not None)
        mean_score = sum(r["rank_score"] for r in results) / n if n else 0.0
        n_hop_found = sum(1 for r in results if r["hop_found"])
        n_hop_clerp = sum(1 for r in results if r["hop_found_in_clerp"])
        n_hop_group = sum(1 for r in results if r["hop_found_in_groups"])

        # Correct + hop found vs wrong + hop found (binary top-1)
        correct_with_hop    = sum(1 for r in results if r["model_correct"] and r["hop_found"])
        correct_without_hop = sum(1 for r in results if r["model_correct"] and not r["hop_found"])
        wrong_with_hop      = sum(1 for r in results if not r["model_correct"] and r["hop_found"])
        wrong_without_hop   = sum(1 for r in results if not r["model_correct"] and not r["hop_found"])

        f.write("## Aggregate Stats\n\n")
        f.write(f"| Metric | Count | Fraction |\n")
        f.write(f"|--------|-------|----------|\n")
        f.write(f"| Model correct (top-1) | {n_correct} | {n_correct/n:.1%} |\n")
        f.write(f"| Correct answer in top-5 | {n_top5} | {n_top5/n:.1%} |\n")
        f.write(f"| Mean rank score (0–1) | — | {mean_score:.2f} |\n")
        f.write(f"| Intermediate hop found (any) | {n_hop_found} | {n_hop_found/n:.1%} |\n")
        f.write(f"| Hop found in feature clerps | {n_hop_clerp} | {n_hop_clerp/n:.1%} |\n")
        f.write(f"| Hop found in supernode groups | {n_hop_group} | {n_hop_group/n:.1%} |\n")
        f.write(f"\n")
        f.write(f"## Correctness × Hop Presence\n\n")
        f.write(f"| | Hop found | No hop found |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Model correct** | {correct_with_hop} | {correct_without_hop} |\n")
        f.write(f"| **Model wrong** | {wrong_with_hop} | {wrong_without_hop} |\n\n")

        if wrong_with_hop + wrong_without_hop > 0:
            wrong_hop_rate = wrong_with_hop / (wrong_with_hop + wrong_without_hop)
            f.write(f"> When model is **wrong**: hop found in {wrong_hop_rate:.1%} of cases  \n")
        if correct_with_hop + correct_without_hop > 0:
            correct_hop_rate = correct_with_hop / (correct_with_hop + correct_without_hop)
            f.write(f"> When model is **correct**: hop found in {correct_hop_rate:.1%} of cases\n\n")

        # ---- per-variant breakdown ----
        for variant in variants:
            vr = [r for r in results if r["variant"] == variant]
            if not vr:
                continue
            f.write(f"## Per-Prompt Results — variant {variant}\n\n")
            f.write("Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0\n\n")
            f.write("| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |\n")
            f.write("|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|\n")
            for r in vr:
                rank_str = str(r["top_k_rank"]) if r["top_k_rank"] is not None else "—"
                hop_tick = "✓" if r["hop_found"] else "✗"
                groups_str = ", ".join(r["hop_groups"]) if r["hop_groups"] else "—"
                f.write(
                    f"| {r['slug']} "
                    f"| {r['hop_type']} "
                    f"| {r['intermediate_concept']} "
                    f"| {r['predicted']} ({r['predicted_prob']:.1%}) "
                    f"| {rank_str} "
                    f"| {r['rank_score']:.1f} "
                    f"| {hop_tick} ({r['hop_feature_count']} feat, groups: {groups_str}) "
                    f"| {r['hop_feature_count']} ({r['hop_feature_fraction']:.1%}) "
                    f"| {r['hop_mean_influence']:.4f} |\n"
                )
            f.write("\n")

        # ---- "wrong but hop present" spotlight ----
        wrong_hop_cases = [r for r in results if not r["model_correct"] and r["hop_found"]]
        if wrong_hop_cases:
            f.write("## Spotlight: Model Wrong but Intermediate Hop Present\n\n")
            f.write("These are the most interpretability-interesting cases — the model encoded "
                    "the intermediate concept but still predicted incorrectly.\n\n")
            for r in wrong_hop_cases:
                f.write(f"### {r['slug']} ({r['variant']})\n")
                f.write(f"- Prompt type: {r['hop_type']}\n")
                f.write(f"- Intermediate concept: **{r['intermediate_concept']}**\n")
                f.write(f"- Correct answer: {r['correct_answer']}\n")
                f.write(f"- Model predicted: **{r['predicted']}** ({r['predicted_prob']:.1%})\n")
                f.write(f"- Hop features: {r['hop_feature_count']} ({r['hop_feature_fraction']:.1%} of transcoder nodes)\n")
                f.write(f"- Hop in supernode groups: {r['hop_groups']}\n")
                f.write(f"- Mean influence of hop features: {r['hop_mean_influence']:.4f}\n\n")

        # ---- MQuAKE multi-hop validation ----
        if mquake_cases:
            f.write("## MQuAKE Multi-Hop Validation\n\n")
            f.write(
                "For each case: the attribution graph of the **full question** is used to "
                "predict the longest hop the model can correctly resolve "
                "(`predicted_max_hop`). This is compared against the actual deepest "
                "sub-question the model answers correctly (`ground_truth_max_hop`).\n\n"
            )

            n_mq = len(mquake_cases)
            with_gt = [c for c in mquake_cases if c["ground_truth_max_hop"] is not None]
            n_correct_pred = sum(1 for c in with_gt if c["hop_prediction_correct"])
            n_full_correct = sum(1 for c in mquake_cases if c["full_model_correct"])

            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Cases analysed | {n_mq} |\n")
            f.write(f"| Full question correct (top-1) | {n_full_correct} / {n_mq} ({n_full_correct/n_mq:.1%}) |\n")
            if with_gt:
                f.write(f"| Cases with sub-question graphs | {len(with_gt)} |\n")
                f.write(f"| Hop prediction accuracy | {n_correct_pred} / {len(with_gt)} ({n_correct_pred/len(with_gt):.1%}) |\n")
            f.write("\n")

            f.write("| Case | Hops | Full correct? | Predicted max hop | GT max hop | Prediction correct? | Hops in graph |\n")
            f.write("|------|------|---------------|-------------------|------------|---------------------|---------------|\n")
            for c in mquake_cases:
                full_tick = "✓" if c["full_model_correct"] else "✗"
                pred_hop = c["predicted_max_hop"] if c["predicted_max_hop"] is not None else "—"
                gt_hop = c["ground_truth_max_hop"] if c["ground_truth_max_hop"] is not None else "—"
                pred_tick = ("✓" if c["hop_prediction_correct"] else "✗") if c["hop_prediction_correct"] is not None else "—"
                found_str = c["hops_found_in_graph"] or "—"
                f.write(
                    f"| {c['full_slug']} | {c['num_hops']} | {full_tick} "
                    f"| {pred_hop} | {gt_hop} | {pred_tick} | {found_str} |\n"
                )
            f.write("\n")

    print(f"Report -> {md_path}")

    # ------------------------------------------------------------------
    # Write correct-but-no-hop results file
    # ------------------------------------------------------------------
    results_dir = PACKAGE_DIR / "analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    correct_no_hop = [r for r in results if r["model_correct"] and not r["hop_found"]]
    results_path = results_dir / "correct_no_hop.md"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# Correct Prediction — Intermediate Hop Not Found\n\n")
        f.write(f"Cases: {len(correct_no_hop)}\n\n")
        for r in correct_no_hop:
            f.write("---\n")
            f.write(f"## {r['prompt_text']} — {r['slug']}\n\n")
            f.write(f"**Predicted:** {r['predicted']} ({r['predicted_prob']:.1%})  \n")
            f.write(f"**Correct answer:** {r['correct_answer']}  \n")
            f.write(f"**Missing hop:** {r['intermediate_concept']}\n\n")
            f.write("### Supernodes\n")
            groups = [g for g in r["all_supernodes"] if not g.startswith("Emb:") and not g.startswith("Output:")]
            f.write(", ".join(groups) if groups else "—")
            f.write("\n\n")
    print(f"Results -> {results_path}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print(f"  Graphs analysed:       {n}")
    print(f"  Model correct (top-1): {n_correct}/{n} ({n_correct/n:.1%})")
    print(f"  Correct in top-5:      {n_top5}/{n} ({n_top5/n:.1%})")
    print(f"  Mean rank score:       {mean_score:.2f} / 1.00")
    print(f"  Hop found (any):       {n_hop_found}/{n} ({n_hop_found/n:.1%})")
    print(f"  Correct + hop:         {correct_with_hop}")
    print(f"  Wrong + hop:           {wrong_with_hop}  <-- most interesting")
    if mquake_cases:
        with_gt = [c for c in mquake_cases if c["ground_truth_max_hop"] is not None]
        n_correct_pred = sum(1 for c in with_gt if c["hop_prediction_correct"])
        print(f"  --- MQuAKE ({len(mquake_cases)} cases) ---")
        print(f"  Hop prediction acc:    {n_correct_pred}/{len(with_gt)} ({n_correct_pred/len(with_gt):.1%})" if with_gt else "  Sub-question graphs not yet run")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect intermediate hops in attribution graphs.")
    parser.add_argument(
        "--ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH,
        help="Path to ground_truth.csv"
    )
    parser.add_argument(
        "--variants", type=str, default="a0,a1,a2,a3",
        help="Comma-separated grouping variants to analyse (e.g. a0,a3)"
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(args.ground_truth, variants)
