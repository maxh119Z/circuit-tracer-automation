"""
explore_interesting_graphs.py — Open-ended exploration to surface surprising graphs.
writes artifacts/interesting_graphs.csv and artifacts/interesting_graphs.md.

Usage
-----
    python analysis/explore_interesting_graphs.py
    python explore_interesting_graphs.py --graphs_dir ../../test_graphs --top_k 20
    python explore_interesting_graphs.py --min_confidence 0.05 --influence_threshold 0.3 --no_llm
    python analysis/explore_interesting_graphs.py --ground_truth ../prompts/ground_truth_wikipedia.csv --variants a2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
ARTIFACTS_DIR = PACKAGE_DIR / "analysis" / "results"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def get_layer_order(node_id: str) -> float:
    """
    Return a sortable layer index for a node.
      Embedding nodes  (E_...)  → -1
      Transcoder nodes ({L}_...)→  L  (integer)
      Logit nodes are already at max layer; we treat them as N+1.
    """
    if node_id.startswith("E_"):
        return -1.0
    parts = node_id.split("_")
    try:
        return float(parts[0])
    except ValueError:
        return 999.0


def longest_path_through_transcoder(graph: dict, influence_threshold: float) -> int:
    """
    Find the longest path through transcoder nodes whose influence ≥ threshold.

    Only transcoder-to-transcoder edges count toward path length.
    Embedding and logit nodes are used as anchors but not counted.

    Returns the number of transcoder nodes on the longest such path.
    """
    nodes_raw = graph.get("nodes", [])
    links_raw = graph.get("links", [])

    # Build a set of "eligible" node IDs: transcoder nodes above threshold
    eligible: set[str] = set()
    for n in nodes_raw:
        if (
            n.get("feature_type") == "cross layer transcoder"
            and (n.get("influence") or 0.0) >= influence_threshold
        ):
            eligible.add(n["node_id"])

    if not eligible:
        return 0

    # Build adjacency: eligible → eligible only
    children: dict[str, list[str]] = defaultdict(list)
    for link in links_raw:
        src, tgt = link.get("source", ""), link.get("target", "")
        if src in eligible and tgt in eligible:
            children[src].append(tgt)

    # Topological sort by layer order
    topo = sorted(eligible, key=get_layer_order)

    # DP: dp[node] = length of longest path ending at node
    dp: dict[str, int] = {n: 1 for n in eligible}
    for node in topo:
        for child in children[node]:
            if dp[node] + 1 > dp[child]:
                dp[child] = dp[node] + 1

    return max(dp.values()) if dp else 0


def get_model_confidence(graph: dict) -> tuple[str, float]:
    """Return (predicted_token, probability) for the target logit node."""
    for n in graph.get("nodes", []):
        if n.get("is_target_logit"):
            clerp = n.get("clerp", "")
            m = re.search(r'"([^"]+)"', clerp)
            token = m.group(1).strip() if m else clerp.strip()
            return token, float(n.get("token_prob", 0.0))
    # Fall back to highest-prob logit
    logit_nodes = [n for n in graph.get("nodes", []) if n.get("feature_type") == "logit"]
    if logit_nodes:
        top = max(logit_nodes, key=lambda n: float(n.get("token_prob", 0.0)))
        clerp = top.get("clerp", "")
        m = re.search(r'"([^"]+)"', clerp)
        token = m.group(1).strip() if m else clerp.strip()
        return token, float(top.get("token_prob", 0.0))
    return ("", 0.0)


def get_supernode_names(graph: dict) -> list[str]:
    """Parse qParams.supernodes → list of group name strings."""
    raw = graph.get("qParams", {}).get("supernodes", "[]")
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    names = []
    for item in items:
        if isinstance(item, list) and item:
            name = str(item[0])
            if not name.startswith("Emb:") and not name.startswith("Output:"):
                names.append(name)
    return names


def get_top_clerps(graph: dict, k: int = 5) -> list[str]:
    """Return the top-k transcoder node clerps by influence score."""
    nodes = [
        n for n in graph.get("nodes", [])
        if n.get("feature_type") == "cross layer transcoder" and n.get("clerp")
    ]
    nodes.sort(key=lambda n: abs(float(n.get("influence") or 0.0)), reverse=True)
    return [n["clerp"] for n in nodes[:k]]


def confidence_band(confidence: float) -> str:
    """
    Describe the confidence level in terms useful for the LLM judge.
    Very high confidence often means a trivial/obvious answer;
    moderate confidence may indicate more interesting internal reasoning.
    """
    if confidence >= 0.90:
        return "very high (may indicate a trivial or obvious answer)"
    if confidence >= 0.50:
        return "high (model is fairly certain)"
    if confidence >= 0.20:
        return "moderate (model shows non-trivial reasoning)"
    return "low (model is uncertain)"


def compute_metrics(graph: dict, influence_threshold: float) -> dict:
    predicted, confidence = get_model_confidence(graph)
    path_len = longest_path_through_transcoder(graph, influence_threshold)
    transcoder_nodes = sum(
        1 for n in graph.get("nodes", [])
        if n.get("feature_type") == "cross layer transcoder"
    )
    supernode_names = get_supernode_names(graph)
    prompt = graph.get("metadata", {}).get("prompt", "")

    return {
        "prompt": prompt,
        "predicted": predicted,
        "model_confidence": round(confidence, 4),
        "longest_path_length": path_len,
        "num_transcoder_nodes": transcoder_nodes,
        "num_supernode_groups": len(supernode_names),
        "supernode_names": supernode_names,
        "top_clerps": get_top_clerps(graph) if not supernode_names else [],
    }


# ---------------------------------------------------------------------------
# LLM judge — per-criterion scoring
# ---------------------------------------------------------------------------

LLM_JUDGE_MODEL = "gpt-5.4"

# Each LLM criterion: (json_key, human label, description for the prompt)
LLM_CRITERIA: list[tuple[str, str, str]] = [
    (
        "parallel_computation",
        "Parallel computation",
        "The supernodes reflect internal computation toward multiple different "
        "possible next-token predictions. Trivial variations (near-synonyms, "
        "different formattings of the same word) do not count. Medium scores: "
        "supernodes for multiple candidates are present but weakly separated. "
        "High scores: clearly distinct supernode groups for each candidate.",
    ),
    (
        "surprising_supernode",
        "Surprising supernode",
        "A supernode represents a concept or association that is not directly "
        "suggested by the model's input and output. Medium scores: associations "
        "the model might plausibly make but that aren't a given. High scores: "
        "genuinely surprising associations.",
    ),
    (
        "complex_internal_computation",
        "Complex internal computation",
        "The supernodes reflect substantive internal computation — for example, "
        "multi-hop reasoning, or many supernodes that are not simply derivative "
        "of the input or output tokens. Medium scores: some non-trivial "
        "intermediate structure beyond direct lookup. High scores: clear multi-"
        "hop chains or a rich set of intermediate supernodes doing real work.",
    ),
    (
        "other_interesting_structure",
        "Other interesting structure",
        "A catch-all for generally interesting internal structure. If you "
        "notice something notable about the graph that doesn't fall into the "
        "other criteria, score it high here.",
    ),
]

LLM_SYSTEM = (
    "You are a mechanistic interpretability researcher evaluating attribution graphs "
    "from a language model. You score each graph on four dimensions of interestingness, "
    "each on a 1-10 scale, and give a short justification for each score.\n\n"
    "The four dimensions are:\n"
    + "\n".join(f"  {i+1}. {label} — {desc}" for i, (_, label, desc) in enumerate(LLM_CRITERIA))
    + "\n\n"
    "Scoring guidance (use the full 1-10 range):\n"
    "  1-2  = no evidence of this property\n"
    "  3-4  = weak or ambiguous hint\n"
    "  5-6  = plausible but not strongly supported\n"
    "  7-8  = clear, concrete instance worth noting\n"
    "  9-10 = textbook example, worth a writeup\n\n"
    "Score each dimension independently on its own merits. "
    "Do NOT inflate scores because the prompt is interesting; only the supernode "
    "structure matters. For each score, provide a one-sentence justification that "
    "names the specific supernodes or features driving your assessment."
)

# Each line uses {{ and }} so a single .format() call later produces literal { and }.
_CRITERIA_JSON_LINES = ",\n".join(
    f'  "{key}": {{{{"score": <integer 1-10>, "reason": "<one sentence naming the specific supernodes or features>"}}}}'
    for key, _, _ in LLM_CRITERIA
)

LLM_PROMPT_TEMPLATE = (
    'Prompt given to the model: "{prompt}"\n'
    'Model\'s predicted output: "{predicted}" — correct answer is: {correct_answer}\n'
    'Model confidence: {confidence:.1%} — {confidence_band}\n'
    'Supernode groups ({num_supernode_groups} total):\n'
    '{groups}\n\n'
    'Sample descriptions of the most influential individual nodes (by influence score):\n'
    '{clerp_samples}\n\n'
    'Score this graph on each of the four dimensions. Reply in this exact JSON format — '
    'every dimension is required, with both a score and a one-sentence reason:\n\n'
    '{{\n' + _CRITERIA_JSON_LINES + '\n}}'
)


def _empty_verdict(reason: str) -> dict:
    return {key: {"score": 0, "reason": reason} for key, _, _ in LLM_CRITERIA}


def call_llm_judge(prompt: str, predicted: str, correct_answer: str, confidence: float,
                   groups: list[str], clerps: list[str], num_supernode_groups: int,
                   client: OpenAI) -> dict:
    """Call the LLM judge. Returns dict mapping criterion_key -> {score, reason}."""
    groups_str = "\n".join(f"  - {g}" for g in groups) if groups else "  (no supernode groups)"
    clerps_str = "\n".join(f"  - {c}" for c in clerps) if clerps else "  (no descriptions available)"
    user_msg = LLM_PROMPT_TEMPLATE.format(
        prompt=prompt,
        predicted=predicted,
        correct_answer=correct_answer or "unknown",
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        num_supernode_groups=num_supernode_groups,
        groups=groups_str,
        clerp_samples=clerps_str,
    )
    try:
        response = client.chat.completions.create(
            model=LLM_JUDGE_MODEL,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return _empty_verdict("No JSON in response")
        parsed = json.loads(m.group(0))
        verdict: dict[str, dict] = {}
        for key, _, _ in LLM_CRITERIA:
            entry = parsed.get(key) or {}
            if isinstance(entry, (int, float)):
                entry = {"score": int(entry), "reason": ""}
            elif not isinstance(entry, dict):
                entry = {"score": 0, "reason": "Malformed entry"}
            score = entry.get("score", 0)
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = 0
            verdict[key] = {
                "score": max(0, min(10, score)),
                "reason": str(entry.get("reason", "")).strip(),
            }
        return verdict
    except Exception as e:
        return _empty_verdict(f"LLM error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    graphs_dir: Path,
    min_confidence: float,
    influence_threshold: float,
    use_llm: bool,
    ground_truth: Path | None,
    variants: list[str],
) -> None:
    # Build list of (slug, path) to process
    if ground_truth is not None:
        if not ground_truth.exists():
            print(f"ERROR: ground truth CSV not found: {ground_truth}", file=sys.stderr)
            sys.exit(1)
        gt_rows: dict[str, str] = {}
        with open(ground_truth, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                slug = row["slug"].strip()
                if slug and not re.search(r"-h\d+$", slug):
                    gt_rows[slug] = row.get("correct_answer", "").strip()
        candidates_paths: list[tuple[str, Path, str]] = []
        for slug, correct_answer in gt_rows.items():
            found = False
            for variant in variants:
                name = f"{slug}-v2-{variant}.json" if variant else f"{slug}.json"
                p = graphs_dir / name
                if p.exists():
                    candidates_paths.append((f"{slug} ({variant})", p, correct_answer))
                    found = True
                    break
            if not found:
                # Fallback: bare slug filename (no variant suffix)
                p = graphs_dir / f"{slug}.json"
                if p.exists():
                    candidates_paths.append((slug, p, correct_answer))
                else:
                    print(f"  SKIP {slug} — no matching file found")
        if not candidates_paths:
            print("No matching graph files found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(candidates_paths)} graph(s) from ground truth CSV")
    else:
        all_paths = sorted(graphs_dir.glob("*.json"))
        if not all_paths:
            print(f"No graph JSONs found in {graphs_dir}", file=sys.stderr)
            sys.exit(1)
        candidates_paths = [(p.stem, p, "") for p in all_paths]
        print(f"Found {len(candidates_paths)} graph(s) in {graphs_dir}")

    # ------------------------------------------------------------------
    # Step 1-2: Load and compute metrics
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    for label, path, correct_answer in candidates_paths:
        graph = load_graph(path)
        if graph is None:
            continue
        m = compute_metrics(graph, influence_threshold)
        m["slug"] = label
        m["correct_answer"] = correct_answer
        all_metrics.append(m)

    print(f"  Loaded {len(all_metrics)} valid graphs")

    # ------------------------------------------------------------------
    # Step 3: Filter to "clean" graphs — confident model output
    # ------------------------------------------------------------------
    clean = [m for m in all_metrics if m["model_confidence"] >= min_confidence]
    print(f"  {len(clean)} graphs pass confidence filter (≥{min_confidence:.1%})")

    if not clean:
        print("No clean graphs found. Lower --min_confidence.")
        sys.exit(0)

    candidates = clean
    print(f"  Running LLM judge on all {len(candidates)} clean graphs\n")

    # ------------------------------------------------------------------
    # Step 4: LLM judge (all clean graphs)
    # ------------------------------------------------------------------
    client: OpenAI | None = None
    if use_llm:
        client = OpenAI()

    results: list[dict] = []
    for i, m in enumerate(candidates):
        groups = m["supernode_names"]
        verdict: dict[str, dict] = _empty_verdict("LLM not run")

        if use_llm and client is not None:
            print(f"  [{i+1}/{len(candidates)}] LLM judging: {m['slug']} ...", end=" ", flush=True)
            verdict = call_llm_judge(
                m["prompt"], m["predicted"], m.get("correct_answer", ""),
                m["model_confidence"], groups, m["top_clerps"], m["num_supernode_groups"], client
            )
            score_summary = " ".join(f"{verdict[key]['score']}" for key, _, _ in LLM_CRITERIA)
            print(f"scores=[{score_summary}]")

        row = {
            "slug": m["slug"],
            "prompt": m["prompt"],
            "predicted": m["predicted"],
            "correct_answer": m.get("correct_answer", ""),
            "model_confidence": m["model_confidence"],
            "longest_path_length": m["longest_path_length"],
            "num_transcoder_nodes": m["num_transcoder_nodes"],
            "num_supernode_groups": m["num_supernode_groups"],
            "supernode_names": " | ".join(groups),
        }
        for key, _, _ in LLM_CRITERIA:
            row[f"score_{key}"] = verdict[key]["score"]
            row[f"reason_{key}"] = verdict[key]["reason"]
        results.append(row)

    # ------------------------------------------------------------------
    # Build per-category rankings
    # Each entry: (category_key, label, sort_key_fn, display_fn)
    # ------------------------------------------------------------------
    categories: list[tuple] = []
    for key, label, _ in LLM_CRITERIA:
        field = f"score_{key}"
        categories.append((
            key, label,
            (lambda r, fld=field: -r[fld]),         # sort by score desc
            (lambda r, fld=field: f"{r[fld]}/10"),  # display value
        ))
    # Structural rankings (no LLM reason field for these)
    categories.append((
        "longest_path", "Longest reasoning path",
        lambda r: -r["longest_path_length"],
        lambda r: f"{r['longest_path_length']} nodes",
    ))
    categories.append((
        "most_supernode_groups", "Most supernode groups",
        lambda r: -r["num_supernode_groups"],
        lambda r: f"{r['num_supernode_groups']} groups",
    ))
    categories.append((
        "lowest_confidence", "Lowest model confidence",
        lambda r: r["model_confidence"],
        lambda r: f"{r['model_confidence']:.1%}",
    ))

    # ------------------------------------------------------------------
    # Write CSV (one row per graph, all per-criterion scores)
    # ------------------------------------------------------------------
    csv_path = ARTIFACTS_DIR / "interesting_graphs.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV  -> {csv_path}")

    # ------------------------------------------------------------------
    # Write Markdown report — one ranked table per category
    # ------------------------------------------------------------------
    TOP_N = 5
    md_path = ARTIFACTS_DIR / "interesting_graphs.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Interesting Graph Exploration\n\n")
        f.write(f"Graphs scanned: {len(all_metrics)}  \n")
        f.write(f"Passed confidence filter (≥{min_confidence:.1%}): {len(clean)}  \n\n")
        f.write(
            "Each category below is an *independent* ranking. The same graph may appear "
            "at the top of multiple categories, or only one — that's the point.\n\n"
        )

        # Top graph per category — summary
        f.write("## Top Graph per Category\n\n")
        f.write("| Category | Top Graph | Value |\n")
        f.write("|----------|-----------|-------|\n")
        for _, label, sort_key, display in categories:
            top = sorted(results, key=sort_key)[0]
            f.write(f"| {label} | {top['slug']} | {display(top)} |\n")
        f.write("\n")

        # Per-category ranked sections
        for cat_key, label, sort_key, display in categories:
            ranked = sorted(results, key=sort_key)
            top = ranked[0]
            f.write(f"## {label}\n\n")
            f.write(f"**Top: {top['slug']}** — {display(top)}\n\n")
            f.write(f"- **Prompt**: {top['prompt']}\n")
            f.write(f"- **Predicted**: {top['predicted']} ({top['model_confidence']:.1%})")
            if top.get("correct_answer"):
                f.write(f" — correct: {top['correct_answer']}")
            f.write("\n")
            f.write(f"- **Longest reasoning path**: {top['longest_path_length']} nodes\n")
            f.write(f"- **Supernode groups**: {top['supernode_names'] or '(none)'}\n")
            reason_field = f"reason_{cat_key}"
            if reason_field in top and top[reason_field]:
                f.write(f"- **Why**: {top[reason_field]}\n")
            f.write("\n")
            f.write(f"Top {TOP_N}:\n\n")
            f.write("| Rank | Slug | Value | Reason |\n")
            f.write("|------|------|-------|--------|\n")
            for rank, r in enumerate(ranked[:TOP_N], 1):
                reason = r.get(f"reason_{cat_key}", "") or "—"
                # escape pipes in reasons so the table doesn't break
                reason = reason.replace("|", "\\|")
                f.write(f"| {rank} | {r['slug']} | {display(r)} | {reason} |\n")
            f.write("\n")

    print(f"Report -> {md_path}\n")

    # Terminal summary
    print("=" * 55)
    print(f"  Graphs scanned:          {len(all_metrics)}")
    print(f"  Passed confidence filter:{len(clean)}")
    print(f"  Categories ranked:       {len(categories)}")
    if use_llm:
        for key, label, _ in LLM_CRITERIA:
            top = max(results, key=lambda r, f=f"score_{key}": r[f])
            print(f"    {label:32s} top: {top['slug']} ({top[f'score_{key}']}/10)")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore attribution graphs for surprising/interesting structure."
    )
    parser.add_argument(
        "--graphs_dir", type=Path, default=TEST_GRAPHS_DIR,
        help="Directory containing processed graph JSONs (default: test_graphs/)",
    )
    parser.add_argument(
        "--min_confidence", type=float, default=0.05,
        help="Minimum model confidence to include a graph (default: 0.05)",
    )
    parser.add_argument(
        "--influence_threshold", type=float, default=0.1,
        help="Min influence for a node to count in the longest path (default: 0.1)",
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Skip the LLM judge and rank by path length only",
    )
    parser.add_argument(
        "--ground_truth", type=Path, default=None,
        help="CSV with a 'slug' column; only those graphs are processed (skips missing files)",
    )
    parser.add_argument(
        "--variants", type=str, default="a2",
        help="Comma-separated grouping variants to look for (default: a2)",
    )
    args = parser.parse_args()
    run(
        graphs_dir=args.graphs_dir,
        min_confidence=args.min_confidence,
        influence_threshold=args.influence_threshold,
        use_llm=not args.no_llm,
        ground_truth=args.ground_truth,
        variants=[v.strip() for v in args.variants.split(",")],
    )
