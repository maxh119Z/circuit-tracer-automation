"""
explore_interesting_graphs.py — Open-ended exploration to surface surprising graphs.

Pipeline
--------
1. Load all processed graph JSONs from test_graphs/.
2. For each graph compute structural metrics:
     model_confidence    : top logit probability  (high = clean, confident model)
     longest_path_length : longest path through high-influence transcoder nodes in the DAG
     num_transcoder_nodes: total transcoder features in the graph
3. Filter to "clean" graphs — model must be confident (above --min_confidence).
   Rationale (mentor): we don't want uncertain outputs.  The interesting thing is
   clean, confident behaviour whose *internal* nodes are surprising.
4. Rank clean graphs by longest_path_length (deeper reasoning chains = more to explain).
5. For the top --top_k candidates call an LLM judge:
     Input : prompt, predicted answer, supernode group names (or top clerp labels)
     Output: is_interesting (bool), reason (str), score (1-5)
6. Write artifacts/interesting_graphs.csv and artifacts/interesting_graphs.md.

Longest-path algorithm
----------------------
The attribution graph is a DAG: edges flow from embedding nodes (layer "E") through
transcoder feature nodes (layers 0..N) to logit nodes (layer N+1).
We find the longest path only through transcoder nodes whose influence ≥ threshold,
using topological order (nodes sorted by layer index).

Usage
-----
    python explore_interesting_graphs.py
    python explore_interesting_graphs.py --graphs_dir ../../test_graphs --top_k 20
    python explore_interesting_graphs.py --min_confidence 0.05 --influence_threshold 0.3 --no_llm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
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
            names.append(str(item[0]))
    return names


def get_top_clerps(graph: dict, k: int = 8) -> list[str]:
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
# LLM judge
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You are an interpretability researcher evaluating attribution graphs from a "
    "language model. Your job is to judge whether the internal feature groups "
    "that activated for a given prompt are *surprising* or *interesting* from an "
    "interpretability standpoint.\n\n"
    "Interesting means: the internal reasoning path is unexpected, reveals a non-obvious "
    "concept the model relies on, or shows the model doing something more complex than "
    "simple token matching. Uninteresting means: obvious features, surface-level pattern "
    "matching, or nothing unexpected given the input."
)

LLM_PROMPT_TEMPLATE = """\
Prompt given to the model: "{prompt}"
Model's predicted output: "{predicted}"
Model confidence: {confidence:.1%} — {confidence_band}

High-level feature groups that activated (supernode labels):
{groups}

Sample descriptions of the most influential individual nodes (by influence score):
{clerp_samples}

Question: Is this internal reasoning path surprising or interesting from an \
interpretability standpoint? Note: very high confidence on an obvious question \
is typically uninteresting. Moderate confidence with unexpected internal concepts \
is more interesting.

Reply in this exact JSON format:
{{
  "is_interesting": true or false,
  "score": <integer 1-5, where 5 = extremely interesting>,
  "reason": "<one or two sentences explaining why>"
}}"""

# Using Claude as the judge intentionally — descriptions and groupings were
# generated by GPT, so using a different model family here avoids same-model
# validation bias.
LLM_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def call_llm_judge(prompt: str, predicted: str, confidence: float,
                   groups: list[str], clerps: list[str], client) -> dict:
    """Call Claude to judge whether the graph is interesting. Returns parsed JSON."""
    groups_str = "\n".join(f"  - {g}" for g in groups) if groups else "  (no supernode groups)"
    clerps_str = "\n".join(f"  - {c}" for c in clerps) if clerps else "  (no descriptions available)"
    user_msg = LLM_PROMPT_TEMPLATE.format(
        prompt=prompt,
        predicted=predicted,
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        groups=groups_str,
        clerp_samples=clerps_str,
    )
    try:
        response = client.messages.create(
            model=LLM_JUDGE_MODEL,
            max_tokens=256,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        return {"is_interesting": False, "score": 0, "reason": f"LLM error: {e}"}
    return {"is_interesting": False, "score": 0, "reason": "No JSON in response"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    graphs_dir: Path,
    top_k: int,
    min_confidence: float,
    influence_threshold: float,
    use_llm: bool,
) -> None:
    graph_paths = sorted(graphs_dir.glob("*.json"))
    if not graph_paths:
        print(f"No graph JSONs found in {graphs_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(graph_paths)} graph(s) in {graphs_dir}")

    # ------------------------------------------------------------------
    # Step 1-2: Load and compute metrics
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    for path in graph_paths:
        graph = load_graph(path)
        if graph is None:
            continue
        m = compute_metrics(graph, influence_threshold)
        m["slug"] = path.stem
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

    # ------------------------------------------------------------------
    # Step 4: Rank by longest path (deeper reasoning = more interesting)
    # ------------------------------------------------------------------
    ranked = sorted(clean, key=lambda m: m["longest_path_length"], reverse=True)
    candidates = ranked[:top_k]
    print(f"  Top {len(candidates)} candidates by longest path\n")

    # ------------------------------------------------------------------
    # Step 5: LLM judge
    # ------------------------------------------------------------------
    if use_llm:
        try:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
        except ImportError:
            print("WARNING: anthropic package not installed. Skipping LLM judge.")
            use_llm = False

    results: list[dict] = []
    for i, m in enumerate(candidates):
        groups = m["supernode_names"]
        llm_score, llm_interesting, llm_reason = 0, None, ""

        if use_llm:
            print(f"  [{i+1}/{len(candidates)}] LLM judging: {m['slug']} ...", end=" ", flush=True)
            verdict = call_llm_judge(
                m["prompt"], m["predicted"], m["model_confidence"],
                groups, m["top_clerps"], client
            )
            llm_score = verdict.get("score", 0)
            llm_interesting = verdict.get("is_interesting", False)
            llm_reason = verdict.get("reason", "")
            print(f"score={llm_score}, interesting={llm_interesting}")

        results.append({
            "slug": m["slug"],
            "prompt": m["prompt"],
            "predicted": m["predicted"],
            "model_confidence": m["model_confidence"],
            "longest_path_length": m["longest_path_length"],
            "num_transcoder_nodes": m["num_transcoder_nodes"],
            "num_supernode_groups": m["num_supernode_groups"],
            "llm_score": llm_score,
            "llm_interesting": llm_interesting,
            "llm_reason": llm_reason,
            "supernode_names": " | ".join(groups),
        })

    # Sort final results: LLM score first (if available), then path length
    results.sort(key=lambda r: (-(r["llm_score"] or 0), -r["longest_path_length"]))

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    csv_path = ARTIFACTS_DIR / "interesting_graphs.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV  -> {csv_path}")

    # ------------------------------------------------------------------
    # Write Markdown report
    # ------------------------------------------------------------------
    md_path = ARTIFACTS_DIR / "interesting_graphs.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Interesting Graph Exploration\n\n")
        f.write(f"Graphs scanned: {len(all_metrics)}  \n")
        f.write(f"Passed confidence filter (≥{min_confidence:.1%}): {len(clean)}  \n")
        f.write(f"Influence threshold for path: {influence_threshold}  \n")
        f.write(f"Top-K candidates: {len(candidates)}\n\n")

        if use_llm:
            n_interesting = sum(1 for r in results if r["llm_interesting"])
            f.write(f"LLM judge: {n_interesting}/{len(results)} flagged as interesting\n\n")

        # Summary table
        f.write("## Ranked Candidates\n\n")
        f.write("| Slug | Confidence | Longest path | LLM score | Interesting? |\n")
        f.write("|------|------------|:------------:|:---------:|:------------:|\n")
        for r in results:
            tick = ("✓" if r["llm_interesting"] else "✗") if r["llm_interesting"] is not None else "—"
            f.write(
                f"| {r['slug']} | {r['model_confidence']:.1%} "
                f"| {r['longest_path_length']} "
                f"| {r['llm_score'] or '—'} "
                f"| {tick} |\n"
            )
        f.write("\n")

        # Spotlight: top interesting cases
        spotlight = [r for r in results if r["llm_interesting"]] or results[:5]
        f.write("## Spotlight: Most Interesting Graphs\n\n")
        for r in spotlight:
            f.write(f"### {r['slug']}\n")
            f.write(f"- **Prompt**: {r['prompt']}\n")
            f.write(f"- **Predicted**: {r['predicted']} ({r['model_confidence']:.1%})\n")
            f.write(f"- **Longest reasoning path**: {r['longest_path_length']} nodes\n")
            f.write(f"- **Supernode groups**: {r['supernode_names'] or '(none)'}\n")
            if r["llm_reason"]:
                f.write(f"- **Why interesting**: {r['llm_reason']}\n")
            f.write("\n")

    print(f"Report -> {md_path}\n")

    # Terminal summary
    print("=" * 55)
    print(f"  Graphs scanned:          {len(all_metrics)}")
    print(f"  Passed confidence filter:{len(clean)}")
    print(f"  Longest path (max):      {results[0]['longest_path_length'] if results else 0}")
    if use_llm:
        n_interesting = sum(1 for r in results if r["llm_interesting"])
        print(f"  LLM flagged interesting: {n_interesting}/{len(results)}")
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
        "--top_k", type=int, default=20,
        help="Number of top candidates to send to the LLM judge (default: 20)",
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
    args = parser.parse_args()
    run(
        graphs_dir=args.graphs_dir,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        influence_threshold=args.influence_threshold,
        use_llm=not args.no_llm,
    )
