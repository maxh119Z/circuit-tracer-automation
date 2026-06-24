"""
explore_interesting_graphs.py — Open-ended exploration to surface surprising graphs.
Writes analysis/results/interesting_graphs.csv and .md.

Usage
-----
    python analysis/explore_interesting_graphs.py --ground_truth ../prompts/ground_truth_wikipedia.csv --variants a2
    python analysis/explore_interesting_graphs.py --min_confidence 0.05 --influence_threshold 0.3 --no_llm
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

from openai import OpenAI

# On Windows the default console encoding is cp1252, which crashes on the ≥, ->,
# and accented characters printed below / produced by the LLM. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs_new"
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


# Fields written as numbers. When we read an existing CSV back in to merge, the
# csv module yields every value as a string, so these must be coerced back to
# numeric types or the ranking sort keys (which negate / compare them) break.
def _numeric_fields() -> tuple[set[str], set[str]]:
    ints = {"longest_path_length", "num_transcoder_nodes", "num_supernode_groups",
            "divergence_score"}
    ints |= {f"score_{key}" for key, _, _ in LLM_CRITERIA}
    floats = {"model_confidence"}
    return ints, floats


def coerce_numeric(row: dict) -> dict:
    """Coerce known numeric fields of a row read from CSV back to int/float."""
    ints, floats = _numeric_fields()
    for k in ints:
        if k in row and row[k] != "":
            try:
                row[k] = int(float(row[k]))
            except (TypeError, ValueError):
                row[k] = 0
    for k in floats:
        if k in row and row[k] != "":
            try:
                row[k] = float(row[k])
            except (TypeError, ValueError):
                row[k] = 0.0
    return row


def merge_into_existing(new_rows: list[dict], csv_path: Path) -> tuple[list[dict], list[str]]:
    """
    Splice freshly-computed rows into an existing interesting_graphs.csv, keyed by
    slug. Existing rows keep their order; updated slugs are overwritten in place;
    brand-new slugs are appended. Returns (merged_rows, fieldnames) where fieldnames
    is the union of old and new columns (old order first).
    """
    if not csv_path.exists():
        print(f"  --update: no existing CSV at {csv_path}; writing fresh.")
        return new_rows, list(new_rows[0].keys()) if new_rows else []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = list(reader.fieldnames or [])
        existing = [coerce_numeric(dict(r)) for r in reader]

    new_by_slug = {r["slug"]: r for r in new_rows}
    merged: list[dict] = []
    seen: set[str] = set()
    replaced = 0
    for r in existing:
        slug = r.get("slug", "")
        if slug in new_by_slug:
            merged.append(new_by_slug[slug])
            replaced += 1
        else:
            merged.append(r)
        seen.add(slug)
    appended = 0
    for r in new_rows:
        if r["slug"] not in seen:
            merged.append(r)
            appended += 1

    # Union of columns: keep the existing order, then append any new keys.
    fieldnames = list(existing_fieldnames)
    for r in new_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    print(f"  --update: {replaced} row(s) updated, {appended} new row(s) appended "
          f"→ {len(merged)} total.")
    return merged, fieldnames


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

# Override with EXPLORE_JUDGE_MODEL env var or --model on the CLI.
LLM_JUDGE_MODEL = os.environ.get("EXPLORE_JUDGE_MODEL", "gpt-5.4")

# Each LLM criterion: (json_key, human label, description for the prompt).
#
# The taxonomy is derived from the recurring properties that make the examples
# in Anthropic's attribution-graph papers ("On the Biology of a Large Language
# Model" / "Circuit Tracing", 2025) worth highlighting. Each criterion below
# names the paper example(s) it generalizes, and is written to be judgeable from
# what we actually have per graph: prompt, predicted output, supernode group
# names, and the most-influential feature descriptions.
LLM_CRITERIA: list[tuple[str, str, str]] = [
    # NOTE: the former "hidden_intermediate_concept" and "multi_hop_reasoning"
    # criteria were merged into a single dimension — MERGED_HIDDEN_MULTIHOP, defined
    # below and prepended to this list. Fresh runs therefore score the merged
    # dimension directly; there is no longer a separate hidden/multi-hop column.
    (
        "parallel_or_competing",
        "Parallel / competing computation",
        "The model computes several candidate outputs or competing hypotheses at "
        "once, OR combines independent pathways into the answer (e.g. a rough "
        "estimate pathway plus an exact-lookup pathway). (Paper: addition's "
        "parallel approximate+precise routes; planning multiple candidate words.) "
        "Near-synonyms or reformattings of the same token do NOT count. Medium: "
        "multiple candidates present but weakly separated. High: clearly distinct "
        "competing or complementary pathways.",
    ),
    (
        "abstraction_beyond_tokens",
        "Abstraction beyond surface tokens",
        "The most influential supernodes operate on abstract, semantic, or "
        "relational concepts decoupled from the literal surface tokens — "
        "category-level, language-independent, or role-based features — rather "
        "than echoing specific prompt words. (Paper: language-independent "
        "'meaning' features in multilingual circuits.) Medium: some abstraction "
        "beyond the tokens. High: core computation is clearly in an abstract "
        "concept space, not in the surface strings.",
    ),
    (
        "unexpected_mechanism",
        "Unexpected mechanism (headline)",
        "Taken as a whole, does the internal computation DIVERGE from what a "
        "knowledgeable person would predict just from the prompt and the predicted "
        "output? This is the open-ended 'reveals computation different from what "
        "one might expect' signal. (Paper: chain-of-thought unfaithfulness; "
        "mechanisms that contradict the model's own account.) Score LOW when the "
        "supernodes are exactly the obvious ones a person would guess. Score HIGH "
        "when the route to the answer is structurally different from the naive "
        "expectation. Judge the mechanism, not whether the prompt topic is "
        "interesting.",
    ),
    (
        "other_interesting_structure",
        "Other interesting structure",
        "A catch-all for notable internal structure not covered above. If you "
        "notice something genuinely worth a researcher's attention, score it high "
        "here and name it specifically.",
    ),
]

# Merged criterion: the former "hidden_intermediate_concept" and "multi_hop_reasoning"
# criteria, permanently combined into one dimension. It is prepended to LLM_CRITERIA
# below, so it is a first-class criterion scored on every run. The
# --merge-hidden-multihop rescore mode reuses it to convert OLDER reports that still
# carry the two separate columns. Its description is the union of the two source
# prompts so the judge rewards EITHER property.
MERGED_HIDDEN_MULTIHOP: tuple[str, str, str] = (
    "hidden_or_multihop",
    "Hidden intermediate concept / multi-hop reasoning",
    "Score high if EITHER of these holds (they are two faces of the same thing — "
    "internal computation that goes beyond a single surface lookup):\n"
    "  (a) HIDDEN INTERMEDIATE CONCEPT — a supernode represents a concept that is "
    "load-bearing for the answer but appears in NEITHER the prompt NOR the predicted "
    "output; the model computes it internally as a stepping stone (Paper: implicit "
    "'Texas' in Dallas->Austin; the Apollo program behind 'space'). Trivial "
    "restatements of a prompt/output token do not count.\n"
    "  (b) MULTI-HOP REASONING — the supernodes form a genuine chain (input concept "
    "-> one or more intermediate concepts -> output), i.e. composed reasoning steps "
    "rather than a single direct association (Paper: multi-step factual reasoning).\n"
    "Medium: one plausible unstated association OR one clear intermediate step beyond "
    "lookup. High: a specific, genuinely surprising hidden concept clearly doing real "
    "work, OR a multi-link chain where each hop is a distinct, necessary concept.",
)

# Columns retired (and re-scored into the merged column) when converting an OLDER
# report via --merge-hidden-multihop.
MERGE_REPLACES: list[str] = ["hidden_intermediate_concept", "multi_hop_reasoning"]

# Prepend the merged dimension so it is the first criterion (it replaces the two
# removed above). Done after the definition to avoid a forward reference, and before
# LLM_SYSTEM / the JSON template below consume LLM_CRITERIA.
LLM_CRITERIA.insert(0, MERGED_HIDDEN_MULTIHOP)

LLM_SYSTEM = (
    "You are a mechanistic interpretability researcher evaluating attribution graphs "
    "from a language model. You score each graph on several dimensions of interestingness, "
    "each on a 1-10 scale, and give a short justification for each score.\n\n"
    "The dimensions are:\n"
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
    'Score this graph on each of the five dimensions. Reply in this exact JSON format — '
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


def judge_single_criterion(
    criterion: tuple[str, str, str],
    prompt: str, predicted: str, correct_answer: str, confidence: float,
    groups: list[str], clerps: list[str], num_supernode_groups: int,
    client: OpenAI,
) -> dict:
    """Score ONE criterion in isolation. Returns {"score": int, "reason": str}.

    Used by the rescore path so re-judging a single (e.g. merged) dimension costs
    one short call per graph and leaves every other column untouched.
    """
    _key, label, desc = criterion
    system = (
        "You are a mechanistic interpretability researcher evaluating attribution graphs "
        "from a language model. Score the single dimension below on a 1-10 scale and give "
        "a one-sentence justification naming the specific supernodes or features.\n\n"
        f"Dimension — {label}: {desc}\n\n"
        "Scoring guidance (use the full 1-10 range):\n"
        "  1-2  = no evidence of this property\n"
        "  3-4  = weak or ambiguous hint\n"
        "  5-6  = plausible but not strongly supported\n"
        "  7-8  = clear, concrete instance worth noting\n"
        "  9-10 = textbook example, worth a writeup\n\n"
        "Do NOT inflate the score because the prompt topic is interesting; only the "
        "supernode structure matters. Reply as JSON: "
        '{"score": <integer 1-10>, "reason": "<one sentence naming the specific supernodes>"}.'
    )
    groups_str = "\n".join(f"  - {g}" for g in groups) if groups else "  (no supernode groups)"
    clerps_str = "\n".join(f"  - {c}" for c in clerps) if clerps else "  (no descriptions available)"
    user = (
        f'Prompt given to the model: "{prompt}"\n'
        f'Model\'s predicted output: "{predicted}" — correct answer is: {correct_answer or "unknown"}\n'
        f'Model confidence: {confidence:.1%} — {confidence_band(confidence)}\n'
        f'Supernode groups ({num_supernode_groups} total):\n{groups_str}\n\n'
        f'Sample descriptions of the most influential individual nodes:\n{clerps_str}\n\n'
        'Score this single dimension. Reply only as the JSON object described.'
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_JUDGE_MODEL,
            max_completion_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return {"score": 0, "reason": "No JSON in response"}
        parsed = json.loads(m.group(0))
        try:
            score = max(0, min(10, int(parsed.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        return {"score": score, "reason": str(parsed.get("reason", "")).strip()}
    except Exception as e:
        return {"score": 0, "reason": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Predict-then-reveal divergence — the rigorous "unexpected mechanism" signal
#
# Single-shot judging asks the model to rate surprise while already seeing the
# answer, which anchors it. Instead we do two calls:
#   1. PREDICT: given ONLY the prompt + predicted output, list the supernodes a
#      knowledgeable person would expect the graph to contain.
#   2. REVEAL:  show the ACTUAL supernodes and score how much (and how
#      interestingly) they diverge from that prediction.
# A graph scores high only when the real mechanism is structurally different
# from the naive expectation — exactly "computation different from what one
# might expect".
# ---------------------------------------------------------------------------

DIVERGENCE_PREDICT_SYSTEM = (
    "You are a mechanistic interpretability researcher. Given only a prompt and the "
    "model's predicted next token, predict what internal 'supernodes' (named concept "
    "groups) the model's attribution graph most likely contains on the way to that "
    "output. Think about the obvious, expected route a competent person would guess: "
    "which input concepts, intermediate concepts, and output-promoting concepts. "
    "List 4-8 short supernode names, comma-separated, no commentary."
)

DIVERGENCE_SCORE_SYSTEM = (
    "You are a mechanistic interpretability researcher comparing a PREDICTED set of "
    "supernodes (what a knowledgeable person expected an attribution graph to contain) "
    "against the ACTUAL supernodes found in the graph. Score, 1-10, how much the actual "
    "mechanism DIVERGES from the expectation in an interesting way — i.e. reveals "
    "internal computation different from what one would expect.\n"
    "  1-2  = actual supernodes are essentially the predicted ones\n"
    "  3-4  = minor extra detail, same overall route\n"
    "  5-6  = a notable unexpected concept or pathway, but the gist matches\n"
    "  7-8  = the real route is clearly structured differently than expected\n"
    "  9-10 = a genuinely surprising mechanism worth a writeup\n"
    "Reward hidden intermediate concepts, competing/suppression pathways, and "
    "abstraction that a person would not have predicted. Do NOT reward mere noise, "
    "grammatical/structural groups, or near-synonyms of the expected concepts. "
    "Reply as JSON: {\"score\": <1-10>, \"surprising\": \"<the single most unexpected "
    "actual supernode or pathway, or 'none'>\", \"reason\": \"<one sentence>\"}."
)


def predict_expected_mechanism(
    prompt: str, predicted: str, correct_answer: str, client: OpenAI, model: str,
) -> str:
    """Call 1 — list the supernodes a person would expect, given only prompt+output."""
    user = (
        f'Prompt: "{prompt}"\n'
        f'Predicted next token: "{predicted}"'
        + (f' (correct answer: {correct_answer})' if correct_answer else '')
        + "\n\nList the expected supernode names."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=512,
            messages=[
                {"role": "system", "content": DIVERGENCE_PREDICT_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(prediction failed: {e})"


def score_divergence(
    prompt: str, predicted: str, expected: str,
    actual_supernodes: list[str], clerps: list[str],
    client: OpenAI, model: str,
) -> dict:
    """Call 2 — reveal actual supernodes, score divergence from the prediction."""
    actual_str = ", ".join(actual_supernodes) if actual_supernodes else "(none)"
    clerps_str = "\n".join(f"  - {c}" for c in clerps) if clerps else "  (none)"
    user = (
        f'Prompt: "{prompt}"\n'
        f'Predicted next token: "{predicted}"\n\n'
        f'PREDICTED supernodes (what a person expected):\n  {expected}\n\n'
        f'ACTUAL supernodes in the graph:\n  {actual_str}\n\n'
        f'Most influential individual features:\n{clerps_str}\n\n'
        'Score the divergence as instructed.'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=512,
            messages=[
                {"role": "system", "content": DIVERGENCE_SCORE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return {"score": 0, "surprising": "", "reason": "No JSON in response"}
        parsed = json.loads(m.group(0))
        try:
            score = max(0, min(10, int(parsed.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        return {
            "score": score,
            "surprising": str(parsed.get("surprising", "")).strip(),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except Exception as e:
        return {"score": 0, "surprising": "", "reason": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Report writing (shared by full runs and rescore runs)
# ---------------------------------------------------------------------------

# Human labels for every score_<key> column we might encounter, including the
# merged dimension. Lets write_report build rankings from whatever score columns
# are present in the data, so it adapts after a category merge.
def _label_registry() -> dict[str, str]:
    reg = {key: label for key, label, _ in LLM_CRITERIA}
    reg[MERGED_HIDDEN_MULTIHOP[0]] = MERGED_HIDDEN_MULTIHOP[1]
    return reg


def write_report(
    results: list[dict],
    csv_fieldnames: list[str],
    scanned_count: int,
    passed_count: int,
    min_confidence: float,
    csv_path: Path,
    md_path: Path,
    use_llm: bool,
) -> None:
    """Write the CSV and the per-category ranked Markdown report.

    Rankable LLM categories are derived from the ``score_<key>`` columns actually
    present in ``csv_fieldnames`` (not from LLM_CRITERIA), so a merged/rescored
    report ranks exactly the columns it contains.
    """
    labels = _label_registry()

    # LLM-score categories, in column order.
    categories: list[tuple] = []
    for col in csv_fieldnames:
        if not col.startswith("score_"):
            continue
        key = col[len("score_"):]
        label = labels.get(key, key.replace("_", " ").capitalize())
        categories.append((
            key, label,
            (lambda r, fld=col: -int(r.get(fld) or 0)),
            (lambda r, fld=col: f"{int(r.get(fld) or 0)}/10"),
        ))

    # Structural rankings.
    categories.append((
        "longest_path", "Longest reasoning path",
        lambda r: -int(r.get("longest_path_length") or 0),
        lambda r: f"{int(r.get('longest_path_length') or 0)} nodes",
    ))
    categories.append((
        "most_supernode_groups", "Most supernode groups",
        lambda r: -int(r.get("num_supernode_groups") or 0),
        lambda r: f"{int(r.get('num_supernode_groups') or 0)} groups",
    ))
    categories.append((
        "lowest_confidence", "Lowest model confidence",
        lambda r: float(r.get("model_confidence") or 0.0),
        lambda r: f"{float(r.get('model_confidence') or 0.0):.1%}",
    ))
    has_divergence = any(
        str(r.get("divergence_score", "")).strip() not in ("", "None") for r in results
    )
    if has_divergence:
        categories.append((
            "divergence", "Predict-then-reveal divergence",
            lambda r: -int(r.get("divergence_score") or 0),
            lambda r: f"{int(r.get('divergence_score') or 0)}/10",
        ))

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV  -> {csv_path}")

    # Markdown
    TOP_N = 5
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Interesting Graph Exploration\n\n")
        f.write(f"Graphs scanned: {scanned_count}  \n")
        f.write(f"Passed confidence filter (≥{min_confidence:.1%}): {passed_count}  \n\n")
        f.write(
            "Each category below is an *independent* ranking. The same graph may appear "
            "at the top of multiple categories, or only one — that's the point.\n\n"
        )

        f.write("## Top Graph per Category\n\n")
        f.write("| Category | Top Graph | Value |\n")
        f.write("|----------|-----------|-------|\n")
        for _, label, sort_key, display in categories:
            top = sorted(results, key=sort_key)[0]
            f.write(f"| {label} | {top['slug']} | {display(top)} |\n")
        f.write("\n")

        for cat_key, label, sort_key, display in categories:
            ranked = sorted(results, key=sort_key)
            top = ranked[0]
            f.write(f"## {label}\n\n")
            f.write(f"**Top: {top['slug']}** — {display(top)}\n\n")
            f.write(f"- **Prompt**: {top['prompt']}\n")
            f.write(f"- **Predicted**: {top['predicted']} ({float(top.get('model_confidence') or 0.0):.1%})")
            if top.get("correct_answer"):
                f.write(f" — correct: {top['correct_answer']}")
            f.write("\n")
            f.write(f"- **Longest reasoning path**: {int(top.get('longest_path_length') or 0)} nodes\n")
            f.write(f"- **Supernode groups**: {top.get('supernode_names') or '(none)'}\n")
            reason_field = f"reason_{cat_key}"
            if reason_field in top and top[reason_field]:
                f.write(f"- **Why**: {top[reason_field]}\n")
            f.write("\n")
            f.write(f"Top {TOP_N}:\n\n")
            f.write("| Rank | Slug | Value | Reason |\n")
            f.write("|------|------|-------|--------|\n")
            for rank, r in enumerate(ranked[:TOP_N], 1):
                reason = (r.get(f"reason_{cat_key}", "") or "—").replace("|", "\\|")
                f.write(f"| {rank} | {r['slug']} | {display(r)} | {reason} |\n")
            f.write("\n")

    print(f"Report -> {md_path}\n")

    print("=" * 55)
    print(f"  Graphs scanned:          {scanned_count}")
    print(f"  Passed confidence filter:{passed_count}")
    print(f"  Categories ranked:       {len(categories)}")
    if use_llm:
        for cat_key, label, sort_key, _disp in categories:
            top = sorted(results, key=sort_key)[0]
            print(f"    {label:42s} top: {top['slug']}")
    print("=" * 55)


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
    divergence: bool = False,
    only_slugs: set[str] | None = None,
    update: bool = False,
    results_name: str = "interesting_graphs",
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
                    label = f"{slug} ({variant})" if variant else slug
                    candidates_paths.append((label, p, correct_answer))
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
        all_paths = sorted(p for p in graphs_dir.glob("*.json") if p.name != "graph-metadata.json")
        if not all_paths:
            print(f"No graph JSONs found in {graphs_dir}", file=sys.stderr)
            sys.exit(1)
        candidates_paths = [(p.stem, p, "") for p in all_paths]
        print(f"Found {len(candidates_paths)} graph(s) in {graphs_dir}")

    # Restrict to specific slugs (for incremental --update runs). Match against
    # both the base slug (label minus any " (variant)" suffix) and the filename
    # stem, so it works in ground-truth, variant, and glob modes alike.
    if only_slugs:
        candidates_paths = [
            c for c in candidates_paths
            if c[0].split(" (")[0] in only_slugs or Path(c[1]).stem in only_slugs
        ]
        if not candidates_paths:
            print(f"No graphs matched --slugs {sorted(only_slugs)}", file=sys.stderr)
            sys.exit(1)
        print(f"  Restricted to {len(candidates_paths)} graph(s) via --slugs")

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

        # Predict-then-reveal divergence (two extra calls per graph).
        div: dict = {"score": 0, "surprising": "", "reason": "divergence not run"}
        expected_mechanism = ""
        if use_llm and client is not None and divergence:
            expected_mechanism = predict_expected_mechanism(
                m["prompt"], m["predicted"], m.get("correct_answer", ""),
                client, LLM_JUDGE_MODEL,
            )
            div = score_divergence(
                m["prompt"], m["predicted"], expected_mechanism,
                groups, m["top_clerps"], client, LLM_JUDGE_MODEL,
            )
            print(f"      divergence={div['score']}/10  surprising={div['surprising'][:60]!r}")

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
        if divergence:
            row["expected_mechanism"] = expected_mechanism
            row["divergence_score"] = div["score"]
            row["divergence_surprising"] = div["surprising"]
            row["divergence_reason"] = div["reason"]
            # Alias so the markdown ranking (which looks up reason_<cat_key>) can
            # show a justification for the divergence category.
            row["reason_divergence"] = div["surprising"] or div["reason"]
        results.append(row)

    # ------------------------------------------------------------------
    # Merge into an existing report (incremental --update), or report on just
    # the freshly-scored graphs. scanned_count / passed_count drive the report
    # header; in update mode they reflect the full merged set.
    # ------------------------------------------------------------------
    csv_path = ARTIFACTS_DIR / f"{results_name}.csv"
    md_path = ARTIFACTS_DIR / f"{results_name}.md"
    scanned_count = len(all_metrics)
    passed_count = len(clean)
    csv_fieldnames = list(results[0].keys())
    if update:
        results, csv_fieldnames = merge_into_existing(results, csv_path)
        scanned_count = len(results)
        passed_count = len(results)

    write_report(results, csv_fieldnames, scanned_count, passed_count,
                 min_confidence, csv_path, md_path, use_llm)


def _locate_graph(graphs_dir: Path, slug: str, variants: list[str]) -> Path | None:
    """Find the graph JSON for a report slug.

    Reports made with a matched grouping variant store slugs as "base (variant)"
    (e.g. "ancient-greek-s23-9 (a2)"); the on-disk file is "base-v2-variant.json".
    Resolve that form first (preferring the grouped variant file), then fall back
    to a bare-slug filename and the --variants patterns.
    """
    m = re.match(r"^(.*) \(([^)]+)\)$", slug)
    if m:
        base, var = m.group(1), m.group(2)
        for cand in (graphs_dir / f"{base}-v2-{var}.json", graphs_dir / f"{base}.json"):
            if cand.exists():
                return cand

    p = graphs_dir / f"{slug}.json"
    if p.exists():
        return p
    for variant in variants:
        if not variant:
            continue
        p = graphs_dir / f"{slug}-v2-{variant}.json"
        if p.exists():
            return p
    return None


def rescore_merged(
    graphs_dir: Path,
    results_name: str,
    influence_threshold: float,
    min_confidence: float,
    use_llm: bool,
    variants: list[str],
) -> None:
    """Re-score ONLY the merged hidden/multi-hop dimension over an existing report.

    Reads <results_name>.csv, re-judges just the combined criterion for each graph
    (one short LLM call each), drops the two constituent score/reason columns, and
    rewrites <results_name>.csv / .md. Every other column is preserved untouched.
    """
    csv_path = ARTIFACTS_DIR / f"{results_name}.csv"
    md_path = ARTIFACTS_DIR / f"{results_name}.md"
    if not csv_path.exists():
        print(f"ERROR: no existing report to rescore at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [coerce_numeric(dict(r)) for r in reader]

    merged_key, merged_label, _ = MERGED_HIDDEN_MULTIHOP
    score_col, reason_col = f"score_{merged_key}", f"reason_{merged_key}"
    print(f"Rescoring '{merged_label}' over {len(rows)} graph(s) in {csv_path.name}")

    client = OpenAI() if use_llm else None
    scored = missing = 0
    for i, row in enumerate(rows):
        slug = row.get("slug", "")
        path = _locate_graph(graphs_dir, slug, variants)
        if path is None:
            missing += 1
            row[score_col] = 0
            row[reason_col] = "graph file not found"
            print(f"  [{i+1}/{len(rows)}] {slug} — SKIP (no graph file)")
            continue
        graph = load_graph(path)
        if graph is None:
            missing += 1
            row[score_col] = 0
            row[reason_col] = "graph failed to load"
            continue
        m = compute_metrics(graph, influence_threshold)
        if use_llm and client is not None:
            verdict = judge_single_criterion(
                MERGED_HIDDEN_MULTIHOP, m["prompt"], m["predicted"],
                row.get("correct_answer", ""), m["model_confidence"],
                m["supernode_names"], m["top_clerps"], m["num_supernode_groups"], client,
            )
            print(f"  [{i+1}/{len(rows)}] {slug} -> {verdict['score']}/10")
        else:
            verdict = {"score": 0, "reason": "LLM not run"}
        row[score_col] = verdict["score"]
        row[reason_col] = verdict["reason"]
        scored += 1

    # Retire the constituent columns and slot the merged columns into their place.
    drop = {f"score_{k}" for k in MERGE_REPLACES} | {f"reason_{k}" for k in MERGE_REPLACES}
    new_fieldnames = [c for c in fieldnames if c not in drop]
    if score_col not in new_fieldnames:
        insert_at = len(new_fieldnames)
        for idx, c in enumerate(fieldnames):
            if c in drop:
                insert_at = sum(1 for x in fieldnames[:idx] if x not in drop)
                break
        new_fieldnames[insert_at:insert_at] = [score_col, reason_col]
    for row in rows:
        for c in drop:
            row.pop(c, None)

    write_report(rows, new_fieldnames, len(rows), len(rows),
                 min_confidence, csv_path, md_path, use_llm)
    print(f"Rescored {scored} graph(s); {missing} missing. Merged column: {score_col}")


def combine_reports(report_names: list[str], results_name: str, min_confidence: float) -> None:
    """Concatenate several existing reports into one, then regenerate the rankings.

    Pure data merge — no graphs are loaded and no LLM is called. Rows are taken as-is
    from each <name>.csv under analysis/results/, deduplicated by slug (first wins),
    with the column set unioned (first report's order, then any extra columns). The
    combined CSV/MD is written to <results_name>. Inputs should share a schema (e.g.
    all rescored to the merged category) or the rankings will have sparsely-filled
    columns.
    """
    out_csv = ARTIFACTS_DIR / f"{results_name}.csv"
    out_md = ARTIFACTS_DIR / f"{results_name}.md"

    combined: list[dict] = []
    seen: set[str] = set()
    fieldnames: list[str] = []
    dups = 0
    for name in report_names:
        p = ARTIFACTS_DIR / f"{name}.csv"
        if not p.exists():
            print(f"ERROR: report not found: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for c in (reader.fieldnames or []):
                if c not in fieldnames:
                    fieldnames.append(c)
            n = 0
            for r in reader:
                row = coerce_numeric(dict(r))
                slug = row.get("slug", "")
                if slug in seen:
                    dups += 1
                    continue
                seen.add(slug)
                combined.append(row)
                n += 1
        print(f"  + {name}.csv: {n} row(s)")
    if dups:
        print(f"  ({dups} duplicate slug(s) skipped)")
    if not combined:
        print("No rows to combine.", file=sys.stderr)
        sys.exit(1)
    print(f"Combined total: {len(combined)} row(s)")

    write_report(combined, fieldnames, len(combined), len(combined),
                 min_confidence, out_csv, out_md, use_llm=True)


def drop_category(category_key: str, results_name: str, min_confidence: float) -> None:
    """Remove one LLM category's score/reason columns from an existing report and
    regenerate its rankings. Pure data edit — no graphs loaded, no LLM call.
    """
    csv_path = ARTIFACTS_DIR / f"{results_name}.csv"
    md_path = ARTIFACTS_DIR / f"{results_name}.md"
    if not csv_path.exists():
        print(f"ERROR: report not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [coerce_numeric(dict(r)) for r in reader]

    drop = {f"score_{category_key}", f"reason_{category_key}"}
    present = [c for c in drop if c in fieldnames]
    if not present:
        print(f"  {results_name}: no '{category_key}' columns to drop — left unchanged.")
        return
    new_fieldnames = [c for c in fieldnames if c not in drop]
    for row in rows:
        for c in drop:
            row.pop(c, None)

    write_report(rows, new_fieldnames, len(rows), len(rows),
                 min_confidence, csv_path, md_path, use_llm=True)
    print(f"  {results_name}: dropped {present} over {len(rows)} row(s).")


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
    parser.add_argument(
        "--divergence", action="store_true",
        help="Run the predict-then-reveal pass: predict expected supernodes from "
             "prompt+output, then score how much the actual graph diverges. Adds 2 "
             "LLM calls per graph and a 'divergence_score' column/ranking.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"LLM judge model (default: {LLM_JUDGE_MODEL}; or set EXPLORE_JUDGE_MODEL).",
    )
    parser.add_argument(
        "--slugs", type=str, default=None,
        help="Comma-separated slugs to score (only these graphs are processed). "
             "Combine with --update to add/refresh just these rows in the existing report.",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Merge the freshly-scored rows into the existing <results-name>.csv/.md "
             "(matched by slug) instead of overwriting the whole report.",
    )
    parser.add_argument(
        "--results-name", type=str, default="interesting_graphs",
        help="Basename (no extension) of the report files to write/update under "
             "analysis/results/. Default: interesting_graphs (-> interesting_graphs.csv/.md). "
             "Use e.g. interesting_graphs_context for the context-prompt set.",
    )
    parser.add_argument(
        "--merge-hidden-multihop", action="store_true",
        help="Rescore mode: re-judge ONLY the merged 'hidden intermediate concept / "
             "multi-hop reasoning' dimension over the existing <results-name>.csv, drop "
             "the two original columns, and rewrite the report. Uses --graphs_dir to "
             "locate graph files; all other columns are preserved.",
    )
    parser.add_argument(
        "--combine", type=str, default=None,
        help="Combine mode: comma-separated report basenames (under analysis/results/, "
             "no extension) to concatenate into --results-name. Pure data merge — no "
             "graphs loaded, no LLM. Rows are deduplicated by slug; inputs should share "
             "the same category schema.",
    )
    parser.add_argument(
        "--drop-category", type=str, default=None,
        help="Drop an LLM category by key (e.g. suppression_or_inhibition) from the "
             "existing --results-name report: removes its score_/reason_ columns and "
             "regenerates the rankings. Pure data edit — no graphs/LLM.",
    )
    args = parser.parse_args()
    if args.model:
        LLM_JUDGE_MODEL = args.model

    variants = [v.strip() for v in args.variants.split(",")]
    if args.drop_category:
        drop_category(
            category_key=args.drop_category.strip(),
            results_name=args.results_name,
            min_confidence=args.min_confidence,
        )
    elif args.combine:
        combine_reports(
            report_names=[n.strip() for n in args.combine.split(",") if n.strip()],
            results_name=args.results_name,
            min_confidence=args.min_confidence,
        )
    elif args.merge_hidden_multihop:
        rescore_merged(
            graphs_dir=args.graphs_dir,
            results_name=args.results_name,
            influence_threshold=args.influence_threshold,
            min_confidence=args.min_confidence,
            use_llm=not args.no_llm,
            variants=variants,
        )
    else:
        run(
            graphs_dir=args.graphs_dir,
            min_confidence=args.min_confidence,
            influence_threshold=args.influence_threshold,
            use_llm=not args.no_llm,
            ground_truth=args.ground_truth,
            variants=variants,
            divergence=args.divergence,
            only_slugs={s.strip() for s in args.slugs.split(",") if s.strip()} if args.slugs else None,
            update=args.update,
            results_name=args.results_name,
        )
