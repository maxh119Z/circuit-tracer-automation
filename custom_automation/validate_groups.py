"""
Validation — Evaluate the precision of group descriptions using two methods.

Method 1 — Description matching:
    Given a group description and a pool of feature descriptions (positives from
    the group + negatives from other groups in the same graph), can the model
    correctly identify which feature descriptions belong to the group?

Method 2 — Text snippet matching:
    Given a group description and a pool of activating text snippets (positives
    from features in the group + negatives from other features in the graph),
    can the model correctly identify which snippets would activate a feature in
    the group?

Both methods use intra-graph negatives (hard negatives from the same attribution
graph) to make the task non-trivial.

Reads:  artifacts/feature_descriptions.json
        artifacts/feature_groups.json
Writes: artifacts/validation_report.json

Usage:
    OPENAI_API_KEY=sk-xxx python validate_groups.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import (
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    GROUPING_MODEL,
    MANUAL_GROUPS_FILE,
    VALIDATION_HISTORY_FILE,
    VALIDATION_REPORT_FILE,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALIDATION_MODEL = GROUPING_MODEL
CONCURRENCY_LIMIT = 80
MIN_GROUP_SIZE = 2           # Skip groups with fewer features than this
MAX_NEGATIVES_RATIO = 3      # Up to 3x negatives relative to positives per group
MAX_SNIPPETS_PER_FEATURE = 2 # Activating text snippets sampled per feature (Method 2)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Pydantic schema for structured output
# ---------------------------------------------------------------------------

class SelectionOutput(BaseModel):
    selected_indices: list[int] = Field(
        description="1-based indices of items that belong to the group."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[list[dict], dict[str, str]]:
    if not FEATURE_DESCRIPTIONS_FILE.exists():
        log.error("Missing %s — run generate_description.py first.", FEATURE_DESCRIPTIONS_FILE)
        sys.exit(1)
    if not FEATURE_GROUPS_FILE.exists():
        log.error("Missing %s — run generate_supernodes.py first.", FEATURE_GROUPS_FILE)
        sys.exit(1)

    with open(FEATURE_DESCRIPTIONS_FILE) as f:
        features: list[dict] = json.load(f)
    with open(FEATURE_GROUPS_FILE) as f:
        groups: dict[str, str] = json.load(f)

    return features, groups


def build_group_index(
    features: list[dict], groups: dict[str, str]
) -> dict[str, list[dict]]:
    """Map group_name → list of feature dicts, excluding Ungrouped and embedding/logit nodes."""
    id_to_feature = {f["id"]: f for f in features if "id" in f}
    index: dict[str, list[dict]] = {}

    for fid, gname in groups.items():
        # Skip non-semantic group types (embedding/logit nodes have their own naming convention)
        if gname in ("Ungrouped",) or gname.startswith(('Emb: "', 'Output: "')):
            continue
        feat = id_to_feature.get(fid)
        if feat and feat.get("generated_description"):
            index.setdefault(gname, []).append(feat)

    return index


def create_random_group_index(
    group_index: dict[str, list[dict]], rng: random.Random
) -> dict[str, list[dict]]:
    """Random baseline: shuffle features across groups, preserving group sizes."""
    all_features = [f for feats in group_index.values() for f in feats]
    rng.shuffle(all_features)
    random_index: dict[str, list[dict]] = {}
    idx = 0
    for group_name, feats in group_index.items():
        random_index[group_name] = all_features[idx: idx + len(feats)]
        idx += len(feats)
    return random_index


def load_manual_group_index(features: list[dict]) -> dict[str, list[dict]] | None:
    """Load hand-curated groups from MANUAL_GROUPS_FILE if it exists."""
    if not MANUAL_GROUPS_FILE.exists():
        return None
    with open(MANUAL_GROUPS_FILE) as f:
        manual_groups: dict[str, str] = json.load(f)
    index = build_group_index(features, manual_groups)
    log.info("Loaded manual groups from %s (%d groups)", MANUAL_GROUPS_FILE, len(index))
    return index


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_method1_prompt(group_name: str, items: list[tuple[dict, bool]]) -> str:
    """Description matching: which feature descriptions belong to this group?"""
    lines = [
        f'Group description: "{group_name}"\n',
        "Below is a numbered list of feature descriptions from a language model's "
        "internal neurons. Some belong to this group; others do not.\n",
        "Identify which feature descriptions belong to this group.\n",
    ]
    for i, (feat, _) in enumerate(items, 1):
        desc = feat.get("generated_description", "No description")
        lines.append(f"{i}. {desc}")
    lines.append(
        "\nRespond with the 1-based indices of descriptions that belong to this group."
    )
    return "\n".join(lines)


def build_method2_prompt(group_name: str, items: list[tuple[str, bool]]) -> str:
    """Text snippet matching: which activating snippets come from this group?"""
    lines = [
        f'Group description: "{group_name}"\n',
        "Below is a numbered list of text excerpts that strongly activated neurons "
        "inside a language model. Some come from neurons in this group; others do not.\n",
        "Identify which text excerpts would activate a neuron described as this group.\n",
    ]
    for i, (snippet, _) in enumerate(items, 1):
        lines.append(f"{i}. {snippet}")
    lines.append(
        "\nRespond with the 1-based indices of text excerpts that belong to this group."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predicted: set[int], actual_positive_indices: set[int]
) -> dict[str, Any]:
    tp = len(predicted & actual_positive_indices)
    fp = len(predicted - actual_positive_indices)
    fn = len(actual_positive_indices - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_avg(results: list[dict]) -> dict[str, float]:
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(sum(r["precision"] for r in results) / len(results), 4),
        "recall": round(sum(r["recall"] for r in results) / len(results), 4),
        "f1": round(sum(r["f1"] for r in results) / len(results), 4),
    }


def degenerate_stats(results: list[dict]) -> dict[str, int]:
    """Count groups where GPT selected everything or nothing."""
    selected_all = sum(
        1 for r in results
        if r["n_predicted"] == r["n_positive"] + r["n_negative"]
    )
    selected_none = sum(1 for r in results if r["n_predicted"] == 0)
    return {"selected_all": selected_all, "selected_none": selected_none, "total": len(results)}


# ---------------------------------------------------------------------------
# Async validation task (shared by both methods)
# ---------------------------------------------------------------------------

async def run_validation_task(
    group_name: str,
    prompt: str,
    actual_positive_indices: set[int],
    total_items: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    method_label: str,
) -> dict[str, Any]:
    async with sem:
        try:
            response = await client.beta.chat.completions.parse(
                model=VALIDATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=SelectionOutput,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                log.warning("[%s] %s: parsed response was None", method_label, group_name)
                predicted: set[int] = set()
            else:
                # Clamp to valid 1-based range
                predicted = {i for i in parsed.selected_indices if 1 <= i <= total_items}
        except Exception as exc:
            log.warning("[%s] %s: API error — %s", method_label, group_name, exc)
            predicted = set()

    metrics = compute_metrics(predicted, actual_positive_indices)
    log.info(
        "[%s] %-32s  n_pos=%-3d  P=%.2f  R=%.2f  F1=%.2f",
        method_label,
        group_name[:32],
        len(actual_positive_indices),
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return {
        "group": group_name,
        "n_positive": len(actual_positive_indices),
        "n_negative": total_items - len(actual_positive_indices),
        "n_predicted": len(predicted),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Method 1 — Feature description matching
# ---------------------------------------------------------------------------

async def validate_method1(
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    tasks = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

        # Hard negatives: feature descriptions from other groups in the same graph
        neg_pool = [
            f
            for gname, feats in group_index.items()
            if gname != group_name
            for f in feats
        ]
        n_neg = min(len(neg_pool), MAX_NEGATIVES_RATIO * len(pos_features))
        neg_features = rng.sample(neg_pool, n_neg)

        items: list[tuple[dict, bool]] = (
            [(f, True) for f in pos_features] +
            [(f, False) for f in neg_features]
        )
        rng.shuffle(items)

        actual_positive_indices = {i + 1 for i, (_, is_pos) in enumerate(items) if is_pos}
        prompt = build_method1_prompt(group_name, items)

        tasks.append(
            run_validation_task(
                group_name, prompt, actual_positive_indices, len(items),
                client, sem, "Method1"
            )
        )

    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Method 2 — Text snippet matching
# ---------------------------------------------------------------------------

def _get_snippets(feat: dict) -> list[str]:
    return [
        act.get("context", "").strip()
        for act in feat.get("top_activations", [])[:MAX_SNIPPETS_PER_FEATURE]
        if act.get("context", "").strip()
    ]


async def validate_method2(
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    tasks = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

        pos_snippets = [s for f in pos_features for s in _get_snippets(f)]
        if not pos_snippets:
            log.warning("[Method2] %s: no text snippets available, skipping.", group_name)
            continue

        # Hard negatives: activating snippets from features in other groups
        neg_pool_feats = [
            f
            for gname, feats in group_index.items()
            if gname != group_name
            for f in feats
        ]
        neg_snippets_pool = [s for f in neg_pool_feats for s in _get_snippets(f)]
        n_neg = min(len(neg_snippets_pool), MAX_NEGATIVES_RATIO * len(pos_snippets))
        neg_snippets = rng.sample(neg_snippets_pool, n_neg)

        items: list[tuple[str, bool]] = (
            [(s, True) for s in pos_snippets] +
            [(s, False) for s in neg_snippets]
        )
        rng.shuffle(items)

        actual_positive_indices = {i + 1 for i, (_, is_pos) in enumerate(items) if is_pos}
        prompt = build_method2_prompt(group_name, items)

        tasks.append(
            run_validation_task(
                group_name, prompt, actual_positive_indices, len(items),
                client, sem, "Method2"
            )
        )

    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _read_prompt_from_graph() -> str:
    """Read the prompt string from the graph file metadata."""
    try:
        with open(GRAPH_FILE) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        prompt = meta.get("prompt", "")
        if not prompt:
            tokens = meta.get("prompt_tokens", [])
            prompt = "".join(str(t) for t in tokens)
        return prompt or GRAPH_FILE.stem
    except Exception:
        return GRAPH_FILE.stem


def _condition_block(m1: list[dict], m2: list[dict]) -> dict[str, Any]:
    return {
        "method1": {"groups": m1, "macro_avg": macro_avg(m1), "degenerate": degenerate_stats(m1)},
        "method2": {"groups": m2, "macro_avg": macro_avg(m2), "degenerate": degenerate_stats(m2)},
    }


async def main_async() -> None:
    features, groups = load_data()
    group_index = build_group_index(features, groups)

    valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}
    log.info(
        "Loaded %d groups total; %d have ≥%d features for validation.",
        len(group_index), len(valid_groups), MIN_GROUP_SIZE,
    )

    if not valid_groups:
        log.error("No groups with enough features to validate. Exiting.")
        sys.exit(1)

    prompt = _read_prompt_from_graph()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # --- Auto groups ---
    log.info("=== AUTO: Method 1 ===")
    auto_m1 = await validate_method1(valid_groups, client, sem)
    log.info("=== AUTO: Method 2 ===")
    auto_m2 = await validate_method2(valid_groups, client, sem)

    # --- Random baseline (same group sizes, shuffled features) ---
    rng = random.Random(RANDOM_SEED)
    random_groups = create_random_group_index(valid_groups, rng)
    log.info("=== RANDOM: Method 1 ===")
    rand_m1 = await validate_method1(random_groups, client, sem)
    log.info("=== RANDOM: Method 2 ===")
    rand_m2 = await validate_method2(random_groups, client, sem)

    # --- Manual groups (optional — place manual_groups.json in artifacts/) ---
    manual_m1: list[dict] | None = None
    manual_m2: list[dict] | None = None
    manual_group_index = load_manual_group_index(features)
    if manual_group_index:
        valid_manual = {k: v for k, v in manual_group_index.items() if len(v) >= MIN_GROUP_SIZE}
        if valid_manual:
            log.info("=== MANUAL: Method 1 ===")
            manual_m1 = await validate_method1(valid_manual, client, sem)
            log.info("=== MANUAL: Method 2 ===")
            manual_m2 = await validate_method2(valid_manual, client, sem)

    # Build report
    report: dict[str, Any] = {
        "prompt": prompt,
        "timestamp": timestamp,
        "auto": _condition_block(auto_m1, auto_m2),
        "random": _condition_block(rand_m1, rand_m2),
    }
    if manual_m1 is not None and manual_m2 is not None:
        report["manual"] = _condition_block(manual_m1, manual_m2)

    # Overwrite latest report
    with open(VALIDATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Validation report saved → %s", VALIDATION_REPORT_FILE)

    # Append to history
    history: list[dict] = []
    if VALIDATION_HISTORY_FILE.exists():
        with open(VALIDATION_HISTORY_FILE) as f:
            history = json.load(f)
    history.append(report)
    with open(VALIDATION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Appended to history (%d total runs) → %s", len(history), VALIDATION_HISTORY_FILE)

    # Summary table
    conditions: list[tuple[str, str]] = [("Auto", "auto"), ("Random", "random")]
    if "manual" in report:
        conditions.append(("Manual", "manual"))

    n = len(auto_m1)
    print("\n========== VALIDATION SUMMARY ==========")
    print(f"\n{'Condition':<10}  {'M1 P':>6}  {'M1 R':>6}  {'M1 F1':>6}  {'M2 P':>6}  {'M2 R':>6}  {'M2 F1':>6}")
    print("-" * 55)
    for label, key in conditions:
        m1a = report[key]["method1"]["macro_avg"]
        m2a = report[key]["method2"]["macro_avg"]
        print(
            f"{label:<10}  {m1a['precision']:>6.3f}  {m1a['recall']:>6.3f}  {m1a['f1']:>6.3f}"
            f"  {m2a['precision']:>6.3f}  {m2a['recall']:>6.3f}  {m2a['f1']:>6.3f}"
        )
    print(f"\n({n} groups evaluated)")

    print("\n--- Degenerate GPT Behavior ---")
    for label, key in conditions:
        for mkey, mlabel in [("method1", "M1"), ("method2", "M2")]:
            d = report[key][mkey]["degenerate"]
            print(
                f"  {label} {mlabel}: "
                f"selected_all={d['selected_all']}/{d['total']}  "
                f"selected_none={d['selected_none']}/{d['total']}"
            )
    print("=========================================\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
