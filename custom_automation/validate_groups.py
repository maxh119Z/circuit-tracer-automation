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
graph) to make the task non-trivial. Each method is run N_RUNS times with
different random seeds; results are reported as mean ± stderr.

Reads:  artifacts/feature_descriptions.json
        artifacts/feature_groups.json
        artifacts/manual_groups.json  (optional)
Writes: artifacts/validation_report.json
        artifacts/validation_history.json

Usage:
    OPENAI_API_KEY=sk-xxx python validate_groups.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
CONCURRENCY_LIMIT = 20
MIN_GROUP_SIZE = 2           # Skip groups with fewer features than this
MAX_NEGATIVES_RATIO = 3      # Up to 3x negatives relative to positives per group
MAX_SNIPPETS_PER_FEATURE = 2 # Activating text snippets sampled per feature (Method 2)
RANDOM_SEED = 42
N_RUNS = 5                   # Validation runs for error bar estimation


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
    if not index:
        return None
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


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def compute_run_stats(runs: list[list[dict]]) -> list[dict]:
    """Aggregate N runs into per-group mean ± stderr statistics."""
    group_runs: dict[str, list[dict]] = {}
    for run in runs:
        for result in run:
            group_runs.setdefault(result["group"], []).append(result)

    stats = []
    for group, results in sorted(group_runs.items()):
        stat: dict[str, Any] = {"group": group, "n_runs": len(results)}
        for metric in ("precision", "recall", "f1"):
            values = [r[metric] for r in results]
            mean = sum(values) / len(values)
            stat[f"mean_{metric}"] = round(mean, 4)
            stat[f"stderr_{metric}"] = round(_stderr(values), 4)
        stat["n_positive"] = results[0].get("n_positive", 0)
        stats.append(stat)
    return stats


def macro_stats(run_stats: list[dict]) -> dict[str, float]:
    """Macro average ± stderr of per-group mean metrics."""
    if not run_stats:
        return {k: 0.0 for k in (
            "mean_precision", "stderr_precision",
            "mean_recall", "stderr_recall",
            "mean_f1", "stderr_f1",
        )}
    result: dict[str, float] = {}
    for metric in ("precision", "recall", "f1"):
        means = [s[f"mean_{metric}"] for s in run_stats]
        result[f"mean_{metric}"] = round(sum(means) / len(means), 4)
        result[f"stderr_{metric}"] = round(_stderr(means), 4)
    return result


def degenerate_stats_multi(runs: list[list[dict]]) -> dict[str, Any]:
    """Average degenerate GPT behavior (selected all / selected none) across runs."""
    per_run = []
    for run in runs:
        selected_all = sum(
            1 for r in run
            if r["n_predicted"] == r["n_positive"] + r["n_negative"]
        )
        selected_none = sum(1 for r in run if r["n_predicted"] == 0)
        per_run.append({
            "selected_all": selected_all,
            "selected_none": selected_none,
            "total": len(run),
        })
    if not per_run:
        return {"mean_selected_all": 0, "mean_selected_none": 0, "total": 0}
    avg_all = sum(r["selected_all"] for r in per_run) / len(per_run)
    avg_none = sum(r["selected_none"] for r in per_run) / len(per_run)
    return {
        "mean_selected_all": round(avg_all, 2),
        "mean_selected_none": round(avg_none, 2),
        "total": per_run[0]["total"],
    }


# ---------------------------------------------------------------------------
# Attribution coverage
# ---------------------------------------------------------------------------

def compute_attribution_coverage(
    features: list[dict], groups: dict[str, str]
) -> float:
    """Fraction of total influence score flowing through annotated (non-Ungrouped) nodes."""
    id_to_score = {
        f["id"]: float(f.get("influence_score", 0.0))
        for f in features if "id" in f
    }
    total = sum(id_to_score.values())
    if total == 0.0:
        return 0.0
    annotated = sum(
        id_to_score.get(fid, 0.0)
        for fid, gname in groups.items()
        if gname not in ("Ungrouped",)
        and not gname.startswith(('Emb: "', 'Output: "'))
        and fid in id_to_score
    )
    return round(annotated / total, 4)


# ---------------------------------------------------------------------------
# LLM group name regeneration for manual groups
# ---------------------------------------------------------------------------

async def _regenerate_name(
    original_name: str,
    member_features: list[dict],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> str:
    """Ask the LLM to generate a group label from its member feature descriptions."""
    feature_list = "\n".join(
        f"- {f.get('generated_description', 'No description')}"
        for f in member_features
    )
    prompt = (
        "You are an AI interpretability researcher. Below are descriptions of neurons "
        "that form a semantic group inside a language model.\n"
        "Generate a concise label (≤5 words) that best describes what they collectively represent.\n\n"
        f"Features:\n{feature_list}\n\n"
        "Respond with only the label, no explanation."
    )
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=VALIDATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except Exception as exc:
            log.warning("Failed to regenerate label for '%s': %s", original_name, exc)
            return original_name


async def regenerate_manual_group_names(
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, list[dict]]:
    """Re-label manual groups using LLM descriptions to level the playing field vs auto groups."""
    log.info("Regenerating manual group labels via LLM...")
    names = list(group_index.keys())
    new_names = await asyncio.gather(*[
        _regenerate_name(name, group_index[name], client, sem)
        for name in names
    ])
    new_index: dict[str, list[dict]] = {}
    for original, new_name, feats in zip(names, new_names, group_index.values()):
        log.info("  '%s'  →  '%s'", original, new_name)
        new_index[new_name] = feats
    return new_index


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
    seed: int = RANDOM_SEED,
) -> list[dict]:
    rng = random.Random(seed)
    tasks = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

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
    seed: int = RANDOM_SEED,
) -> list[dict]:
    rng = random.Random(seed)
    tasks = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

        pos_snippets = [s for f in pos_features for s in _get_snippets(f)]
        if not pos_snippets:
            log.warning("[Method2] %s: no text snippets available, skipping.", group_name)
            continue

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


def _condition_block(
    m1_runs: list[list[dict]], m2_runs: list[list[dict]]
) -> dict[str, Any]:
    m1_stats = compute_run_stats(m1_runs)
    m2_stats = compute_run_stats(m2_runs)
    return {
        "method1": {
            "groups": m1_stats,
            "macro_avg": macro_stats(m1_stats),
            "degenerate": degenerate_stats_multi(m1_runs),
        },
        "method2": {
            "groups": m2_stats,
            "macro_avg": macro_stats(m2_stats),
            "degenerate": degenerate_stats_multi(m2_runs),
        },
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

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Attribution coverage (not run-dependent — computed once)
    auto_coverage = compute_attribution_coverage(features, groups)
    log.info("Auto attribution coverage: %.1f%%", auto_coverage * 100)

    # N runs — auto + random
    auto_m1_runs: list[list[dict]] = []
    auto_m2_runs: list[list[dict]] = []
    rand_m1_runs: list[list[dict]] = []
    rand_m2_runs: list[list[dict]] = []

    for run_idx in range(N_RUNS):
        seed = RANDOM_SEED + run_idx
        log.info("=== Run %d/%d (seed=%d) ===", run_idx + 1, N_RUNS, seed)

        log.info("  AUTO: Method 1")
        auto_m1_runs.append(await validate_method1(valid_groups, client, sem, seed))
        log.info("  AUTO: Method 2")
        auto_m2_runs.append(await validate_method2(valid_groups, client, sem, seed))

        rng = random.Random(seed)
        random_groups = create_random_group_index(valid_groups, rng)
        log.info("  RANDOM: Method 1")
        rand_m1_runs.append(await validate_method1(random_groups, client, sem, seed))
        log.info("  RANDOM: Method 2")
        rand_m2_runs.append(await validate_method2(random_groups, client, sem, seed))

    # Manual groups (optional)
    manual_m1_runs: list[list[dict]] = []
    manual_m2_runs: list[list[dict]] = []
    manual_coverage: float | None = None

    manual_group_index = load_manual_group_index(features)
    if manual_group_index:
        with open(MANUAL_GROUPS_FILE) as f:
            manual_groups_raw: dict[str, str] = json.load(f)
        manual_coverage = compute_attribution_coverage(features, manual_groups_raw)
        log.info("Manual attribution coverage: %.1f%%", manual_coverage * 100)

        # Regenerate group labels via LLM to level the playing field
        manual_group_index = await regenerate_manual_group_names(manual_group_index, client, sem)
        valid_manual = {k: v for k, v in manual_group_index.items() if len(v) >= MIN_GROUP_SIZE}

        if valid_manual:
            for run_idx in range(N_RUNS):
                seed = RANDOM_SEED + run_idx
                log.info("  MANUAL: Run %d/%d", run_idx + 1, N_RUNS)
                manual_m1_runs.append(await validate_method1(valid_manual, client, sem, seed))
                manual_m2_runs.append(await validate_method2(valid_manual, client, sem, seed))

    # Build report
    prompt = _read_prompt_from_graph()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report: dict[str, Any] = {
        "prompt": prompt,
        "timestamp": timestamp,
        "n_runs": N_RUNS,
        "auto": {
            **_condition_block(auto_m1_runs, auto_m2_runs),
            "attribution_coverage": auto_coverage,
        },
        "random": _condition_block(rand_m1_runs, rand_m2_runs),
    }
    if manual_m1_runs:
        report["manual"] = {
            **_condition_block(manual_m1_runs, manual_m2_runs),
            "attribution_coverage": manual_coverage,
        }

    # Save report
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

    print(f"\n========== VALIDATION SUMMARY ({N_RUNS} runs) ==========")
    print(f"\n{'Condition':<10}  {'M1 F1':>14}  {'M2 F1':>14}  {'Coverage':>9}")
    print("-" * 57)
    for label, key in conditions:
        m1a = report[key]["method1"]["macro_avg"]
        m2a = report[key]["method2"]["macro_avg"]
        cov = report[key].get("attribution_coverage")
        cov_str = f"{cov:.1%}" if cov is not None else "    N/A"
        print(
            f"{label:<10}  "
            f"{m1a['mean_f1']:>6.3f}±{m1a['stderr_f1']:.3f}  "
            f"{m2a['mean_f1']:>6.3f}±{m2a['stderr_f1']:.3f}  "
            f"{cov_str:>9}"
        )

    print(f"\n--- Per-group M1 F1 (mean ± stderr over {N_RUNS} runs) ---")
    auto_stats = report["auto"]["method1"]["groups"]
    for s in sorted(auto_stats, key=lambda x: x["mean_f1"], reverse=True):
        print(f"  {s['group'][:42]:<42}  {s['mean_f1']:.3f} ± {s['stderr_f1']:.3f}")

    print("\n--- Degenerate GPT Behavior ---")
    for label, key in conditions:
        for mkey, mlabel in [("method1", "M1"), ("method2", "M2")]:
            d = report[key][mkey]["degenerate"]
            print(
                f"  {label} {mlabel}: "
                f"mean_selected_all={d['mean_selected_all']}/{d['total']}  "
                f"mean_selected_none={d['mean_selected_none']}/{d['total']}"
            )
    print("=" * 57 + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()