"""
Validation — Evaluate group quality and description quality.

Group-level methods:

  Method 1 — Feature identification (1-in-10):
      For each feature in a group (up to 5), present its description among 9
      negative descriptions. Ask the model to pick the one that best matches
      the group name. Score: binary accuracy (1 if correct, 0 if wrong).
      Random chance baseline: 10%.

  Method 2 — Text snippet matching (5-in-10):
      Present 5 positive text snippets (from group features) and 5 negative
      snippets. Tell the model exactly 5 are from the group. Score: fraction
      of the 5 positives correctly identified (|correct ∩ actual| / 5).
      Random chance baseline: 50%.

Description-level methods:

  Method D1 — Description accuracy (1-in-10):
      For each feature, present the feature's raw evidence (activations +
      output tokens) and ask the model to pick its description from a lineup
      of 10 descriptions (1 correct + 9 from other features). Tests whether
      the description accurately summarizes the feature's evidence.
      Random chance baseline: 10%.

  Method D2 — Description snippet match (5-in-10):
      For each feature, given its description, present 10 text snippets (5
      from this feature's top activations + 5 from other features). Ask the
      model to pick the 5 that activated this feature. Score: fraction of
      the 5 positives correctly identified (|correct ∩ actual| / 5).
      Random chance baseline: 50%.

Aggregation:
    For each group (or "all" for D1), we compute one mean score per run
    (averaging over trials within that run). We then report mean ± stderr
    across the N_RUNS run-level means.

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
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import (
    DESCRIPTION_VARIANT,
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    GROUPING_MODEL,
    GROUPING_TOP_K_SEED,
    GROUPING_VARIANT,
    MANUAL_GROUPS_FILE,
    VALIDATION_HISTORY_FILE,
    VALIDATION_REPORT_FILE,
    setup_logging,
)

log = setup_logging()

# Markdown report — same location as JSON, .md extension
VALIDATION_MARKDOWN_FILE = VALIDATION_REPORT_FILE.with_suffix(".md")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALIDATION_MODEL = GROUPING_MODEL
CONCURRENCY_LIMIT = 500      
MIN_GROUP_SIZE = 2            # Groups with <2 features are skipped
MAX_FEATURES_M1 = 5           # Test up to 5 features per group in Method 1
N_NEG_FEATURES_M1 = 9         # 9 negatives per trial in Method 1 (1 + 9 = 10)
N_POS_SNIPPETS_M2 = 5         # 5 positive snippets in Method 2
N_NEG_SNIPPETS_M2 = 5         # 5 negative snippets in Method 2 (5 + 5 = 10)
MAX_FEATURES_D1 = 50          # Test up to 50 features total for description accuracy
N_NEG_DESCS_D1 = 9            # 9 negative descriptions per trial (1 + 9 = 10)
MAX_FEATURES_D2 = 50          # Test up to 50 features total for description snippet accuracy
N_POS_SNIPPETS_D2 = 5         # 5 positive snippets in Method D2
N_NEG_SNIPPETS_D2 = 5         # 5 negative snippets in Method D2 (5 + 5 = 10)
RANDOM_SEED = 42
N_RUNS = 5


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class SingleSelection(BaseModel):
    """Method 1: model picks exactly 1 feature from the list."""
    selected_index: int = Field(
        description="1-based index of the single feature that best matches the group."
    )


class FiveSelection(BaseModel):
    """Method 2: model picks exactly 5 snippets from the list."""
    selected_indices: list[int] = Field(
        description="Exactly 5 unique 1-based indices of text snippets that belong to this group."
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
        log.info("No manual groups found.")
        return None

    with open(MANUAL_GROUPS_FILE) as f:
        manual_groups: dict[str, str] = json.load(f)

    feature_ids = {f["id"] for f in features if "id" in f}
    missing_ids = [fid for fid in manual_groups if fid not in feature_ids]

    no_desc_ids = []
    id_to_feature = {f["id"]: f for f in features if "id" in f}
    for fid in manual_groups:
        feat = id_to_feature.get(fid)
        if feat is not None and not feat.get("generated_description"):
            no_desc_ids.append(fid)

    index = build_group_index(features, manual_groups)

    raw_group_counts = defaultdict(int)
    for fid, gname in manual_groups.items():
        if gname != "Ungrouped" and not gname.startswith(('Emb: "', 'Output: "')):
            raw_group_counts[gname] += 1

    built_group_counts = {g: len(v) for g, v in index.items()}
    valid_group_counts = {g: n for g, n in built_group_counts.items() if n >= MIN_GROUP_SIZE}

    log.info("Loaded manual groups file: %s", MANUAL_GROUPS_FILE)
    log.info("Manual assignments in file: %d", len(manual_groups))
    log.info("Manual feature IDs missing from feature_descriptions: %d", len(missing_ids))
    log.info("Manual feature IDs with empty generated_description: %d", len(no_desc_ids))
    log.info("Manual groups before feature filtering: %d", len(raw_group_counts))
    log.info("Manual groups after build_group_index: %d", len(index))
    log.info("Manual groups with >= %d features: %d", MIN_GROUP_SIZE, len(valid_group_counts))

    if missing_ids:
        log.info("Example missing manual IDs: %s", missing_ids[:10])
    if no_desc_ids:
        log.info("Example no-description manual IDs: %s", no_desc_ids[:10])

    if raw_group_counts:
        log.info("Top raw manual group sizes: %s",
                 sorted(raw_group_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    if built_group_counts:
        log.info("Top built manual group sizes: %s",
                 sorted(built_group_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    if not index:
        log.warning("Manual groups loaded but no valid grouped features were found.")
        return None

    return index

# ---------------------------------------------------------------------------
# Negative pool builders
# ---------------------------------------------------------------------------

def build_medium_neg_pool(features: list[dict], groups: dict[str, str]) -> list[dict]:
    """Ungrouped features that have descriptions — harder than other named groups."""
    id_to_feature = {f["id"]: f for f in features if "id" in f}
    return [
        id_to_feature[fid]
        for fid, gname in groups.items()
        if gname == "Ungrouped"
        and fid in id_to_feature
        and id_to_feature[fid].get("generated_description")
    ]


def build_hard_neg_pool(features: list[dict]) -> list[dict]:
    """Features beyond Phase 1's seed window — not seen by Phase 1 grouping."""
    sorted_feats = sorted(
        [f for f in features if "influence_score" in f],
        key=lambda f: float(f.get("influence_score", 0.0)),
        reverse=True,
    )
    return [
        f for f in sorted_feats[GROUPING_TOP_K_SEED:]
        if f.get("generated_description")
    ]


# ---------------------------------------------------------------------------
# Snippet formatting with <<<>>> triggers
# ---------------------------------------------------------------------------

def _format_snippet(act: dict) -> str:
    """Format an activation example with <<<>>> around the trigger token."""
    context = act.get("context", "").strip()
    trigger = act.get("trigger", "").strip()
    if trigger and trigger in context:
        return context.replace(trigger, f"<<<{trigger}>>>", 1)
    if trigger:
        return f"{context} [Activates on: <<<{trigger}>>>]"
    return context


def _get_formatted_snippets(feat: dict, max_n: int = 5) -> list[str]:
    """Get up to max_n formatted text snippets from a feature's top activations."""
    return [
        _format_snippet(act)
        for act in feat.get("top_activations", [])[:max_n]
        if act.get("context", "").strip()
    ]


# ---------------------------------------------------------------------------
# Prompt builders — item count is dynamic, never lies
# ---------------------------------------------------------------------------

def build_m1_prompt(group_name: str, items: list[tuple[str, bool]]) -> str:
    """Method 1: pick the 1 feature (out of N) that matches the group."""
    n = len(items)
    lines = [
        f'Group: "{group_name}"\n',
        f"Below are {n} feature descriptions from neurons inside a language model. "
        "Exactly 1 of these belongs to the group above.\n",
        "Pick the single feature description that best matches the group.\n",
    ]
    for i, (desc, _) in enumerate(items, 1):
        lines.append(f"{i}. {desc}")
    lines.append(
        "\nRespond with the 1-based index of the best match."
    )
    return "\n".join(lines)


def build_m2_prompt(group_name: str, items: list[tuple[str, bool]], n_pos: int) -> str:
    """Method 2: pick the n_pos text snippets (out of N) from this group."""
    n = len(items)
    lines = [
        f'Group: "{group_name}"\n',
        f"Below are {n} text excerpts that strongly activated neurons inside a "
        "language model. The key activating tokens are highlighted with <<<>>>. "
        f"Exactly {n_pos} of these come from neurons in this group.\n",
        f"Identify which {n_pos} text excerpts belong to this group.\n",
    ]
    for i, (snippet, _) in enumerate(items, 1):
        lines.append(f"{i}. {snippet}")
    lines.append(
        f"\nRespond with the 1-based indices of the {n_pos} excerpts from this group."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------

async def run_m1_trial(
    group_name: str,
    prompt: str,
    correct_idx: int,
    n_items: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> float:
    """Run one Method 1 trial. Returns 1.0 if correct, 0.0 if wrong."""
    async with sem:
        try:
            resp = await client.beta.chat.completions.parse(
                model=VALIDATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=SingleSelection
            )
            parsed = resp.choices[0].message.parsed
            picked = parsed.selected_index if parsed else -1
        except Exception as exc:
            log.warning("[M1] %s: API error — %s", group_name, exc)
            picked = -1

    # Validate: must be in [1, n_items]
    if picked < 1 or picked > n_items:
        log.debug("[M1] %s: invalid index %d (n_items=%d)", group_name, picked, n_items)
        return 0.0

    correct = 1.0 if picked == correct_idx else 0.0
    return correct


async def run_m2_task(
    group_name: str,
    prompt: str,
    actual_positive_indices: set[int],
    n_items: int,
    n_expected: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> float:
    """
    Run one Method 2 task. Returns |correct ∩ actual| / n_expected.

    Output validation: the model must return exactly n_expected unique indices
    in [1, n_items]. If it doesn't, score is 0 (response discarded).
    """
    async with sem:
        try:
            resp = await client.beta.chat.completions.parse(
                model=VALIDATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=FiveSelection
            )
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                log.debug("[M2] %s: parsed response was None", group_name)
                return 0.0
            raw_indices = parsed.selected_indices
        except Exception as exc:
            log.warning("[M2] %s: API error — %s", group_name, exc)
            return 0.0

    # Validate: exactly n_expected unique indices, all in [1, n_items]
    valid_indices = [i for i in raw_indices if 1 <= i <= n_items]
    unique_valid = list(dict.fromkeys(valid_indices))  # deduplicate, preserve order

    if len(unique_valid) != n_expected:
        log.debug(
            "[M2] %s: expected %d unique indices, got %d valid (%d raw). Discarding.",
            group_name, n_expected, len(unique_valid), len(raw_indices),
        )
        return 0.0

    predicted = set(unique_valid)
    n_correct = len(predicted & actual_positive_indices)
    accuracy = n_correct / len(actual_positive_indices)
    return accuracy


# ---------------------------------------------------------------------------
# Task generation — Method 1
# ---------------------------------------------------------------------------

def generate_m1_tasks(
    group_index: dict[str, list[dict]],
    seed: int,
    external_neg_pool: list[dict] | None = None,
) -> list[dict]:
    """
    Generate Method 1 trial specs for one run.

    If external_neg_pool is provided, use it as the negative source (medium/hard).
    Otherwise fall back to other groups in group_index (easy).
    If fewer than N_NEG_FEATURES_M1 negatives available, use what we have
    (prompt reflects actual count, never lies about "10").
    """
    rng = random.Random(seed)
    tasks: list[dict] = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

        if external_neg_pool is not None:
            neg_pool = external_neg_pool
        else:
            neg_pool = [
                f for gname, feats in group_index.items()
                if gname != group_name
                for f in feats
            ]
        if not neg_pool:
            continue

        if len(pos_features) <= MAX_FEATURES_M1:
            test_features = list(pos_features)
        else:
            test_features = rng.sample(pos_features, MAX_FEATURES_M1)

        for pos_feat in test_features:
            n_neg = min(len(neg_pool), N_NEG_FEATURES_M1)
            neg_feats = rng.sample(neg_pool, n_neg)

            items: list[tuple[str, bool]] = [
                (pos_feat.get("generated_description", "No description"), True)
            ] + [
                (f.get("generated_description", "No description"), False)
                for f in neg_feats
            ]
            rng.shuffle(items)

            correct_idx = next(i + 1 for i, (_, is_pos) in enumerate(items) if is_pos)
            n_items = len(items)

            tasks.append({
                "group_name": group_name,
                "prompt": build_m1_prompt(group_name, items),
                "correct_idx": correct_idx,
                "n_items": n_items,
            })

    return tasks


# ---------------------------------------------------------------------------
# Task generation — Method 2
# ---------------------------------------------------------------------------

def generate_m2_tasks(
    group_index: dict[str, list[dict]],
    seed: int,
    external_neg_pool: list[dict] | None = None,
) -> list[dict]:
    """
    Generate Method 2 task specs for one run.

    Positive snippets: sample exactly N_POS_SNIPPETS_M2 from the group.
    If fewer available, skip this group.
    Negative snippets: drawn from external_neg_pool (medium/hard) or other groups (easy).
    If fewer available, use what we have (prompt reflects actual count).
    """
    rng = random.Random(seed)
    tasks: list[dict] = []

    for group_name, pos_features in group_index.items():
        if len(pos_features) < MIN_GROUP_SIZE:
            continue

        all_pos_snippets = [
            s for f in pos_features for s in _get_formatted_snippets(f)
        ]
        if len(all_pos_snippets) < N_POS_SNIPPETS_M2:
            log.warning(
                "[M2] %s: only %d snippets (need %d), skipping.",
                group_name, len(all_pos_snippets), N_POS_SNIPPETS_M2,
            )
            continue

        pos_snippets = rng.sample(all_pos_snippets, N_POS_SNIPPETS_M2)

        if external_neg_pool is not None:
            neg_snippet_pool = [
                s for f in external_neg_pool for s in _get_formatted_snippets(f)
            ]
        else:
            neg_snippet_pool = [
                s
                for gname, feats in group_index.items()
                if gname != group_name
                for f in feats
                for s in _get_formatted_snippets(f)
            ]
        n_neg = min(len(neg_snippet_pool), N_NEG_SNIPPETS_M2)
        if n_neg == 0:
            log.warning("[M2] %s: no negative snippets available, skipping.", group_name)
            continue
        neg_snippets = rng.sample(neg_snippet_pool, n_neg)

        items: list[tuple[str, bool]] = (
            [(s, True) for s in pos_snippets] +
            [(s, False) for s in neg_snippets]
        )
        rng.shuffle(items)

        actual_positive_indices = {
            i + 1 for i, (_, is_pos) in enumerate(items) if is_pos
        }
        n_pos = len(pos_snippets)
        n_items = len(items)

        tasks.append({
            "group_name": group_name,
            "prompt": build_m2_prompt(group_name, items, n_pos),
            "actual_positive_indices": actual_positive_indices,
            "n_items": n_items,
            "n_expected": n_pos,
        })

    return tasks


# ---------------------------------------------------------------------------
# Prompt builder — Method D1 (description accuracy)
# ---------------------------------------------------------------------------

def _format_feature_evidence(feat: dict, prompt_text: str) -> str:
    """Format a feature's raw evidence (activations + output tokens) for D1 trials."""
    lines = [f"Feature {feat.get('id', '?')}:\n"]

    lines.append("--- PROMPT CONTEXT ---")
    lines.append(prompt_text)

    lines.append("\n--- INPUT ACTIVATIONS ---")
    for i, act in enumerate(feat.get("top_activations", [])[:10], 1):
        lines.append(f"Excerpt {i}: {_format_snippet(act)}")

    lines.append("\n--- GLOBAL OUTPUT TOKENS ---")
    promotes = feat.get("promotes", [])
    demotes = feat.get("demotes", [])
    lines.append(f"Top Promoted Tokens: {', '.join(promotes) if promotes else 'None available'}")
    lines.append(f"Top Demoted Tokens: {', '.join(demotes) if demotes else 'None available'}")

    return "\n".join(lines)


def build_d1_prompt(evidence: str, items: list[tuple[str, bool]]) -> str:
    """Method D1: given feature evidence, pick the 1 correct description from N."""
    n = len(items)
    lines = [
        "Below is evidence about a single feature neuron in a language model "
        "(its input activations and output tokens).\n",
        evidence,
        f"\nBelow are {n} candidate descriptions. Exactly 1 of these correctly "
        "describes this feature. Pick the single best match.\n",
    ]
    for i, (desc, _) in enumerate(items, 1):
        lines.append(f"{i}. {desc}")
    lines.append(
        "\nRespond with the 1-based index of the description that best matches "
        "the feature evidence above."
    )
    return "\n".join(lines)


def build_d2_prompt(description: str, items: list[tuple[str, bool]], n_pos: int) -> str:
    """Method D2: given a feature description, pick the n_pos snippets that activated it."""
    n = len(items)
    lines = [
        f'Feature description: "{description}"\n',
        f"Below are {n} text excerpts that strongly activated neurons inside a "
        "language model. The key activating tokens are highlighted with <<<>>>. "
        f"Exactly {n_pos} of these come from the neuron described above.\n",
        f"Identify which {n_pos} text excerpts match this feature description.\n",
    ]
    for i, (snippet, _) in enumerate(items, 1):
        lines.append(f"{i}. {snippet}")
    lines.append(
        f"\nRespond with the 1-based indices of the {n_pos} excerpts that activated this feature."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task generation — Method D1
# ---------------------------------------------------------------------------

def generate_d1_tasks(
    features: list[dict],
    prompt_text: str,
    seed: int,
    max_features: int = MAX_FEATURES_D1,
) -> list[dict]:
    """
    Generate Method D1 trial specs for one run.

    For each feature (up to max_features), present its evidence and a lineup
    of descriptions (1 correct + N_NEG_DESCS_D1 from other features).
    """
    rng = random.Random(seed)

    # Only features that have descriptions and activations
    eligible = [
        f for f in features
        if f.get("generated_description")
        and f.get("generated_description") != "Error generating description"
        and f.get("top_activations")
    ]

    if len(eligible) < 2:
        return []

    if len(eligible) > max_features:
        test_features = rng.sample(eligible, max_features)
    else:
        test_features = list(eligible)

    tasks: list[dict] = []
    for pos_feat in test_features:
        neg_pool = [f for f in eligible if f["id"] != pos_feat["id"]]
        if not neg_pool:
            continue

        n_neg = min(len(neg_pool), N_NEG_DESCS_D1)
        neg_feats = rng.sample(neg_pool, n_neg)

        items: list[tuple[str, bool]] = [
            (pos_feat["generated_description"], True)
        ] + [
            (f["generated_description"], False)
            for f in neg_feats
        ]
        rng.shuffle(items)

        correct_idx = next(i + 1 for i, (_, is_pos) in enumerate(items) if is_pos)
        evidence = _format_feature_evidence(pos_feat, prompt_text)

        tasks.append({
            "group_name": "description_accuracy",  # pseudo-group for aggregation
            "prompt": build_d1_prompt(evidence, items),
            "correct_idx": correct_idx,
            "n_items": len(items),
            "feature_id": pos_feat["id"],
        })

    return tasks


# ---------------------------------------------------------------------------
# Task generation — Method D2
# ---------------------------------------------------------------------------

def generate_d2_tasks(
    features: list[dict],
    seed: int,
    max_features: int = MAX_FEATURES_D2,
) -> list[dict]:
    """
    Generate Method D2 trial specs for one run.

    For each feature (up to max_features), present its description alongside
    10 text snippets (N_POS_SNIPPETS_D2 from this feature + N_NEG_SNIPPETS_D2
    from other features). Ask the model to pick the ones that activated it.
    """
    rng = random.Random(seed)

    eligible = [
        f for f in features
        if f.get("generated_description")
        and f.get("generated_description") != "Error generating description"
        and len(_get_formatted_snippets(f)) >= N_POS_SNIPPETS_D2
    ]

    if len(eligible) < 2:
        return []

    if len(eligible) > max_features:
        test_features = rng.sample(eligible, max_features)
    else:
        test_features = list(eligible)

    tasks: list[dict] = []
    for pos_feat in test_features:
        pos_snippets = _get_formatted_snippets(pos_feat, N_POS_SNIPPETS_D2)

        neg_pool = [
            s
            for f in eligible
            if f["id"] != pos_feat["id"]
            for s in _get_formatted_snippets(f)
        ]
        if len(neg_pool) < N_NEG_SNIPPETS_D2:
            continue

        neg_snippets = rng.sample(neg_pool, N_NEG_SNIPPETS_D2)

        items: list[tuple[str, bool]] = (
            [(s, True) for s in pos_snippets] +
            [(s, False) for s in neg_snippets]
        )
        rng.shuffle(items)

        actual_positive_indices = {
            i + 1 for i, (_, is_pos) in enumerate(items) if is_pos
        }

        tasks.append({
            "group_name": "description_snippet_accuracy",  # pseudo-group for aggregation
            "prompt": build_d2_prompt(pos_feat["generated_description"], items, N_POS_SNIPPETS_D2),
            "actual_positive_indices": actual_positive_indices,
            "n_items": len(items),
            "n_expected": N_POS_SNIPPETS_D2,
            "feature_id": pos_feat["id"],
        })

    return tasks


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def aggregate_by_run(
    tagged_results: list[tuple[int, str, float]],  # (run_idx, group_name, score)
) -> tuple[list[dict], dict[str, float]]:
    """
    Correct run-level aggregation:
    1. For each (group, run), compute the mean score across trials in that run.
    2. For each group, report mean ± stderr across the run-level means.
    3. Macro average: mean of per-group means ± stderr of per-group means.
    """
    # Step 1: group by (group_name, run_idx) → list of trial scores
    nested: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run_idx, group_name, score in tagged_results:
        nested[group_name][run_idx].append(score)

    # Step 2: per-group run-level means
    per_group_stats: list[dict] = []
    for group_name in sorted(nested.keys()):
        run_means: list[float] = []
        total_trials = 0
        for run_idx in sorted(nested[group_name].keys()):
            trials = nested[group_name][run_idx]
            total_trials += len(trials)
            run_means.append(sum(trials) / len(trials))

        mean_acc = sum(run_means) / len(run_means)
        stderr_acc = _stderr(run_means)
        per_group_stats.append({
            "group": group_name,
            "mean_accuracy": round(mean_acc, 4),
            "stderr_accuracy": round(stderr_acc, 4),
            "n_runs": len(run_means),
            "total_trials": total_trials,
        })

    # Step 3: macro average across groups
    if per_group_stats:
        group_means = [s["mean_accuracy"] for s in per_group_stats]
        macro = {
            "mean_accuracy": round(sum(group_means) / len(group_means), 4),
            "stderr_accuracy": round(_stderr(group_means), 4),
        }
    else:
        macro = {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}

    return per_group_stats, macro


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
# LLM group name regeneration
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
        "You are an AI mechanistic interpretability researcher. Below are descriptions of neurons "
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
                max_completion_tokens=2048,
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except Exception as exc:
            log.warning("Failed to regenerate label for '%s': %s", original_name, exc)
            return original_name


async def regenerate_group_names(
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, list[dict]]:
    """
    Re-label groups using LLM-generated descriptions.

    Handles name collisions: if two groups get the same regenerated label,
    appends a numeric suffix to disambiguate.
    """
    log.info("Regenerating group labels via LLM (%d groups)...", len(group_index))
    names = list(group_index.keys())
    new_names = await asyncio.gather(*[
        _regenerate_name(name, group_index[name], client, sem)
        for name in names
    ])

    # Handle collisions
    new_index: dict[str, list[dict]] = {}
    name_counts: dict[str, int] = {}
    for original, new_name, feats in zip(names, new_names, group_index.values()):
        if new_name in new_index:
            count = name_counts.get(new_name, 1) + 1
            name_counts[new_name] = count
            deduped = f"{new_name} ({count})"
            log.warning("  Name collision: '%s' → '%s' (deduped to '%s')", original, new_name, deduped)
            new_name = deduped
        else:
            name_counts[new_name] = 1
        log.info("  '%s'  →  '%s'", original, new_name)
        new_index[new_name] = feats
    return new_index


# ---------------------------------------------------------------------------
# Run all validation for one condition
# ---------------------------------------------------------------------------

# async def run_condition(
#     label: str,
#     group_index: dict[str, list[dict]],
#     client: AsyncOpenAI,
#     sem: asyncio.Semaphore,
# ) -> dict[str, Any]:
#     """
#     Run N_RUNS of both methods for one condition, fully parallelized.
#
#     All tasks across all runs are generated upfront and fired in one
#     asyncio.gather. The semaphore handles rate limiting. Results are
#     tagged with run_idx for correct run-level aggregation.
#     """
#     valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}
#     if not valid_groups:
#         log.warning("[%s] No valid groups for validation.", label)
#         return {
#             "method1": {"groups": [], "macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}},
#             "method2": {"groups": [], "macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}},
#         }
#
#     m1_specs: list[tuple[int, dict]] = []
#     m2_specs: list[tuple[int, dict]] = []
#
#     for run_idx in range(N_RUNS):
#         seed = RANDOM_SEED + run_idx
#         for spec in generate_m1_tasks(valid_groups, seed):
#             m1_specs.append((run_idx, spec))
#         for spec in generate_m2_tasks(valid_groups, seed):
#             m2_specs.append((run_idx, spec))
#
#     log.info(
#         "[%s] Launching %d M1 trials + %d M2 tasks across %d runs…",
#         label, len(m1_specs), len(m2_specs), N_RUNS,
#     )
#
#     m1_coros = [
#         run_m1_trial(s["group_name"], s["prompt"], s["correct_idx"], s["n_items"], client, sem)
#         for _, s in m1_specs
#     ]
#     m2_coros = [
#         run_m2_task(s["group_name"], s["prompt"], s["actual_positive_indices"], s["n_items"], s["n_expected"], client, sem)
#         for _, s in m2_specs
#     ]
#
#     all_results = await asyncio.gather(*m1_coros, *m2_coros)
#     m1_scores = all_results[:len(m1_coros)]
#     m2_scores = all_results[len(m1_coros):]
#
#     m1_tagged = [
#         (run_idx, spec["group_name"], score)
#         for (run_idx, spec), score in zip(m1_specs, m1_scores)
#     ]
#     m2_tagged = [
#         (run_idx, spec["group_name"], score)
#         for (run_idx, spec), score in zip(m2_specs, m2_scores)
#     ]
#
#     m1_groups, m1_macro = aggregate_by_run(m1_tagged)
#     m2_groups, m2_macro = aggregate_by_run(m2_tagged)
#
#     log.info("[%s] M1 macro: %.1f%% ± %.1f%%", label, m1_macro["mean_accuracy"] * 100, m1_macro["stderr_accuracy"] * 100)
#     log.info("[%s] M2 macro: %.1f%% ± %.1f%%", label, m2_macro["mean_accuracy"] * 100, m2_macro["stderr_accuracy"] * 100)
#
#     return {
#         "method1": {"groups": m1_groups, "macro_avg": m1_macro, "total_trials": len(m1_tagged)},
#         "method2": {"groups": m2_groups, "macro_avg": m2_macro, "total_tasks": len(m2_tagged)},
#     }


async def run_condition(
    label: str,
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    medium_neg_pool: list[dict] | None = None,
    hard_neg_pool: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run N_RUNS of both methods for one condition across three negative-source difficulties.

    easy   — negatives from other named groups in the graph
    medium — negatives from Ungrouped features (in-distribution but unlabeled)
    hard   — negatives from held-out features ranked 100–200 by influence score

    All tasks across all difficulties and runs are generated upfront and fired in one
    asyncio.gather. Results are tagged by (neg_source, run_idx) for aggregation.
    """
    valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}
    empty_method: dict[str, Any] = {"groups": [], "macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}}
    if not valid_groups:
        log.warning("[%s] No valid groups for validation.", label)
        return {src: {"method1": empty_method, "method2": empty_method} for src in ("easy", "medium", "hard")}

    neg_sources: list[tuple[str, list[dict] | None]] = [
        ("easy",   None),
        ("medium", medium_neg_pool),
        ("hard",   hard_neg_pool),
    ]

    all_m1_specs: list[tuple[str, int, dict]] = []  # (neg_source, run_idx, spec)
    all_m2_specs: list[tuple[str, int, dict]] = []

    for neg_source, ext_pool in neg_sources:
        if ext_pool is not None and len(ext_pool) < N_NEG_FEATURES_M1:
            log.warning("[%s/%s] Negative pool too small (%d), skipping.",
                        label, neg_source, len(ext_pool))
            continue
        for run_idx in range(N_RUNS):
            seed = RANDOM_SEED + run_idx
            for spec in generate_m1_tasks(valid_groups, seed, ext_pool):
                all_m1_specs.append((neg_source, run_idx, spec))
            for spec in generate_m2_tasks(valid_groups, seed, ext_pool):
                all_m2_specs.append((neg_source, run_idx, spec))

    log.info("[%s] Launching %d M1 + %d M2 tasks (easy/medium/hard × %d runs)…",
             label, len(all_m1_specs), len(all_m2_specs), N_RUNS)

    m1_coros = [
        run_m1_trial(s["group_name"], s["prompt"], s["correct_idx"], s["n_items"], client, sem)
        for _, _, s in all_m1_specs
    ]
    m2_coros = [
        run_m2_task(s["group_name"], s["prompt"], s["actual_positive_indices"], s["n_items"], s["n_expected"], client, sem)
        for _, _, s in all_m2_specs
    ]

    all_raw = await asyncio.gather(*m1_coros, *m2_coros)
    m1_scores = all_raw[:len(m1_coros)]
    m2_scores = all_raw[len(m1_coros):]

    m1_by_source: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    m2_by_source: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for (src, run_idx, spec), score in zip(all_m1_specs, m1_scores):
        m1_by_source[src].append((run_idx, spec["group_name"], score))
    for (src, run_idx, spec), score in zip(all_m2_specs, m2_scores):
        m2_by_source[src].append((run_idx, spec["group_name"], score))

    results: dict[str, Any] = {}
    for neg_source in ("easy", "medium", "hard"):
        if neg_source not in m1_by_source and neg_source not in m2_by_source:
            results[neg_source] = {"method1": empty_method, "method2": empty_method}
            continue
        m1_groups, m1_macro = aggregate_by_run(m1_by_source.get(neg_source, []))
        m2_groups, m2_macro = aggregate_by_run(m2_by_source.get(neg_source, []))
        log.info("[%s/%s] M1: %.1f%% ± %.1f%%  M2: %.1f%% ± %.1f%%",
                 label, neg_source,
                 m1_macro["mean_accuracy"] * 100, m1_macro["stderr_accuracy"] * 100,
                 m2_macro["mean_accuracy"] * 100, m2_macro["stderr_accuracy"] * 100)
        results[neg_source] = {
            "method1": {"groups": m1_groups, "macro_avg": m1_macro, "total_trials": len(m1_by_source.get(neg_source, []))},
            "method2": {"groups": m2_groups, "macro_avg": m2_macro, "total_tasks": len(m2_by_source.get(neg_source, []))},
        }

    return results


async def run_random_condition(
    group_index: dict[str, list[dict]],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Random baseline: each run gets a fresh shuffle of features across groups.

    Separate from run_condition because the group_index itself changes per run.
    """
    valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}

    m1_specs: list[tuple[int, dict]] = []
    m2_specs: list[tuple[int, dict]] = []

    for run_idx in range(N_RUNS):
        seed = RANDOM_SEED + run_idx
        rng = random.Random(seed)
        shuffled = create_random_group_index(valid_groups, rng)

        for spec in generate_m1_tasks(shuffled, seed):
            m1_specs.append((run_idx, spec))
        for spec in generate_m2_tasks(shuffled, seed):
            m2_specs.append((run_idx, spec))

    log.info("[Random] Launching %d M1 + %d M2 tasks…", len(m1_specs), len(m2_specs))

    m1_coros = [
        run_m1_trial(s["group_name"], s["prompt"], s["correct_idx"], s["n_items"], client, sem)
        for _, s in m1_specs
    ]
    m2_coros = [
        run_m2_task(s["group_name"], s["prompt"], s["actual_positive_indices"], s["n_items"], s["n_expected"], client, sem)
        for _, s in m2_specs
    ]

    all_results = await asyncio.gather(*m1_coros, *m2_coros)
    m1_scores = all_results[:len(m1_coros)]
    m2_scores = all_results[len(m1_coros):]

    m1_tagged = [(ri, s["group_name"], sc) for (ri, s), sc in zip(m1_specs, m1_scores)]
    m2_tagged = [(ri, s["group_name"], sc) for (ri, s), sc in zip(m2_specs, m2_scores)]

    m1_groups, m1_macro = aggregate_by_run(m1_tagged)
    m2_groups, m2_macro = aggregate_by_run(m2_tagged)

    log.info("[Random] M1 macro: %.1f%% ± %.1f%%", m1_macro["mean_accuracy"] * 100, m1_macro["stderr_accuracy"] * 100)
    log.info("[Random] M2 macro: %.1f%% ± %.1f%%", m2_macro["mean_accuracy"] * 100, m2_macro["stderr_accuracy"] * 100)

    # Random only runs easy (shuffled within groups is already the baseline)
    return {
        "easy": {
            "method1": {"groups": m1_groups, "macro_avg": m1_macro, "total_trials": len(m1_tagged)},
            "method2": {"groups": m2_groups, "macro_avg": m2_macro, "total_tasks": len(m2_tagged)},
        }
    }


# ---------------------------------------------------------------------------
# Run description accuracy (D1)
# ---------------------------------------------------------------------------

async def run_description_accuracy(
    features: list[dict],
    prompt_text: str,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Run N_RUNS of Method D1: given feature evidence, pick the correct description.

    Returns a dict with macro accuracy stats, compatible with the report format.
    """
    all_specs: list[tuple[int, dict]] = []
    for run_idx in range(N_RUNS):
        seed = RANDOM_SEED + run_idx
        for spec in generate_d1_tasks(features, prompt_text, seed):
            all_specs.append((run_idx, spec))

    if not all_specs:
        log.warning("[D1] No description accuracy tasks generated.")
        return {"macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}, "total_trials": 0}

    log.info("[D1] Launching %d description accuracy trials across %d runs…",
             len(all_specs), N_RUNS)

    coros = [
        run_m1_trial(
            s["group_name"], s["prompt"], s["correct_idx"], s["n_items"], client, sem
        )
        for _, s in all_specs
    ]

    scores = await asyncio.gather(*coros)
    tagged = [
        (run_idx, spec["group_name"], score)
        for (run_idx, spec), score in zip(all_specs, scores)
    ]

    _, macro = aggregate_by_run(tagged)
    log.info("[D1] Description accuracy: %.1f%% ± %.1f%%",
             macro["mean_accuracy"] * 100, macro["stderr_accuracy"] * 100)

    return {
        "macro_avg": macro,
        "total_trials": len(tagged),
    }


# ---------------------------------------------------------------------------
# Run description snippet accuracy (D2)
# ---------------------------------------------------------------------------

async def run_description_snippet_accuracy(
    features: list[dict],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Run N_RUNS of Method D2: given a feature description, pick the correct activating snippets.

    Returns a dict with macro accuracy stats, compatible with the report format.
    """
    all_specs: list[tuple[int, dict]] = []
    for run_idx in range(N_RUNS):
        seed = RANDOM_SEED + run_idx
        for spec in generate_d2_tasks(features, seed):
            all_specs.append((run_idx, spec))

    if not all_specs:
        log.warning("[D2] No description snippet accuracy tasks generated.")
        return {"macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}, "total_tasks": 0}

    log.info("[D2] Launching %d description snippet accuracy tasks across %d runs…",
             len(all_specs), N_RUNS)

    coros = [
        run_m2_task(
            s["group_name"], s["prompt"], s["actual_positive_indices"],
            s["n_items"], s["n_expected"], client, sem,
        )
        for _, s in all_specs
    ]

    scores = await asyncio.gather(*coros)
    tagged = [
        (run_idx, spec["group_name"], score)
        for (run_idx, spec), score in zip(all_specs, scores)
    ]

    _, macro = aggregate_by_run(tagged)
    log.info("[D2] Description snippet accuracy: %.1f%% ± %.1f%%",
             macro["mean_accuracy"] * 100, macro["stderr_accuracy"] * 100)

    return {
        "macro_avg": macro,
        "total_tasks": len(tagged),
    }


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _fmt_macro(d: dict) -> str:
    return f"{d.get('mean_accuracy', 0.0):.1%} ± {d.get('stderr_accuracy', 0.0):.1%}"


def _get_macro(report: dict, condition: str, neg_source: str, method: str) -> dict:
    return (
        report.get(condition, {})
        .get(neg_source, {})
        .get(method, {})
        .get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
    )


def write_markdown_report(report: dict, path: Path) -> None:
    """Write a flat, human-readable markdown validation summary."""

    n_runs = report.get("n_runs", N_RUNS)
    chance_m1 = 1 / (1 + N_NEG_FEATURES_M1)
    chance_m2 = N_POS_SNIPPETS_M2 / (N_POS_SNIPPETS_M2 + N_NEG_SNIPPETS_M2)
    chance_d1 = 1 / (1 + N_NEG_DESCS_D1)
    chance_d2 = N_POS_SNIPPETS_D2 / (N_POS_SNIPPETS_D2 + N_NEG_SNIPPETS_D2)

    L: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    L += [
        "# Validation Report",
        "",
        f"**Prompt:** {report.get('prompt', 'Unknown')}",
        f"**Description variant:** `{DESCRIPTION_VARIANT}`  |  **Grouping variant:** `{GROUPING_VARIANT}`",
        f"**Date:** {report.get('timestamp', '')}  |  **Runs:** {n_runs}",
    ]
    auto_cov = report.get("auto", {}).get("attribution_coverage")
    if auto_cov is not None:
        L.append(f"**Auto Attribution Coverage:** {auto_cov:.1%}")
    L += ["", "---", ""]

    # ── Method Scores ────────────────────────────────────────────────────────
    L += [
        "## Method Scores",
        "",
        "Scores are **mean ± stderr** across runs.  "
        "Higher is better. Random baseline is shuffled group assignments.",
        "",
    ]

    # M1 / M2 — three difficulty levels
    L += [
        "### M1 and M2 — Group Validation",
        "",
        "Negative sources:  "
        "**easy** = other named groups  |  "
        "**medium** = Ungrouped features  |  "
        f"**hard** = features outside the top-{GROUPING_TOP_K_SEED} seed window",
        "",
        f"| | Easy | Medium | Hard | Chance |",
        f"|:--|-----:|-------:|-----:|-------:|",
    ]

    m1 = {s: _get_macro(report, "auto", s, "method1") for s in ("easy", "medium", "hard")}
    m2 = {s: _get_macro(report, "auto", s, "method2") for s in ("easy", "medium", "hard")}
    rm1 = _get_macro(report, "random", "easy", "method1")
    rm2 = _get_macro(report, "random", "easy", "method2")

    L.append(f"| **M1** Feature ID (1-in-{1+N_NEG_FEATURES_M1}) "
             f"| {_fmt_macro(m1['easy'])} | {_fmt_macro(m1['medium'])} | {_fmt_macro(m1['hard'])} "
             f"| {chance_m1:.0%} |")
    L.append(f"| **M2** Text Match ({N_POS_SNIPPETS_M2}-in-{N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2}) "
             f"| {_fmt_macro(m2['easy'])} | {_fmt_macro(m2['medium'])} | {_fmt_macro(m2['hard'])} "
             f"| {chance_m2:.0%} |")
    L.append(f"| *Random M1* | {_fmt_macro(rm1)} | — | — | {chance_m1:.0%} |")
    L.append(f"| *Random M2* | {_fmt_macro(rm2)} | — | — | {chance_m2:.0%} |")

    m1_trials = report.get("auto", {}).get("easy", {}).get("method1", {}).get("total_trials", 0)
    m2_tasks  = report.get("auto", {}).get("easy", {}).get("method2", {}).get("total_tasks", 0)
    L += ["", f"*Easy: {m1_trials} M1 trials · {m2_tasks} M2 tasks*", ""]

    # D1 / D2 — no difficulty levels
    L += [
        "### D1 and D2 — Description Validation",
        "",
        f"| | Score | Chance | Tasks |",
        f"|:--|------:|-------:|------:|",
    ]

    d1 = report.get("description_accuracy", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
    d1_n = report.get("description_accuracy", {}).get("total_trials", 0)
    d2 = report.get("description_snippet_accuracy", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
    d2_n = report.get("description_snippet_accuracy", {}).get("total_tasks", 0)

    L.append(f"| **D1** Evidence → pick correct description (1-in-{1+N_NEG_DESCS_D1}) "
             f"| {_fmt_macro(d1)} | {chance_d1:.0%} | {d1_n} |")
    L.append(f"| **D2** Description → pick correct snippets ({N_POS_SNIPPETS_D2}-in-{N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2}) "
             f"| {_fmt_macro(d2)} | {chance_d2:.0%} | {d2_n} |")
    L += [""]

    # Manual (optional)
    if "manual" in report:
        man_cov = report["manual"].get("attribution_coverage")
        L += [
            "### Manual Groups (comparison)",
            "",
        ]
        if man_cov is not None:
            L.append(f"**Manual Attribution Coverage:** {man_cov:.1%}")
        L += [
            "",
            f"| | Easy | Medium | Hard | Chance |",
            f"|:--|-----:|-------:|-----:|-------:|",
        ]
        mm1 = {s: _get_macro(report, "manual", s, "method1") for s in ("easy", "medium", "hard")}
        mm2 = {s: _get_macro(report, "manual", s, "method2") for s in ("easy", "medium", "hard")}
        L.append(f"| **M1** Feature ID "
                 f"| {_fmt_macro(mm1['easy'])} | {_fmt_macro(mm1['medium'])} | {_fmt_macro(mm1['hard'])} "
                 f"| {chance_m1:.0%} |")
        L.append(f"| **M2** Text Match "
                 f"| {_fmt_macro(mm2['easy'])} | {_fmt_macro(mm2['medium'])} | {_fmt_macro(mm2['hard'])} "
                 f"| {chance_m2:.0%} |")
        L += [""]

    L += ["---", ""]

    # ── Per-Group Detail ─────────────────────────────────────────────────────
    L += [
        "## Per-Group Detail",
        "",
        "Auto grouping, easy difficulty. Sorted by accuracy, highest first.",
        "",
    ]

    m1_groups = report.get("auto", {}).get("easy", {}).get("method1", {}).get("groups", [])
    m2_groups = report.get("auto", {}).get("easy", {}).get("method2", {}).get("groups", [])

    if m1_groups:
        L += [
            "### M1 — Feature Identification",
            "",
            "| Group | Accuracy | ±Stderr | Runs | Trials |",
            "|:------|----------:|--------:|-----:|-------:|",
        ]
        for s in sorted(m1_groups, key=lambda x: x["mean_accuracy"], reverse=True):
            L.append(f"| {s['group'][:55]} "
                     f"| {s['mean_accuracy']:.1%} "
                     f"| {s['stderr_accuracy']:.1%} "
                     f"| {s['n_runs']} "
                     f"| {s['total_trials']} |")
        L += [""]

    if m2_groups:
        L += [
            "### M2 — Text Snippet Match",
            "",
            "| Group | Accuracy | ±Stderr | Runs | Tasks |",
            "|:------|----------:|--------:|-----:|------:|",
        ]
        for s in sorted(m2_groups, key=lambda x: x["mean_accuracy"], reverse=True):
            L.append(f"| {s['group'][:55]} "
                     f"| {s['mean_accuracy']:.1%} "
                     f"| {s['stderr_accuracy']:.1%} "
                     f"| {s['n_runs']} "
                     f"| {s['total_trials']} |")
        L += [""]

    # Medium and hard — macro only (not per-group, too verbose)
    L += [
        "### Medium and Hard Difficulties (Auto — macro only)",
        "",
        "| Difficulty | M1 | M2 |",
        "|:-----------|---:|---:|",
        f"| Medium | {_fmt_macro(m1['medium'])} | {_fmt_macro(m2['medium'])} |",
        f"| Hard   | {_fmt_macro(m1['hard'])} | {_fmt_macro(m2['hard'])} |",
        "",
        "---",
        "",
        f"*Generated {report.get('timestamp', '')} · "
        f"desc=`{DESCRIPTION_VARIANT}` · group=`{GROUPING_VARIANT}`*",
    ]

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Markdown report saved → %s", path)


# ---------------------------------------------------------------------------
# Read prompt from graph
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async() -> None:
    features, groups = load_data()
    group_index = build_group_index(features, groups)

    valid_count = sum(1 for v in group_index.values() if len(v) >= MIN_GROUP_SIZE)
    log.info(
        "Loaded %d groups total; %d have ≥%d features for validation.",
        len(group_index), valid_count, MIN_GROUP_SIZE,
    )

    if valid_count == 0:
        log.error("No groups with enough features to validate. Exiting.")
        sys.exit(1)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    auto_coverage = compute_attribution_coverage(features, groups)
    log.info("Auto attribution coverage: %.1f%%", auto_coverage * 100)

    # Build medium and hard negative pools
    medium_neg_pool = build_medium_neg_pool(features, groups)
    hard_neg_pool = build_hard_neg_pool(features)
    log.info("Negative pools — medium: %d features, hard: %d features",
             len(medium_neg_pool), len(hard_neg_pool))

    # Read prompt text for D1 evidence formatting
    prompt_text = _read_prompt_from_graph()

    auto_result, random_result, d1_result, d2_result = await asyncio.gather(
        run_condition("Auto", group_index, client, sem, medium_neg_pool, hard_neg_pool),
        run_random_condition(group_index, client, sem),
        run_description_accuracy(features, prompt_text, client, sem),
        run_description_snippet_accuracy(features, client, sem),
    )

    # Manual groups (optional)
    manual_result: dict[str, Any] | None = None
    manual_coverage: float | None = None

    manual_group_index = load_manual_group_index(features)
    if manual_group_index:
        with open(MANUAL_GROUPS_FILE) as f:
            manual_groups_raw: dict[str, str] = json.load(f)
        manual_coverage = compute_attribution_coverage(features, manual_groups_raw)
        log.info("Manual attribution coverage: %.1f%%", manual_coverage * 100)

        manual_medium = build_medium_neg_pool(features, manual_groups_raw)
        manual_group_index = await regenerate_group_names(manual_group_index, client, sem)
        manual_result = await run_condition("Manual", manual_group_index, client, sem, manual_medium, hard_neg_pool)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    prompt = prompt_text  # already loaded above for D1
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report: dict[str, Any] = {
        "prompt": prompt,
        "timestamp": timestamp,
        "n_runs": N_RUNS,
        "design": {
            "method1": f"1-in-{1+N_NEG_FEATURES_M1} feature identification (chance={1/(1+N_NEG_FEATURES_M1):.0%})",
            "method2": f"{N_POS_SNIPPETS_M2}-in-{N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2} text snippet matching (chance={N_POS_SNIPPETS_M2/(N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2):.0%})",
            "method_d1": f"1-in-{1+N_NEG_DESCS_D1} description accuracy (chance={1/(1+N_NEG_DESCS_D1):.0%})",
            "method_d2": f"{N_POS_SNIPPETS_D2}-in-{N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2} description snippet match (chance={N_POS_SNIPPETS_D2/(N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2):.0%})",
            "aggregation": "per-group run-level means, then mean ± stderr across runs",
            "neg_sources": "easy=other named groups, medium=Ungrouped features, hard=features ranked 100-200 by influence",
            "m2_output_validation": "response must contain exactly N_POS unique indices in [1,N]; discarded otherwise (score=0)",
            "label_regeneration": "manual only; auto uses pipeline names",
        },
        "auto": {
            **auto_result,
            "attribution_coverage": auto_coverage,
        },
        "random": random_result,
        "description_accuracy": d1_result,
        "description_snippet_accuracy": d2_result,
    }
    if manual_result:
        report["manual"] = {
            **manual_result,
            "attribution_coverage": manual_coverage,
        }

    with open(VALIDATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Validation report saved → %s", VALIDATION_REPORT_FILE)

    write_markdown_report(report, VALIDATION_MARKDOWN_FILE)

    # Append to history
    history: list[dict] = []
    if VALIDATION_HISTORY_FILE.exists():
        with open(VALIDATION_HISTORY_FILE) as f:
            history = json.load(f)
    history.append(report)
    with open(VALIDATION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Appended to history (%d total) → %s", len(history), VALIDATION_HISTORY_FILE)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    cond_keys: list[tuple[str, str]] = [("Auto", "auto"), ("Random", "random")]
    if "manual" in report:
        cond_keys.append(("Manual", "manual"))

    chance_m1 = 1 / (1 + N_NEG_FEATURES_M1)
    chance_m2 = N_POS_SNIPPETS_M2 / (N_POS_SNIPPETS_M2 + N_NEG_SNIPPETS_M2)
    chance_d1 = 1 / (1 + N_NEG_DESCS_D1)
    chance_d2 = N_POS_SNIPPETS_D2 / (N_POS_SNIPPETS_D2 + N_NEG_SNIPPETS_D2)

    print(f"\n{'='*80}")
    print(f"  VALIDATION SUMMARY ({N_RUNS} runs, run-level aggregation)")
    print(f"  M1: 1-in-{1+N_NEG_FEATURES_M1} feature ID (chance={chance_m1:.0%})")
    print(f"  M2: {N_POS_SNIPPETS_M2}-in-{N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2} text match (chance={chance_m2:.0%})")
    print(f"  D1: 1-in-{1+N_NEG_DESCS_D1} description accuracy (chance={chance_d1:.0%})")
    print(f"  D2: {N_POS_SNIPPETS_D2}-in-{N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2} description snippet match (chance={chance_d2:.0%})")
    print(f"  Neg sources: easy=other groups | medium=Ungrouped | hard=unseen features (beyond Phase 1 seed of {GROUPING_TOP_K_SEED})")
    print(f"{'='*80}")

    # Description-level methods — printed first
    d1_macro = report.get("description_accuracy", {}).get("macro_avg", {})
    d1_trials = report.get("description_accuracy", {}).get("total_trials", 0)
    print(f"\n  Description Accuracy  (D1): "
          f"{d1_macro.get('mean_accuracy', 0.0):.1%} ± {d1_macro.get('stderr_accuracy', 0.0):.1%}  "
          f"({d1_trials} trials, chance={chance_d1:.0%})")

    d2_macro = report.get("description_snippet_accuracy", {}).get("macro_avg", {})
    d2_tasks = report.get("description_snippet_accuracy", {}).get("total_tasks", 0)
    print(f"  Description Snippets  (D2): "
          f"{d2_macro.get('mean_accuracy', 0.0):.1%} ± {d2_macro.get('stderr_accuracy', 0.0):.1%}  "
          f"({d2_tasks} tasks, chance={chance_d2:.0%})")

    print(f"\n{'Condition':<16}  {'M1 Accuracy':>16}  {'M2 Accuracy':>16}  {'Coverage':>9}")
    print("-" * 70)

    for cond_label, key in cond_keys:
        cov = report[key].get("attribution_coverage")
        cov_str = f"{cov:.1%}" if cov is not None else "    N/A"
        sources = ("easy", "medium", "hard") if key != "random" else ("easy",)
        for src in sources:
            src_data = report[key].get(src, {})
            m1a = src_data.get("method1", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
            m2a = src_data.get("method2", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
            row_label = f"{cond_label}/{src}"
            cov_col = cov_str if src == sources[0] else "       -"
            print(
                f"{row_label:<16}  "
                f"{m1a['mean_accuracy']:>6.1%} ± {m1a['stderr_accuracy']:.1%}    "
                f"{m2a['mean_accuracy']:>6.1%} ± {m2a['stderr_accuracy']:.1%}    "
                f"{cov_col:>9}"
            )

    print(f"\n--- Per-group M1 Accuracy — Auto/easy (run-level mean ± stderr) ---")
    for s in sorted(report["auto"].get("easy", {}).get("method1", {}).get("groups", []),
                    key=lambda x: x["mean_accuracy"], reverse=True):
        print(
            f"  {s['group'][:40]:<40}  "
            f"{s['mean_accuracy']:.1%} ± {s['stderr_accuracy']:.1%}  "
            f"({s['n_runs']} runs, {s['total_trials']} trials)"
        )

    print(f"\n--- Per-group M2 Accuracy — Auto/easy (run-level mean ± stderr) ---")
    for s in sorted(report["auto"].get("easy", {}).get("method2", {}).get("groups", []),
                    key=lambda x: x["mean_accuracy"], reverse=True):
        print(
            f"  {s['group'][:40]:<40}  "
            f"{s['mean_accuracy']:.1%} ± {s['stderr_accuracy']:.1%}  "
            f"({s['n_runs']} runs, {s['total_trials']} trials)"
        )

    print(f"{'='*80}\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()