"""
Anthropic-style validation — Evaluate group/description quality across four
canonical conditions, with Neuronpedia-derived human labels as a baseline.

Conditions (toggle via VALIDATION_CONDITIONS env var, default: all four):
  - random                  : ours-full groups, features shuffled across them
  - human                   : manual_groups.json (Neuronpedia share URL → fetcher)
  - ours-full               : auto pipeline output AFTER phase-3 reconciliation
  - ours-no-reconciliation  : auto pipeline output BEFORE phase-3 (snapshot file)

Other env knobs:
  VALIDATION_MIN_GROUP_SIZE  groups with < N members are excluded (both sides).
  VALIDATION_STANDARDIZE_NAMES=1   regenerate group labels via LLM for ALL
                                   non-random conditions, isolating naming
                                   quality from grouping quality.
  VALIDATION_DIFFICULTY      "medium" (default) | "both" (legacy easy+medium).

Output files (suffixed `_anthropic` to coexist with validate_groups.py):
  validation_report_<desc>_<group>_anthropic.json
  validation_report_<desc>_<group>_anthropic.md
  validation_history_<desc>_<group>_anthropic.json

For sweeping over min_group_size, see run_validation_sweep.py.

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

Reads:  artifacts/<slug>/feature_descriptions_<desc>.json
        artifacts/<slug>/feature_groups_<desc>_<group>.json
        artifacts/<slug>/feature_groups_<desc>_<group>_pre3.json  (optional, for ours-no-reconciliation)
        artifacts/<slug>/manual_groups.json                       (optional, for human)
Writes: artifacts/<slug>/validation_report_<desc>_<group>_anthropic.json
        artifacts/<slug>/validation_history_<desc>_<group>_anthropic.json

Usage:
    OPENAI_API_KEY=sk-xxx CURRENT_SLUG=<slug> python validate_anthropic_groups.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import (
    DESCRIPTION_VARIANT,
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    FEATURE_GROUPS_PRE3_FILE,
    GRAPH_FILE,
    GROUPING_MODEL,
    GROUPING_TOP_K_SEED,
    GROUPING_VARIANT,
    MANUAL_GROUPS_FILE,
    VALIDATION_CONDITIONS,
    VALIDATION_DIFFICULTY,
    VALIDATION_HISTORY_FILE,
    VALIDATION_MIN_GROUP_SIZE,
    VALIDATION_REPORT_FILE,
    VALIDATION_STANDARDIZE_NAMES,
    setup_logging,
)

log = setup_logging()

# This script writes its own report files (suffixed `_anthropic`) so it can
# coexist with the original validate_groups.py without overwriting its output.
VALIDATION_REPORT_FILE = VALIDATION_REPORT_FILE.with_name(
    VALIDATION_REPORT_FILE.stem + "_anthropic" + VALIDATION_REPORT_FILE.suffix
)
VALIDATION_HISTORY_FILE = VALIDATION_HISTORY_FILE.with_name(
    VALIDATION_HISTORY_FILE.stem + "_anthropic" + VALIDATION_HISTORY_FILE.suffix
)
VALIDATION_MARKDOWN_FILE = VALIDATION_REPORT_FILE.with_suffix(".md")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALIDATION_MODEL = GROUPING_MODEL
CONCURRENCY_LIMIT = 500
# MIN_GROUP_SIZE controls eligibility for M1/M2 trials on BOTH sides (auto + manual).
# Sweep externally by setting VALIDATION_MIN_GROUP_SIZE in the env.
MIN_GROUP_SIZE = VALIDATION_MIN_GROUP_SIZE
# Difficulty: "medium" (default) skips the old easy split. "both" preserves legacy behaviour.
DIFFICULTY = VALIDATION_DIFFICULTY if VALIDATION_DIFFICULTY in ("medium", "both") else "medium"
# Default condition set when VALIDATION_CONDITIONS is unset/empty.
ALL_CONDITIONS = ("random", "human", "ours-full", "ours-no-reconciliation")
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


def load_condition_groups(
    condition: str,
    features: list[dict],
) -> tuple[dict[str, str], dict[str, list[dict]]] | None:
    """Load (groups_raw, group_index) for one of the four canonical conditions.

    Conditions:
      - human                    : reads MANUAL_GROUPS_FILE (Neuronpedia-derived)
      - ours-full                : reads FEATURE_GROUPS_FILE (post phase 3)
      - ours-no-reconciliation   : reads FEATURE_GROUPS_PRE3_FILE
      - random                   : not handled here — caller shuffles ours-full

    Returns None when the required file is missing or yields no usable groups.
    """
    if condition == "human":
        if not MANUAL_GROUPS_FILE.exists():
            log.info("[human] %s missing — skipping condition.", MANUAL_GROUPS_FILE)
            return None
        with open(MANUAL_GROUPS_FILE) as f:
            groups: dict[str, str] = json.load(f)
        index = build_group_index(features, groups)
        if not index:
            log.warning("[human] No usable groups in %s.", MANUAL_GROUPS_FILE)
            return None
        return groups, index

    if condition == "ours-full":
        if not FEATURE_GROUPS_FILE.exists():
            log.error("[ours-full] %s missing — run generate_supernodes.py first.", FEATURE_GROUPS_FILE)
            return None
        with open(FEATURE_GROUPS_FILE) as f:
            groups = json.load(f)
        index = build_group_index(features, groups)
        return (groups, index) if index else None

    if condition == "ours-no-reconciliation":
        if not FEATURE_GROUPS_PRE3_FILE.exists():
            log.warning(
                "[ours-no-reconciliation] %s missing — re-run generate_supernodes.py "
                "after the pre-phase-3 snapshot was added.",
                FEATURE_GROUPS_PRE3_FILE,
            )
            return None
        with open(FEATURE_GROUPS_PRE3_FILE) as f:
            groups = json.load(f)
        index = build_group_index(features, groups)
        return (groups, index) if index else None

    log.error("Unknown condition '%s'.", condition)
    return None


# ---------------------------------------------------------------------------
# Negative pool builders
# ---------------------------------------------------------------------------

def build_medium_neg_pool(features: list[dict], groups: dict[str, str]) -> list[dict]:
    """Features that act as Ungrouped negatives — have a description, but are
    not part of any named supernode (excluding embedding/logit special groups).

    A feature is treated as a "medium" negative if it is either explicitly
    labeled Ungrouped *or* missing from `groups` entirely (the latter is the
    common case for human-labeled manual files, which only enumerate positives).
    Embedding/logit nodes are excluded — they're real labels, not negatives.
    """
    pool: list[dict] = []
    for f in features:
        fid = f.get("id")
        if not fid:
            continue
        gname = groups.get(fid, "Ungrouped")
        if gname.startswith(('Emb: "', 'Output: "')):
            continue
        if gname == "Ungrouped" and f.get("generated_description"):
            pool.append(f)
    return pool


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
) -> dict[str, Any]:
    """
    Run N_RUNS of both methods for one condition across two negative-source difficulties.

    easy   — negatives from other named groups in the graph
    medium — negatives from Ungrouped features (in-distribution but unlabeled)

    All tasks across all difficulties and runs are generated upfront and fired in one
    asyncio.gather. Results are tagged by (neg_source, run_idx) for aggregation.
    """
    valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}
    empty_method: dict[str, Any] = {"groups": [], "macro_avg": {"mean_accuracy": 0.0, "stderr_accuracy": 0.0}}
    expected_sources = ("easy", "medium") if DIFFICULTY == "both" else ("medium",)
    if not valid_groups:
        log.warning("[%s] No valid groups for validation.", label)
        return {src: {"method1": empty_method, "method2": empty_method} for src in expected_sources}

    neg_sources: list[tuple[str, list[dict] | None]] = []
    if DIFFICULTY == "both":
        neg_sources.append(("easy", None))
    neg_sources.append(("medium", medium_neg_pool))

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

    log.info("[%s] Launching %d M1 + %d M2 tasks (easy/medium × %d runs)…",
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
    for neg_source in expected_sources:
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
    medium_neg_pool: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Random baseline: each run gets a fresh shuffle of features across groups.

    Negatives follow DIFFICULTY: when medium-only, we use the same Ungrouped
    pool as the other conditions so the comparison is apples-to-apples. When
    "both", we keep legacy easy-difficulty (other-group) behaviour.
    """
    valid_groups = {k: v for k, v in group_index.items() if len(v) >= MIN_GROUP_SIZE}
    use_medium = (DIFFICULTY == "medium")
    neg_pool_arg = medium_neg_pool if use_medium else None
    neg_source_label = "medium" if use_medium else "easy"

    m1_specs: list[tuple[int, dict]] = []
    m2_specs: list[tuple[int, dict]] = []

    for run_idx in range(N_RUNS):
        seed = RANDOM_SEED + run_idx
        rng = random.Random(seed)
        shuffled = create_random_group_index(valid_groups, rng)

        for spec in generate_m1_tasks(shuffled, seed, neg_pool_arg):
            m1_specs.append((run_idx, spec))
        for spec in generate_m2_tasks(shuffled, seed, neg_pool_arg):
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

    log.info("[Random/%s] M1 macro: %.1f%% ± %.1f%%", neg_source_label, m1_macro["mean_accuracy"] * 100, m1_macro["stderr_accuracy"] * 100)
    log.info("[Random/%s] M2 macro: %.1f%% ± %.1f%%", neg_source_label, m2_macro["mean_accuracy"] * 100, m2_macro["stderr_accuracy"] * 100)

    return {
        neg_source_label: {
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


def _macro_for(report: dict, condition: str, neg_source: str, method: str) -> dict:
    return (
        report.get("conditions", {})
        .get(condition, {})
        .get(neg_source, {})
        .get(method, {})
        .get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
    )


def write_markdown_report(report: dict, path: Path) -> None:
    """Write a flat, human-readable markdown summary of a multi-condition report."""

    n_runs = report.get("n_runs", N_RUNS)
    chance_m1 = 1 / (1 + N_NEG_FEATURES_M1)
    chance_m2 = N_POS_SNIPPETS_M2 / (N_POS_SNIPPETS_M2 + N_NEG_SNIPPETS_M2)
    chance_d1 = 1 / (1 + N_NEG_DESCS_D1)
    chance_d2 = N_POS_SNIPPETS_D2 / (N_POS_SNIPPETS_D2 + N_NEG_SNIPPETS_D2)

    difficulty = report.get("difficulty", DIFFICULTY)
    sources = ("easy", "medium") if difficulty == "both" else ("medium",)
    conditions = report.get("conditions", {})

    L: list[str] = [
        "# Validation Report",
        "",
        f"**Prompt:** {report.get('prompt', 'Unknown')}",
        f"**Description variant:** `{DESCRIPTION_VARIANT}`  |  **Grouping variant:** `{GROUPING_VARIANT}`",
        f"**Min group size:** {report.get('min_group_size', MIN_GROUP_SIZE)}  |  "
        f"**Difficulty:** {difficulty}  |  "
        f"**Standardized names:** {report.get('standardize_names', VALIDATION_STANDARDIZE_NAMES)}",
        f"**Date:** {report.get('timestamp', '')}  |  **Runs:** {n_runs}",
        "",
        "---",
        "",
        "## Method Scores by Condition",
        "",
        f"M1 = 1-in-{1+N_NEG_FEATURES_M1} feature ID (chance {chance_m1:.0%}). "
        f"M2 = {N_POS_SNIPPETS_M2}-in-{N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2} text match (chance {chance_m2:.0%}). "
        "Scores are mean ± stderr across runs.",
        "",
    ]

    header_cells = ["Condition"] + [f"M1 ({s})" for s in sources] + [f"M2 ({s})" for s in sources] + ["Coverage", "Groups"]
    L.append("| " + " | ".join(header_cells) + " |")
    L.append("|" + "|".join([":--"] + ["---:"] * (len(header_cells) - 1)) + "|")

    for cond in ("random", "human", "ours-no-reconciliation", "ours-full"):
        if cond not in conditions:
            continue
        c = conditions[cond]
        if c.get("skipped"):
            L.append(f"| **{cond}** | _skipped: {c.get('reason','')}_ |")
            continue
        cov = c.get("attribution_coverage")
        cov_str = f"{cov:.1%}" if cov is not None else "—"
        n_valid = c.get("n_groups_valid")
        groups_str = f"{n_valid}/{c.get('n_groups_total', '?')}" if n_valid is not None else "—"
        cells = [f"**{cond}**"]
        for s in sources:
            cells.append(_fmt_macro(_macro_for(report, cond, s, "method1")))
        for s in sources:
            cells.append(_fmt_macro(_macro_for(report, cond, s, "method2")))
        cells += [cov_str, groups_str]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "---", ""]

    L += [
        "## Description-Level Validation",
        "",
        "| Method | Score | Chance | N |",
        "|:--|---:|---:|---:|",
    ]
    d1 = report.get("description_accuracy", {}).get("macro_avg", {})
    d1_n = report.get("description_accuracy", {}).get("total_trials", 0)
    d2 = report.get("description_snippet_accuracy", {}).get("macro_avg", {})
    d2_n = report.get("description_snippet_accuracy", {}).get("total_tasks", 0)
    L.append(f"| **D1** Evidence -> pick description (1-in-{1+N_NEG_DESCS_D1}) "
             f"| {_fmt_macro(d1)} | {chance_d1:.0%} | {d1_n} |")
    L.append(f"| **D2** Description -> pick snippets ({N_POS_SNIPPETS_D2}-in-{N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2}) "
             f"| {_fmt_macro(d2)} | {chance_d2:.0%} | {d2_n} |")
    L += ["", "---", ""]

    detail_cond = "ours-full" if "ours-full" in conditions else next(iter(conditions), None)
    if detail_cond:
        for s in sources:
            m1_groups = conditions.get(detail_cond, {}).get(s, {}).get("method1", {}).get("groups", [])
            if not m1_groups:
                continue
            L += [
                f"## Per-Group Detail — {detail_cond} / {s}",
                "",
                "| Group | M1 Accuracy | ±Stderr | Trials |",
                "|:--|---:|---:|---:|",
            ]
            for sg in sorted(m1_groups, key=lambda x: x["mean_accuracy"], reverse=True):
                L.append(
                    f"| {sg['group'][:55]} "
                    f"| {sg['mean_accuracy']:.1%} "
                    f"| {sg['stderr_accuracy']:.1%} "
                    f"| {sg['total_trials']} |"
                )
            L += [""]
            break

    L += [
        "---",
        "",
        f"*Generated {report.get('timestamp', '')} · "
        f"desc=`{DESCRIPTION_VARIANT}` · group=`{GROUPING_VARIANT}` · "
        f"min_group_size={report.get('min_group_size', MIN_GROUP_SIZE)}*",
    ]

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Markdown report saved → %s", path)


# ---------------------------------------------------------------------------
# Helpers — prompt loading + condition orchestration
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


def _rebuild_groups_raw_from_index(
    new_index: dict[str, list[dict]],
    old_groups_raw: dict[str, str],
) -> dict[str, str]:
    """After regenerate_group_names, derive a fresh {fid: label} mapping.

    Features in `new_index` get the new label. Features absent from `new_index`
    (Ungrouped, Emb, Output) keep their original labels so downstream filters
    work correctly.
    """
    new_raw: dict[str, str] = {}
    for new_name, feats in new_index.items():
        for f in feats:
            fid = f.get("id")
            if fid:
                new_raw[fid] = new_name
    for fid, label in old_groups_raw.items():
        new_raw.setdefault(fid, label)
    return new_raw


def _count_named_groups(groups_raw: dict[str, str]) -> int:
    return len({
        g for g in groups_raw.values()
        if g != "Ungrouped" and not g.startswith(('Emb: "', 'Output: "'))
    })


def _resolve_requested_conditions() -> list[str]:
    raw = VALIDATION_CONDITIONS.strip()
    if not raw:
        return list(ALL_CONDITIONS)
    requested = [c.strip() for c in raw.split(",") if c.strip()]
    invalid = [c for c in requested if c not in ALL_CONDITIONS]
    if invalid:
        log.error("Unknown VALIDATION_CONDITIONS values: %s. Valid: %s", invalid, ALL_CONDITIONS)
        sys.exit(1)
    return requested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async() -> None:
    if not FEATURE_DESCRIPTIONS_FILE.exists():
        log.error("Missing %s — run generate_description.py first.", FEATURE_DESCRIPTIONS_FILE)
        sys.exit(1)
    with open(FEATURE_DESCRIPTIONS_FILE) as f:
        features: list[dict] = json.load(f)

    requested = _resolve_requested_conditions()
    log.info("Conditions: %s", requested)
    log.info(
        "Min group size: %d  |  Difficulty: %s  |  Standardize names: %s",
        MIN_GROUP_SIZE, DIFFICULTY, VALIDATION_STANDARDIZE_NAMES,
    )

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    prompt_text = _read_prompt_from_graph()

    d1_task = run_description_accuracy(features, prompt_text, client, sem)
    d2_task = run_description_snippet_accuracy(features, client, sem)

    loaded: dict[str, tuple[dict[str, str], dict[str, list[dict]]]] = {}
    skipped: dict[str, str] = {}
    for cond in requested:
        if cond == "random":
            base = load_condition_groups("ours-full", features)
            if base is None:
                skipped[cond] = "ours-full base groups unavailable"
                continue
            loaded[cond] = base
        else:
            res = load_condition_groups(cond, features)
            if res is None:
                skipped[cond] = "data file missing"
                continue
            loaded[cond] = res

    if VALIDATION_STANDARDIZE_NAMES:
        log.info("Standardizing group names via LLM (applied to non-random conditions)…")
        for cond in list(loaded.keys()):
            if cond == "random":
                continue
            groups_raw, group_index = loaded[cond]
            new_index = await regenerate_group_names(group_index, client, sem)
            new_raw = _rebuild_groups_raw_from_index(new_index, groups_raw)
            loaded[cond] = (new_raw, new_index)

    coros: list[Any] = []
    cond_order: list[str] = []
    for cond in requested:
        if cond not in loaded:
            continue
        groups_raw, group_index = loaded[cond]
        medium_neg_pool = build_medium_neg_pool(features, groups_raw)
        log.info("[%s] Negative pool (medium): %d features", cond, len(medium_neg_pool))
        if cond == "random":
            coros.append(run_random_condition(group_index, client, sem, medium_neg_pool))
        else:
            coros.append(run_condition(cond, group_index, client, sem, medium_neg_pool))
        cond_order.append(cond)

    if coros:
        gathered = await asyncio.gather(*coros, d1_task, d2_task)
    else:
        gathered = [None] * len(cond_order) + list(await asyncio.gather(d1_task, d2_task))
    n_cond = len(cond_order)
    cond_results_raw = gathered[:n_cond]
    d1_result, d2_result = gathered[n_cond], gathered[n_cond + 1]

    condition_results: dict[str, Any] = {}
    for cond, out in zip(cond_order, cond_results_raw):
        groups_raw, group_index = loaded[cond]
        condition_results[cond] = {
            **out,
            "attribution_coverage": compute_attribution_coverage(features, groups_raw),
            "n_groups_total": _count_named_groups(groups_raw),
            "n_groups_valid": sum(1 for v in group_index.values() if len(v) >= MIN_GROUP_SIZE),
        }
    for cond, reason in skipped.items():
        condition_results[cond] = {"skipped": True, "reason": reason}

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "prompt": prompt_text,
        "timestamp": timestamp,
        "n_runs": N_RUNS,
        "min_group_size": MIN_GROUP_SIZE,
        "difficulty": DIFFICULTY,
        "standardize_names": VALIDATION_STANDARDIZE_NAMES,
        "design": {
            "method1": f"1-in-{1+N_NEG_FEATURES_M1} feature identification (chance={1/(1+N_NEG_FEATURES_M1):.0%})",
            "method2": f"{N_POS_SNIPPETS_M2}-in-{N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2} text snippet matching (chance={N_POS_SNIPPETS_M2/(N_POS_SNIPPETS_M2+N_NEG_SNIPPETS_M2):.0%})",
            "method_d1": f"1-in-{1+N_NEG_DESCS_D1} description accuracy (chance={1/(1+N_NEG_DESCS_D1):.0%})",
            "method_d2": f"{N_POS_SNIPPETS_D2}-in-{N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2} description snippet match (chance={N_POS_SNIPPETS_D2/(N_POS_SNIPPETS_D2+N_NEG_SNIPPETS_D2):.0%})",
            "aggregation": "per-group run-level means, then mean ± stderr across runs",
            "neg_sources": "medium=Ungrouped/unassigned features (excluding Emb/Output)",
            "conditions": {
                "random": "ours-full groups, features shuffled across them",
                "human": "manual_groups.json — human/Neuronpedia-derived labels",
                "ours-full": "auto pipeline output after phase-3 reconciliation",
                "ours-no-reconciliation": "auto pipeline output before phase-3 (snapshot)",
            },
        },
        "conditions": condition_results,
        "description_accuracy": d1_result,
        "description_snippet_accuracy": d2_result,
    }

    with open(VALIDATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Validation report saved → %s", VALIDATION_REPORT_FILE)

    write_markdown_report(report, VALIDATION_MARKDOWN_FILE)

    history: list[dict] = []
    if VALIDATION_HISTORY_FILE.exists():
        try:
            with open(VALIDATION_HISTORY_FILE) as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    history.append(report)
    with open(VALIDATION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Appended to history (%d total) → %s", len(history), VALIDATION_HISTORY_FILE)

    # Console summary
    chance_m1 = 1 / (1 + N_NEG_FEATURES_M1)
    chance_m2 = N_POS_SNIPPETS_M2 / (N_POS_SNIPPETS_M2 + N_NEG_SNIPPETS_M2)
    chance_d1 = 1 / (1 + N_NEG_DESCS_D1)
    chance_d2 = N_POS_SNIPPETS_D2 / (N_POS_SNIPPETS_D2 + N_NEG_SNIPPETS_D2)

    print(f"\n{'='*84}")
    print(f"  VALIDATION SUMMARY  ({N_RUNS} runs · min_group_size={MIN_GROUP_SIZE} · difficulty={DIFFICULTY})")
    print(f"  M1 chance={chance_m1:.0%}  M2 chance={chance_m2:.0%}  "
          f"D1 chance={chance_d1:.0%}  D2 chance={chance_d2:.0%}")
    print(f"{'='*84}")

    d1m = report["description_accuracy"].get("macro_avg", {})
    d2m = report["description_snippet_accuracy"].get("macro_avg", {})
    print(f"  D1 (description accuracy)  : {_fmt_macro(d1m)}  ({report['description_accuracy'].get('total_trials', 0)} trials)")
    print(f"  D2 (description snippets)  : {_fmt_macro(d2m)}  ({report['description_snippet_accuracy'].get('total_tasks', 0)} tasks)")
    print()

    sources = ("easy", "medium") if DIFFICULTY == "both" else ("medium",)
    header = (
        f"  {'Condition':<26}"
        + "".join(f"{'M1 ' + s:>16}" for s in sources)
        + "".join(f"{'M2 ' + s:>16}" for s in sources)
        + f"{'Coverage':>10}{'Groups':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in ("random", "human", "ours-no-reconciliation", "ours-full"):
        if cond not in condition_results:
            continue
        c = condition_results[cond]
        if c.get("skipped"):
            print(f"  {cond:<26}  skipped: {c.get('reason','')}")
            continue
        row = f"  {cond:<26}"
        for s in sources:
            m = _macro_for(report, cond, s, "method1")
            row += f"{m['mean_accuracy']*100:>8.1f}±{m['stderr_accuracy']*100:.1f}%  "
        for s in sources:
            m = _macro_for(report, cond, s, "method2")
            row += f"{m['mean_accuracy']*100:>8.1f}±{m['stderr_accuracy']*100:.1f}%  "
        cov = c.get("attribution_coverage")
        cov_str = f"{cov*100:.1f}%" if cov is not None else "  —  "
        n_valid = c.get("n_groups_valid", 0)
        n_total = c.get("n_groups_total", 0)
        row += f"{cov_str:>10}{f'{n_valid}/{n_total}':>10}"
        print(row)

    print(f"{'='*84}\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


