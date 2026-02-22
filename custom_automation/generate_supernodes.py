"""
Step 3 — Semantically group features into supernodes using OpenAI.

Two-phase approach:
  Phase 1: Discover ~8 groups from the top-K most influential features.
  Phase 2: Assign remaining features to existing groups in batches.

Reads:  artifacts/feature_descriptions.json
Writes: artifacts/feature_groups.json

Usage:
    OPENAI_API_KEY=sk-xxx python group_features.py
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    GROUPING_BATCH_SIZE,
    GROUPING_MODEL,
    GROUPING_TOP_K_SEED,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# OpenAI client — reads OPENAI_API_KEY from environment automatically.
# NEVER hardcode your API key in source code.
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    log.error("OPENAI_API_KEY not set. Run with:  OPENAI_API_KEY=sk-xxx python group_features.py")
    sys.exit(1)

client = OpenAI()

# ---------------------------------------------------------------------------
# Pydantic Schemas (for structured output)
# ---------------------------------------------------------------------------


class Assignment(BaseModel):
    feature_id: str = Field(description="The feature ID.")
    group_name: str = Field(description="The group name (or 'Ungrouped').")


class GroupDef(BaseModel):
    group_name: str = Field(description="The semantic name of the supernode group.")
    rationale: str = Field(description="Brief rationale for why these features are clustered.")


class Phase1Output(BaseModel):
    groups: list[GroupDef] = Field(description="The ~8 high-level groups discovered.")
    assignments: list[Assignment] = Field(description="List of feature_id to group_name assignments.")


class Phase2Output(BaseModel):
    assignments: list[Assignment] = Field(description="List of feature_id to group_name assignments.")
    new_groups: list[GroupDef] = Field(description="Any NEW groups created during this batch (if absolutely necessary).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_and_sort_features() -> tuple[list[dict], str]:
    """Load described features sorted by influence, and the original prompt text."""
    log.info("Loading descriptions from %s", FEATURE_DESCRIPTIONS_FILE)

    if not FEATURE_DESCRIPTIONS_FILE.exists():
        log.error("Missing %s — run add_description.py first.", FEATURE_DESCRIPTIONS_FILE)
        sys.exit(1)

    with open(FEATURE_DESCRIPTIONS_FILE, "r") as f:
        desc_data = json.load(f)

    features = [
        {
            "id": item.get("id"),
            "score": float(item.get("influence_score", 0.0)),
            "desc": item.get("generated_description", "No description"),
        }
        for item in desc_data
    ]

    # Highest influence first.
    features.sort(key=lambda x: x["score"], reverse=True)

    # Grab prompt context from graph.
    prompt_text = "Unknown Prompt"
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)
        input_tokens = graph.get("input_tokens", [])
        if input_tokens:
            prompt_text = "".join(str(t) for t in input_tokens).replace("\n", " ")

    return features, prompt_text


def format_feature_list(batch: list[dict]) -> str:
    return "\n".join(f"ID: {f['id']} | Desc: {f['desc']}" for f in batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    features, prompt_text = load_and_sort_features()
    if not features:
        log.error("No described features found.")
        return

    log.info("Total features to process: %d", len(features))

    active_groups: dict[str, str] = {}        # name → rationale
    final_assignments: dict[str, str] = {}    # feature_id → group_name

    # ==================================================================
    # PHASE 1: DISCOVERY (seed with top-K)
    # ==================================================================
    seed_features = features[:GROUPING_TOP_K_SEED]
    log.info("Phase 1: Discovering supernodes from top %d features …", GROUPING_TOP_K_SEED)

    phase1_prompt = f"""You are an expert AI interpretability researcher analyzing the internal representations of a large language model.
Context: The model was given the following prompt: "{prompt_text}"

Task: Below is a list of the {GROUPING_TOP_K_SEED} most influential features (and their descriptions) that activated during this prompt. Your goal is to cluster these features into meaningful semantic groups ("super nodes").

Constraints:
1. Target Size: Aim for roughly 8 high-level groups, though this is a flexible target.
2. Do Not Force Structure: Neural network features are often messy and polysemantic. If a feature does not clearly fit into a group, assign it to "Ungrouped". Do not force loose connections.
3. Hyper-Relevant Outliers: A single feature can constitute its own distinct group if it represents a highly specific and crucial concept related to the prompt.
4. Keep group name concise and usually in format of "Say a __" (e.g. "Say a proper noun")

Features to Cluster:
{format_feature_list(seed_features)}
"""

    response = client.beta.chat.completions.parse(
        model=GROUPING_MODEL,
        messages=[{"role": "user", "content": phase1_prompt}],
        response_format=Phase1Output,
    )

    p1 = response.choices[0].message.parsed

    for g in p1.groups:
        active_groups[g.group_name] = g.rationale
    for a in p1.assignments:
        final_assignments[a.feature_id] = a.group_name

    log.info("Established %d initial supernodes.", len(active_groups))

    # ==================================================================
    # PHASE 2: ASSIGNMENT (rolling batches)
    # ==================================================================
    remaining = features[GROUPING_TOP_K_SEED:]

    if remaining:
        log.info("Phase 2: Assigning remaining %d features …", len(remaining))

        for i in tqdm(range(0, len(remaining), GROUPING_BATCH_SIZE)):
            batch = remaining[i : i + GROUPING_BATCH_SIZE]
            groups_context = json.dumps(active_groups, indent=2)

            phase2_prompt = f"""You are an expert AI interpretability researcher analyzing the internal representations of a large language model.
Context: The model was given the prompt: "{prompt_text}"

Current State: We have already established the following feature groups and rationales:
{groups_context}

Task: Below is a new batch of feature descriptions. For each feature, evaluate whether it belongs in one of the existing groups.

Rules:
1. Assign the feature to an existing group if it strongly aligns with the group's rationale.
2. If a feature represents a strong, distinct concept not covered by existing groups, you may define a new group.
3. If a feature is noisy or does not clearly fit anywhere, assign it to "Ungrouped". Do not force structure.

New Batch:
{format_feature_list(batch)}
"""

            response = client.beta.chat.completions.parse(
                model=GROUPING_MODEL,
                messages=[{"role": "user", "content": phase2_prompt}],
                response_format=Phase2Output,
                temperature=0.1,
            )

            p2 = response.choices[0].message.parsed

            for a in p2.assignments:
                final_assignments[a.feature_id] = a.group_name

            for new_g in p2.new_groups:
                if new_g.group_name not in active_groups:
                    active_groups[new_g.group_name] = new_g.rationale
                    log.info("New group created mid-stream: %s", new_g.group_name)

    # ==================================================================
    # SAVE
    # ==================================================================
    with open(FEATURE_GROUPS_FILE, "w") as f:
        json.dump(final_assignments, f, indent=2)

    log.info(
        "Done — mapped %d features across %d supernodes → %s",
        len(final_assignments),
        len(active_groups),
        FEATURE_GROUPS_FILE,
    )


if __name__ == "__main__":
    main()