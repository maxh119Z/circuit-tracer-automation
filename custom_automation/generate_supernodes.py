"""
Step 3 — Semantically group features into supernodes using OpenAI.

Two-phase approach:
  Phase 1: Discover groups from the top-K most influential features.
  Phase 2: Assign remaining features to existing groups in concurrent batches.

Reads:  artifacts/feature_descriptions.json
Writes: artifacts/feature_groups.json

Usage:
    OPENAI_API_KEY=sk-xxx python group_features.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

from openai import AsyncOpenAI
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

GROUPING_TOP_K_SEED = 60

# ---------------------------------------------------------------------------
# Async OpenAI client
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    log.error("OPENAI_API_KEY not set.")
    sys.exit(1)

client = AsyncOpenAI()

# ---------------------------------------------------------------------------
# Pydantic schemas (structured output)
# ---------------------------------------------------------------------------


class Assignment(BaseModel):
    feature_id: str = Field(description="The feature ID.")
    group_name: str = Field(description="The group name (or 'Ungrouped').")


class GroupDef(BaseModel):
    group_name: str = Field(description="Semantic name of the supernode group.")
    rationale: str = Field(description="Brief rationale for this cluster.")


class Phase1Output(BaseModel):
    groups: list[GroupDef] = Field(description="High-level groups discovered.")
    assignments: list[Assignment] = Field(description="Feature-to-group assignments.")


class Phase2Output(BaseModel):
    assignments: list[Assignment] = Field(description="Feature-to-group assignments.")
    new_groups: list[GroupDef] = Field(description="Any NEW groups created (only if absolutely necessary).")


# ---------------------------------------------------------------------------
# Shared prompt preamble
# ---------------------------------------------------------------------------

GROUPING_PHILOSOPHY = """
Important principles:
- The goal is a **cohesive attribution graph** that highlights *intent and meaning*.
- Features encoding prepositions, articles, punctuation, conjunctions, or other
  purely grammatical / syntactic scaffolding (e.g. "of", "the", "is", ",") should
  mostly go to "Ungrouped". They rarely carry attribution-relevant signal.
- Do NOT force a fixed number of groups. Create as few or as many groups as the
  data genuinely supports. Fewer, cleaner groups are better than many noisy ones.
- If a feature is ambiguous, polysemantic, or low-signal, prefer "Ungrouped".
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_and_sort_features() -> tuple[list[dict], str]:
    """Load described features sorted by influence; return them plus the prompt text."""
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
    features.sort(key=lambda x: x["score"], reverse=True)

    prompt_text = "Unknown Prompt"
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)
        metadata = graph.get("metadata", {})
        prompt_text = metadata.get("prompt", "")
        if not prompt_text:
            input_tokens = metadata.get("prompt_tokens", [])
            if input_tokens:
                prompt_text = "".join(str(t) for t in input_tokens).replace("\n", " ")
        if not prompt_text:
            prompt_text = "Unknown Prompt"

    return features, prompt_text


def format_feature_list(batch: list[dict]) -> str:
    return "\n".join(f"ID: {f['id']} | Desc: {f['desc']}" for f in batch)


async def process_batch(
    batch: list[dict], 
    groups_context: str, 
    prompt_text: str, 
    semaphore: asyncio.Semaphore
) -> Phase2Output:
    """Assign a single batch of features to existing (or new) groups."""
    
    prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the prompt: {prompt_text}

Current groups and rationales:
{groups_context}

{GROUPING_PHILOSOPHY}

Task: For each feature below, assign it to an existing group if it strongly aligns,
assign it to "Ungrouped" if it is noisy or purely structural, or — only if truly
necessary — define a new group for a genuinely distinct concept.

Features:
{format_feature_list(batch)}
"""

    # The semaphore restricts how many of these blocks can run simultaneously
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
            temperature=1,
        )
        return response.choices[0].message.parsed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    features, prompt_text = load_and_sort_features()
    if not features:
        log.error("No described features found.")
        return

    log.info("Total features: %d", len(features))

    active_groups: dict[str, str] = {}      # name → rationale
    final_assignments: dict[str, str] = {}  # feature_id → group_name

    # ==================================================================
    # PHASE 1 — Discover groups from top-K seed features
    # ==================================================================
    seed_features = features[:GROUPING_TOP_K_SEED]
    log.info("Phase 1: Discovering groups from top %d features…", GROUPING_TOP_K_SEED)

    phase1_prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the following prompt: {prompt_text}

Below are the {GROUPING_TOP_K_SEED} most influential features that activated during this prompt.
Cluster them into meaningful semantic groups ("supernodes").

{GROUPING_PHILOSOPHY}

Additional guidance:
- A single feature can be its own group if it represents a highly specific, crucial
  concept tied to the prompt.
- Group naming convention (two tiers):
    • Conceptual / semantic (encodes a background concept, entity, or relationship):
      short descriptive noun phrase — e.g. "U.S. geography", "capital cities".
    • Output-driving (proximal predictor steering toward a specific token):
      prefix with "say" — e.g. "say Austin", "say a capital".
  Ask: is the feature representing a fact (conceptual) or pushing a token (output-driving)?
  When in doubt, prefer the conceptual label.

Features:
{format_feature_list(seed_features)}
"""

    response = await client.beta.chat.completions.parse(
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
    # PHASE 2 — Assign remaining features concurrently
    # ==================================================================
    remaining = features[GROUPING_TOP_K_SEED:]

    if remaining:
        log.info("Phase 2: Assigning remaining %d features…", len(remaining))
        groups_context = json.dumps(active_groups, indent=2)

        # Set maximum concurrent OpenAI requests to avoid 429 rate limits
        MAX_CONCURRENT_REQUESTS = 15 
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        tasks = [
            process_batch(
                remaining[i : i + GROUPING_BATCH_SIZE], 
                groups_context, 
                prompt_text, 
                semaphore
            )
            for i in range(0, len(remaining), GROUPING_BATCH_SIZE)
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            p2 = await coro
            for a in p2.assignments:
                final_assignments[a.feature_id] = a.group_name
            for g in p2.new_groups:
                if g.group_name not in active_groups:
                    active_groups[g.group_name] = g.rationale
                    log.info("New group created mid-stream: %s", g.group_name)

    # ==================================================================
    # AUTO-GROUP EMBEDDING & OUTPUT (LOGIT) NODES
    # ==================================================================
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)

        logit_nodes = []
        for node in graph.get("nodes", []):
            nid = str(node.get("node_id", ""))
            ftype = node.get("feature_type", "")

            if ftype == "embedding":
                final_assignments[nid] = "Embedding"
            elif ftype == "logit" or node.get("is_target_logit") is True:
                logit_nodes.append(node)

        # Isolate the single highest-probability logit node
        if logit_nodes:
            top_node, max_p = None, -1.0

            for n in logit_nodes:
                p = float(n.get("token_prob", 0.0))
                nid = str(n.get("node_id", ""))
                match = re.search(r"p=([0-9.]+)", nid)
                if match:
                    p = float(match.group(1))
                if p > max_p:
                    max_p, top_node = p, n

            if top_node:
                nid = str(top_node.get("node_id", ""))
                token_str = top_node.get("logitToken")
                if not token_str:
                    match = re.search(r'"([^"]+)"', nid)
                    token_str = match.group(1) if match else nid.replace("Output", "").split("(")[0].strip()
                final_assignments[nid] = f"Predicted Output '{token_str}'"

        log.info("Auto-grouped embedding and isolated top output node.")

    # ==================================================================
    # SAVE
    # ==================================================================
    with open(FEATURE_GROUPS_FILE, "w") as f:
        json.dump(final_assignments, f, indent=2)

    log.info(
        "Done — %d features across %d supernodes → %s",
        len(final_assignments),
        len(active_groups),
        FEATURE_GROUPS_FILE,
    )


if __name__ == "__main__":
    asyncio.run(main())