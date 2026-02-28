"""
Step 3 — Semantically group features into supernodes using OpenAI.

Two-phase approach:
  Phase 1: Discover groups from the top-K most influential features.
  Phase 2: Assign remaining features to existing groups in concurrent batches.

Post-processing:
  - Embedding nodes: grouped by semantic role of their token (skip function words)
  - Logit nodes: named by actual predicted token, top-p selection for diversity

Reads:  artifacts/feature_descriptions.json
Writes: artifacts/feature_groups.json

Usage:
    OPENAI_API_KEY=sk-xxx python generate_supernodes.py
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
    setup_logging,
)

log = setup_logging()

# Override config — use 60 seed features for better group discovery
GROUPING_TOP_K_SEED = 60

# Top-p threshold for logit nodes: include output tokens until cumulative prob >= this
LOGIT_TOP_P = 0.90
LOGIT_MAX_NODES = 4

# Tokens that are purely structural / function words — skip as embedding groups
FUNCTION_WORDS = frozenset({
    # Articles & determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Prepositions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    # Pronouns
    "it", "its", "he", "she", "they", "them", "his", "her", "we", "you",
    "i", "me", "my", "your", "our", "their",
    # Auxiliaries & copulas
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    # Common structural
    "not", "no", "if", "then", "than", "as", "which", "who", "whom",
    "what", "when", "where", "how", "that",
    # Punctuation-like tokens
    ",", ".", ":", ";", "!", "?", "'", '"', "(", ")", "-", "—", "",
    # BOS / special
    "<bos>", "<eos>", "<pad>", "<s>", "</s>",
    # Other common stopwords
    "containing",
})


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
        log.error("Missing %s — run generate_description.py first.", FEATURE_DESCRIPTIONS_FILE)
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

    # Extract prompt text from graph metadata (nested under "metadata", not top-level)
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
    semaphore: asyncio.Semaphore,
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
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
            temperature=1,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            log.warning("Phase 2 batch returned None — skipping batch.")
            return Phase2Output(assignments=[], new_groups=[])
        return parsed


# ---------------------------------------------------------------------------
# Post-processing: Embedding & Logit grouping
# ---------------------------------------------------------------------------

def group_embedding_nodes(graph_data: dict, final_assignments: dict[str, str]) -> None:
    """Group embedding nodes by their token, skipping function words.

    Meaningful content tokens each get their own group named:
        Emb: "<token>"
    Function words / punctuation are left Ungrouped (not pinned).
    """
    metadata = graph_data.get("metadata", {})
    prompt_tokens: list[str] = metadata.get("prompt_tokens", [])

    for node in graph_data.get("nodes", []):
        nid = str(node.get("node_id", ""))
        ftype = node.get("feature_type", "")

        if ftype != "embedding":
            continue

        # Figure out which token this embedding represents
        ctx_idx = node.get("ctx_idx", -1)
        if 0 <= ctx_idx < len(prompt_tokens):
            token = str(prompt_tokens[ctx_idx])
        else:
            token = "?"

        # Clean the token for comparison
        token_clean = token.strip().lower().replace("▁", "").replace("Ġ", "")

        if token_clean in FUNCTION_WORDS or len(token_clean) <= 1:
            # Skip — don't assign to any group (stays ungrouped / unpinned)
            final_assignments.pop(nid, None)
        else:
            # Give it a descriptive group name
            display_token = token.strip().replace("▁", "").replace("Ġ", "")
            final_assignments[nid] = f'Emb: "{display_token}"'

    log.info("Embedding grouping complete (function words filtered out).")


def group_logit_nodes(graph_data: dict, final_assignments: dict[str, str]) -> None:
    """Group logit (output) nodes using top-p selection.

    Collects logit nodes sorted by probability, includes tokens until
    cumulative probability >= LOGIT_TOP_P. Each gets a descriptive name.
    If one token dominates, only that one is shown.
    """
    logit_nodes: list[dict] = []
    for node in graph_data.get("nodes", []):
        ftype = node.get("feature_type", "")
        if ftype == "logit" or node.get("is_target_logit") is True:
            logit_nodes.append(node)

    if not logit_nodes:
        return

    # Sort by probability descending
    logit_nodes.sort(key=lambda n: float(n.get("token_prob", 0.0)), reverse=True)

    # Top-p selection
    cumulative_p = 0.0
    selected: list[tuple[dict, float]] = []
    for n in logit_nodes:
        p = float(n.get("token_prob", 0.0))
        selected.append((n, p))
        cumulative_p += p
        if cumulative_p >= LOGIT_TOP_P or len(selected) >= LOGIT_MAX_NODES:
            break

    # Also always include the target logit if it exists and wasn't already selected
    selected_ids = {str(n.get("node_id", "")) for n, _ in selected}
    for n in logit_nodes:
        if n.get("is_target_logit") and str(n.get("node_id", "")) not in selected_ids:
            p = float(n.get("token_prob", 0.0))
            selected.append((n, p))
            break

    log.info("Logit top-p selection: %d tokens (cumulative p=%.3f)", len(selected), cumulative_p)

    # Name each selected logit node
    for n, p in selected:
        nid = str(n.get("node_id", ""))

        # Extract the actual token text
        token_str: str = n.get("logitToken", "")
        if not token_str:
            # Try to parse from clerp: 'Output " Dallas" (p=0.026)'
            clerp: str = n.get("clerp", "")
            match = re.search(r'"([^"]+)"', clerp)
            if match:
                token_str = match.group(1)
            else:
                token_str = nid

        token_str = token_str.strip()
        pct = f"{p:.1%}"

        if n.get("is_target_logit"):
            final_assignments[nid] = f'Output: "{token_str}" ({pct}) [target]'
        else:
            final_assignments[nid] = f'Output: "{token_str}" ({pct})'

    # Remove any old logit assignments for nodes NOT selected
    final_selected_ids = {str(n.get("node_id", "")) for n, _ in selected}
    for n in logit_nodes:
        nid = str(n.get("node_id", ""))
        if nid not in final_selected_ids:
            final_assignments.pop(nid, None)

    log.info("Logit nodes named: %s",
             ", ".join(f'"{n.get("logitToken", "?").strip()}" ({p:.1%})' for n, p in selected))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    features, prompt_text = load_and_sort_features()
    if not features:
        log.error("No described features found.")
        return

    log.info("Total features: %d", len(features))
    log.info("Prompt: %s", prompt_text)

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

    if p1 is None:
        log.error("Phase 1 parsing returned None — check OpenAI response.")
        return

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

        MAX_CONCURRENT_REQUESTS = 67
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        tasks = [
            process_batch(
                remaining[i : i + GROUPING_BATCH_SIZE],
                groups_context,
                prompt_text,
                semaphore,
            )
            for i in range(0, len(remaining), GROUPING_BATCH_SIZE)
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            p2: Phase2Output = await coro
            for a in p2.assignments:
                final_assignments[a.feature_id] = a.group_name
            for g in p2.new_groups:
                if g.group_name not in active_groups:
                    active_groups[g.group_name] = g.rationale
                    log.info("New group created mid-stream: %s", g.group_name)

    # ==================================================================
    # POST-PROCESSING — Embedding & Logit nodes
    # ==================================================================
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            graph_data: dict = json.load(f)

        group_embedding_nodes(graph_data, final_assignments)
        group_logit_nodes(graph_data, final_assignments)

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