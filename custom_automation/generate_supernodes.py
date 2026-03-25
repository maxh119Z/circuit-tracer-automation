"""
Step 3 — Semantically group features into supernodes using OpenAI.

Three-phase approach:
  Phase 1: Discover groups from the top-30 most influential features.
  Phase 2: Assign remaining features to existing groups in concurrent batches.
  Phase 3: Reconciliation — merge duplicates, fix misassignments, split broad groups.

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
from typing import Any

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

# Phase 1 seed size
GROUPING_TOP_K_SEED = 100

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


class RenameAction(BaseModel):
    old_name: str = Field(description="The current group name.")
    new_name: str = Field(description="The new group name (must be <= 5 words).")

class MergeAction(BaseModel):
    groups_to_merge: list[str] = Field(description="List of group names to merge together.")
    merged_name: str = Field(description="The name for the merged group (must be <= 5 words).")



class SplitAction(BaseModel):
    group_to_split: str = Field(description="The group name to split.")
    new_subgroups: list[GroupDef] = Field(description="The new subgroups to create.")
    reassignments: list[Assignment] = Field(description="Feature reassignments into the new subgroups.")


class ReassignAction(BaseModel):
    feature_id: str = Field(description="The feature ID to reassign.")
    from_group: str = Field(description="Current group name.")
    to_group: str = Field(description="Target group name (can be 'Ungrouped' or a new name).")


class Phase3Output(BaseModel):
    renames: list[RenameAction] = Field(default_factory=list, description="Groups to rename.")
    merges: list[MergeAction] = Field(default_factory=list, description="Groups to merge together.")
    splits: list[SplitAction] = Field(default_factory=list, description="Groups to split into subgroups.")
    reassignments: list[ReassignAction] = Field(default_factory=list, description="Individual features to move between groups.")
    dropped_groups: list[str] = Field(default_factory=list, description="Groups to dissolve entirely (members become Ungrouped).")

# ---------------------------------------------------------------------------
# Shared prompt preamble
# ---------------------------------------------------------------------------

GROUPING_PHILOSOPHY = """
Important principles:
- The goal is a cohesive attribution graph that highlights the main intent and meaning of the prompt.
- Do not force a fixed number of groups. Create only groups that are clearly supported by the data.
- Prefer clear, human-readable semantic groups.
- If a feature is weak, isolated, or not meaningfully relevant to the main prompt circuitry, assign it to "Ungrouped".
- Features encoding prepositions, articles, punctuation, conjunctions, or other purely grammatical / syntactic scaffolding
  (e.g. "of", "the", "is", ",") should usually go to "Ungrouped" unless they clearly promote a meaningful token.

RELEVANCE TO THE MAIN PROMPT:
- A group should represent a concept that is genuinely relevant to the main semantic structure of the prompt.
- Do not create groups for one-time incidental associations, or weak thematic echoes.
- If a feature reflects a real concept but that concept is not central to the prompt or not supported by nearby related features, prefer "Ungrouped".
- A valid group should usually feel meaningfully connected to other groups in the graph, not like an isolated curiosity.

GROUP GRANULARITY:
- Do not group features only because they belong to the same broad topic.
- Preserve meaningful distinctions in abstraction level when those distinctions are relevant to the main prompt.
- Broad categories and their stable subtypes should usually be separate groups only when both are actually supported and relevant.
- If one feature represents a general domain and another represents a specific subtype within that domain, do not automatically merge them.
- Use the model's predicted output tokens to guide granularity decisions.
  Distinctions that are irrelevant to the actual output and attribution graph can be merged.
  Distinctions that explain WHY the model chose one output over another should be preserved or split further.

SEMANTIC ROLE — "SAY X" vs "X ITSELF" (CRITICAL):
- Before grouping features together, ask whether they serve the same semantic role.
- Features should usually be separated if they differ in role, framing, or function, even when they share the same topic.
- A feature that introduces, says, frames, or sets up a concept is different from a feature representing the concept itself.
  These MUST be in separate groups.
- For example, "say a ___ and ___" should usually be a different group from the concept "___ and ___" by itself.
- Likewise, "introduce a ___ and ___" should usually be a different group from the bare concept.
- Do not merge a linguistic frame with the semantic content being framed.
- Surface overlap is not enough reason to merge two groups.

HOW TO TELL "SAY" FROM "CONCEPT" FEATURES:
- Look at the feature description carefully. If the description mentions firing on function words,
  prepositions, articles, punctuation, or structural tokens adjacent to content — it is a "say" feature.
- If the description mentions firing on content words (nouns, verbs, proper names, domain terms)
  that directly embody a concept — it is a "concept" feature.
- Descriptions that start with "say", "introduce", or mention "framing" / "setting up" → "say" group.
- Descriptions tagged [SAY] or [CONCEPT] should be respected as classification signals.
- When descriptions are ambiguous, consider: would removing this feature change WHAT the model
  talks about (concept) or HOW it structures its output (say)?
- Never group a "say X" feature with an "X itself" feature. This is the single most important
  grouping rule.

Common role differences that justify separate groups include:
  - broad domain knowledge
  - more specific subdomain knowledge
  - named entities / specific referents
  - terminology or jargon
  - actions or behaviors
  - evaluative / comparative language
  - discourse framing or introducing language ("say" features)
  - output-driving or token-predictive features

SUBGROUP AWARENESS:
- If a group mixes multiple semantic roles or multiple abstraction levels, split it.
- It is better to have 2–3 precise groups than one vague bucket.
- Distinctions are often useful when they help explain the attribution graph.
- Do not prefer supergroups with too many features.
- Small groups are acceptable if they are interpretable, prompt-relevant, and helpful for explaining the graph.

NAMING:
- Group names should be <= 5 words.
- Group names should sound natural to a human reading a graph.
- Prefer short everyday phrasing over analytic or technical wording.
- If a group is about referring to, naming, or introducing something in text, prefer "say ..." style names. Think human nature language (layman's).
- Examples of preferred names:
  - "say a city"
  - "say a location"
  - "say a team"
  - "say a method"
- Prefer "say X" over "X (mention)", "X mention", or "mention X".
- Prefer "say a location" over "location mention", "location reference", or "mentioned location".
- Avoid parentheses unless absolutely necessary.
- Avoid labels with words like "mention", "reference", "entity", "concept", "topic", or "pattern" when a simpler natural phrase would work.
- If the member features are primarily described as framing, introducing, or slot-filling language, preserve that style in the group name.
- Avoid converting a natural framing-style cluster into a more abstract label unless the distinction is clearly unimportant.

PROPER NOUNS & ENTITIES:
- Distinguish generic entity-type features from features that track particular named entities if consistent and relevant.

UNGROUPED IS NOT BAD:
- "Ungrouped" is not a failure state.
- Use "Ungrouped" for features that are isolated and not clearly relevant to the main prompt, or not part of a cohesive cluster.
"""
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_sort_features() -> tuple[list[dict], str, str]:

    """Load described features sorted by influence; return them plus the prompt text."""
    log.info("Loading descriptions from %s", FEATURE_DESCRIPTIONS_FILE)

    if not FEATURE_DESCRIPTIONS_FILE.exists():
        log.error("Missing %s — run generate_description.py first.", FEATURE_DESCRIPTIONS_FILE)
        sys.exit(1)

    with open(FEATURE_DESCRIPTIONS_FILE, "r") as f:
        desc_data = json.load(f)

    features = []
    for item in desc_data:
        feat: dict[str, Any] = {
            "id": item.get("id"),
            "score": float(item.get("influence_score", 0.0)),
            "desc": item.get("generated_description", "No description"),
        }
        # Extract trigger tokens from top activations for grouping context
        triggers: list[str] = []
        for act in item.get("top_activations", [])[:5]:
            t = act.get("trigger", "").strip()
            if t and t not in triggers:
                triggers.append(t)
        if triggers:
            feat["triggers"] = triggers
        # Extract promoted tokens
        promotes = item.get("promotes", [])
        if promotes:
            feat["promotes"] = promotes[:5]
        features.append(feat)

    features.sort(key=lambda x: x["score"], reverse=False)

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

    output_tokens_str = ""
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)
        logit_nodes = [
            n for n in graph.get("nodes", [])
            if n.get("feature_type") == "logit" or n.get("is_target_logit")
        ]
        logit_nodes.sort(key=lambda n: float(n.get("token_prob", 0.0)), reverse=True)
        top_outputs = []
        for n in logit_nodes[:5]:
            clerp = n.get("clerp", "")
            match = re.search(r'"([^"]+)"', clerp)
            tok = match.group(1).strip() if match else "?"
            prob = float(n.get("token_prob", 0.0))
            top_outputs.append(f'"{tok}" ({prob:.1%})')
        if top_outputs:
            output_tokens_str = "Model's top predicted outputs: " + ", ".join(top_outputs)
            print(output_tokens_str)

    return features, prompt_text, output_tokens_str


def format_feature_list(batch: list[dict], include_evidence: bool = False) -> str:
    """Format features for grouping prompts.

    When *include_evidence* is True, append trigger tokens and promoted tokens
    so the grouping model can distinguish "say X" from "X itself".
    """
    lines: list[str] = []
    for f in batch:
        parts = [f"ID: {f['id']} | Desc: {f['desc']}"]
        if include_evidence:
            triggers = f.get("triggers")
            if triggers:
                parts.append(f"  Trigger tokens: {', '.join(triggers)}")
            promotes = f.get("promotes")
            if promotes:
                parts.append(f"  Promoted outputs: {', '.join(promotes)}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


async def process_batch(
    batch: list[dict],
    groups_context: str,
    prompt_text: str,
    output_context: str,
    semaphore: asyncio.Semaphore,
) -> Phase2Output:
    """Assign a single batch of features to existing (or new) groups."""
    prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the prompt: {prompt_text}

{output_context}

Current groups and rationales:
{groups_context}

{GROUPING_PHILOSOPHY}

Task: For each feature below, assign it to one of the existing groups if it strongly aligns.
Assign it to "Ungrouped" if it is noisy, polysemantic, or not clearly relevant to the main prompt semantics.
Only create a new group if the feature reflects a genuinely distinct semantic subgroup not covered by any existing group, is clearly relevant to the prompt, and would likely be shared by multiple related features.
Do not force fit a feature into an existing group when the semantic role or abstraction level does not match.

Features:
{format_feature_list(batch)}
"""
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            log.warning("Phase 2 batch returned None — skipping batch.")
            return Phase2Output(assignments=[], new_groups=[])
        return parsed


# ---------------------------------------------------------------------------
# Phase 3 — Reconciliation
# ---------------------------------------------------------------------------

def build_group_summary(
    final_assignments: dict[str, str],
    all_features: list[dict],
) -> str:
    """Build a summary of all groups with their member features + descriptions."""
    # Map feature id → description
    id_to_desc: dict[str, str] = {f["id"]: f["desc"] for f in all_features}

    # Build group → members
    group_members: dict[str, list[str]] = {}
    for fid, gname in final_assignments.items():
        if gname == "Ungrouped":
            continue
        group_members.setdefault(gname, []).append(fid)

    lines: list[str] = []
    for gname, members in sorted(group_members.items()):
        lines.append(f"\n## {gname} ({len(members)} members)")
        for fid in members[:15]:  # Cap at 15 per group to stay within context
            desc = id_to_desc.get(fid, "no description")
            lines.append(f"  - {fid}: {desc}")
        if len(members) > 15:
            lines.append(f"  ... and {len(members) - 15} more")

    # Also report ungrouped count
    ungrouped = sum(1 for g in final_assignments.values() if g == "Ungrouped")
    lines.append(f"\n## Ungrouped: {ungrouped} features")

    return "\n".join(lines)


def apply_phase3(
    phase3: Phase3Output,
    final_assignments: dict[str, str],
    active_groups: dict[str, str],
) -> None:
    """Apply Phase 3 reconciliation actions to the assignments in place."""

    # 1. Renames
    for rename in phase3.renames:
        old, new = rename.old_name, rename.new_name
        if old in active_groups:
            active_groups[new] = active_groups.pop(old)
        for fid in list(final_assignments):
            if final_assignments[fid] == old:
                final_assignments[fid] = new
        log.info("Renamed: '%s' → '%s'", old, new)

    # 2. Merges
    for merge in phase3.merges:
        for old_name in merge.groups_to_merge:
            for fid in list(final_assignments):
                if final_assignments[fid] == old_name:
                    final_assignments[fid] = merge.merged_name
            active_groups.pop(old_name, None)
        active_groups[merge.merged_name] = f"Merged from: {', '.join(merge.groups_to_merge)}"
        log.info("Merged: %s → '%s'", merge.groups_to_merge, merge.merged_name)

    # 3. Splits
    for split in phase3.splits:
        # Add new subgroups
        for sg in split.new_subgroups:
            active_groups[sg.group_name] = sg.rationale
        # Reassign features
        for a in split.reassignments:
            final_assignments[a.feature_id] = a.group_name
        # Remove the old group
        active_groups.pop(split.group_to_split, None)
        log.info("Split: '%s' → %s", split.group_to_split,
                 [sg.group_name for sg in split.new_subgroups])

    # 4. Individual reassignments
    for ra in phase3.reassignments:
        if ra.feature_id in final_assignments:
            final_assignments[ra.feature_id] = ra.to_group
            log.info("Reassigned: %s from '%s' → '%s'", ra.feature_id, ra.from_group, ra.to_group)

    # 5. Dropped groups
    for gname in phase3.dropped_groups:
        for fid in list(final_assignments):
            if final_assignments[fid] == gname:
                final_assignments[fid] = "Ungrouped"
        active_groups.pop(gname, None)
        log.info("Dropped group: '%s' (members → Ungrouped)", gname)


# ---------------------------------------------------------------------------
# Post-processing: Embedding & Logit grouping
# ---------------------------------------------------------------------------

def group_embedding_nodes(graph_data: dict, final_assignments: dict[str, str]) -> None:
    """Group embedding nodes by their token, skipping function words."""
    metadata = graph_data.get("metadata", {})
    prompt_tokens: list[str] = metadata.get("prompt_tokens", [])

    for node in graph_data.get("nodes", []):
        nid = str(node.get("node_id", ""))
        ftype = node.get("feature_type", "")

        if ftype != "embedding":
            continue

        ctx_idx = node.get("ctx_idx", -1)
        if 0 <= ctx_idx < len(prompt_tokens):
            token = str(prompt_tokens[ctx_idx])
        else:
            token = "?"

        token_clean = token.strip().lower().replace("▁", "").replace("Ġ", "")

        if token_clean in FUNCTION_WORDS or len(token_clean) <= 1:
            final_assignments.pop(nid, None)
        else:
            display_token = token.strip().replace("▁", "").replace("Ġ", "")
            final_assignments[nid] = f'Emb: "{display_token}"'

    log.info("Embedding grouping complete (function words filtered out).")


def group_logit_nodes(graph_data: dict, final_assignments: dict[str, str]) -> None:
    """Group logit (output) nodes using top-p selection, skipping function words."""
    logit_nodes: list[dict] = []
    for node in graph_data.get("nodes", []):
        ftype = node.get("feature_type", "")
        if ftype == "logit" or node.get("is_target_logit") is True:
            logit_nodes.append(node)

    if not logit_nodes:
        return

    logit_nodes.sort(key=lambda n: float(n.get("token_prob", 0.0)), reverse=True)

    cumulative_p = 0.0
    selected: list[tuple[dict, float]] = []
    for n in logit_nodes:
        p = float(n.get("token_prob", 0.0))

        token_str = n.get("logitToken", "")
        if not token_str:
            clerp = n.get("clerp", "")
            match = re.search(r'"([^"]+)"', clerp)
            token_str = match.group(1) if match else ""
        token_clean = token_str.strip().lower().replace("▁", "").replace("Ġ", "")
        if token_clean in FUNCTION_WORDS or len(token_clean) <= 1:
            continue

        selected.append((n, p))
        cumulative_p += p
        if cumulative_p >= LOGIT_TOP_P or len(selected) >= LOGIT_MAX_NODES:
            break

    # Always include target logit
    selected_ids = {str(n.get("node_id", "")) for n, _ in selected}
    for n in logit_nodes:
        if n.get("is_target_logit") and str(n.get("node_id", "")) not in selected_ids:
            p = float(n.get("token_prob", 0.0))
            selected.append((n, p))
            break

    log.info("Logit top-p selection: %d tokens (cumulative p=%.3f)", len(selected), cumulative_p)

    for n, p in selected:
        nid = str(n.get("node_id", ""))
        token_str_: str = n.get("logitToken", "")
        if not token_str_:
            clerp_: str = n.get("clerp", "")
            match = re.search(r'"([^"]+)"', clerp_)
            if match:
                token_str_ = match.group(1)
            else:
                token_str_ = nid

        token_str_ = token_str_.strip()
        pct = f"{p:.1%}"

        if n.get("is_target_logit"):
            final_assignments[nid] = f'Output: "{token_str_}" ({pct}) [target]'
        else:
            final_assignments[nid] = f'Output: "{token_str_}" ({pct})'

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
    features, prompt_text, output_context = load_and_sort_features()
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

{output_context}

Below are the {GROUPING_TOP_K_SEED} most influential features that activated during this prompt.
Cluster them into meaningful semantic groups ("supernodes").

{GROUPING_PHILOSOPHY}

Additional guidance:
- A single feature may form its own group only when it reflects a stable, reusable semantic pattern rather than a one-off surface detail.
- A feature or group completely unrelated or uncorrelated to the rest of the prompt context can be removed.
- Separate features when they differ in abstraction level or semantic role, even if they share a broad topic.
- Prefer labels that make the graph easier for a human to read, not just labels that are taxonomically tidy.
- A small but meaningful subgroup may be worth keeping if it tells a distinct local story in the graph.
- If several seed features already share a natural framing-style description such as "say a location" or "introduce a comparison", prefer preserving that style in the group name rather than collapsing it to a bare concept label.
- When a cluster reflects a discourse role, the group name should usually also reflect that discourse role.
- When in doubt between one broad group and two narrower groups, prefer the two narrower groups.
  Phase 3 will merge any that turn out to be redundant after seeing the full picture, therefore your role is to capture all relevant subgroups right now.
- Err on the side of more specific, more granular groups at this stage.

USING TRIGGER TOKENS AND PROMOTED OUTPUTS FOR GROUPING:
- Each feature includes its trigger tokens (the tokens that activate it) and promoted output tokens.
- Use trigger tokens to determine whether a feature is "say X" or "X itself":
  * If triggers are function words (prepositions, articles, punctuation) → likely a "say" feature.
  * If triggers are content words (nouns, names, domain terms) → likely a concept feature.
- NEVER put a "say" feature and a concept feature in the same group, even if they relate to the same topic.
- Use promoted outputs to confirm what concept a feature is driving toward.

Features:
{format_feature_list(seed_features, include_evidence=True)}
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
    # TEMPORARILY DISABLED — comment back in to re-enable Phase 2
    # remaining = features[GROUPING_TOP_K_SEED:]

    # if remaining:
    #     log.info("Phase 2: Assigning remaining %d features…", len(remaining))
    #     groups_context = json.dumps(active_groups, indent=2)

    #     MAX_CONCURRENT_REQUESTS = 67
    #     semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    #     tasks = [
    #         process_batch(
    #             remaining[i : i + GROUPING_BATCH_SIZE],
    #             groups_context,
    #             prompt_text,
    #             output_context,
    #             semaphore,
    #         )
    #         for i in range(0, len(remaining), GROUPING_BATCH_SIZE)
    #     ]
    #
    #     for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
    #         p2: Phase2Output = await coro
    #         for a in p2.assignments:
    #             final_assignments[a.feature_id] = a.group_name
    #         for g in p2.new_groups:
    #             if g.group_name not in active_groups:
    #                 active_groups[g.group_name] = g.rationale
    #                 log.info("New group created mid-stream: %s", g.group_name)

    # ==================================================================
    # PHASE 3 — Reconciliation
    # ==================================================================
    log.info("Phase 3: Reconciling groups…")

    group_summary = build_group_summary(final_assignments, features)
    num_groups = len({g for g in final_assignments.values() if g != "Ungrouped"})

    phase3_prompt = f"""You are an expert AI interpretability researcher reviewing the output of an automated feature grouping pipeline.

Context: The model was given the prompt: {prompt_text}

{output_context}

The pipeline produced {num_groups} groups from {len(final_assignments)} features. Your job is to
clean up the result — rename unclear groups, split groups that are too broad,
and reassign misplaced features.

{GROUPING_PHILOSOPHY}

Only make changes you are confident about. Do not restructure for the sake of it.

REVIEW CHECKLIST:
1. SAY vs CONCEPT MIXING: Does any group mix "say X" features (firing on function words / structural tokens that introduce a concept) with "X itself" features (firing on content words that ARE the concept)? → This is the highest-priority issue. Split them into separate groups.
2. OVERLY BROAD: Does any group mix features with clearly different semantic roles or abstraction levels? → Split it.
3. OVER-MERGED SUBTYPES: Does any group combine a broad category with a narrower stable subtype that is still useful and interpretable in this graph? → Keep separate or split it.
4. IRRELEVANT GROUPS: Drop only groups that are clearly isolated, or off-topic to any functionalty in the prompt; they do not add variational information or isn't related to the prompt.
5. LOCAL USEFULNESS: Do not remove a subgroup merely because it is small if it is graph-useful, interpretable, and connected to the main prompt circuitry.
6. MISASSIGNED: Are any features obviously in the wrong group? → Reassign.
7. NAMING: Are group names clear, natural, and <= 5 words? → Rename if needed.
8. DUPLICATE GROUPS: Are any two groups the same and without worthy and relevant nuances to stay separated? Merge them only if completely confident that detail is irrelevant to the attribution graph and final ouptut (use sparingly).
- Prefer preserving a small meaningful subgroup over collapsing it into a broader group, unless the subgroup is clearly noisy or redundant; information in attribution graph is usually good to know but not everything.
- A group does not need to be globally perfect; it should be locally interpretable and useful in the graph.
- Small groups are acceptable if they capture a distinct, prompt-relevant circuit.
- Do not merge away distinctions like subtype, role, or local sports category if they help explain the graph.
- A feature introducing or saying a concept is different from a feature representing that concept by itself.
- Rename technical or unnatural labels into plain natural language.
- Prefer "say X" over "X (mention)", "X mention", or "mention X".
- Clarification: Say should  be used in an "introduction" or "mentioning" context; if the features literally represent X, do not use "say." Do not use "say" for the sake of it.
- Avoid parentheses and analytic suffixes when a simpler human phrase works.

IMPORTANT:
- Only make changes you are confident about. Do not restructure for the sake of it.
- If the grouping looks good, return empty lists for all actions.

Current grouping:
{group_summary}
"""

    response3 = await client.beta.chat.completions.parse(
        model=GROUPING_MODEL,
        messages=[{"role": "user", "content": phase3_prompt}],
        response_format=Phase3Output,
    )
    p3 = response3.choices[0].message.parsed

    if p3 is None:
        log.warning("Phase 3 parsing returned None — skipping reconciliation.")
    else:
        total_actions = (
            len(p3.renames) + len(p3.merges) + len(p3.splits)
            + len(p3.reassignments) + len(p3.dropped_groups)
        )
        if total_actions == 0:
            log.info("Phase 3: No changes needed — grouping looks clean.")
        else:
            log.info("Phase 3: Applying %d actions…", total_actions)
            apply_phase3(p3, final_assignments, active_groups)

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

    final_group_count = len({g for g in final_assignments.values() if g != "Ungrouped"})
    log.info(
        "Done — %d features across %d groups → %s",
        len(final_assignments),
        final_group_count,
        FEATURE_GROUPS_FILE,
    )


if __name__ == "__main__":
    asyncio.run(main())