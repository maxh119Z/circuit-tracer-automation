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


from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    GROUPING_BATCH_SIZE,
    GROUPING_MODEL,
    GROUPING_VARIANT,
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
    group_name: str = Field(description="The group name (or 'Ungrouped'). MUST be 5 words or fewer.")


class GroupDef(BaseModel):
    group_name: str = Field(description="Semantic name of the supernode group. MUST be 5 words or fewer. Simple, natural phrasing only.")
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

# ---------------------------------------------------------------------------
# Grouping prompt variants (a0–a3)
# ---------------------------------------------------------------------------

# a0 — Original: balanced semantic-role-aware grouping (SAY vs CONCEPT focus)
GROUPING_PHILOSOPHY_A0 = """
GOAL: Produce a cohesive attribution graph that highlights the main intent and meaning of the prompt.

RELEVANCE & UNGROUPED:
- Assign to "Ungrouped": weak, isolated, or noisy features; features not meaningfully connected to the main prompt semantics.
- Purely grammatical tokens (prepositions, articles, conjunctions, punctuation, copulas) go to "Ungrouped" unless they clearly promote a semantically meaningful content token.
- "say X" groups are ONLY valid when X is a meaningful content word or category (e.g., "say a city", "say a state"). Do NOT create "say X" groups when X is a function word, grammatical structure, or syntactic role — e.g., "say 'is'", "say a relative clause", "say 'of'", "say a preposition" are never valid groups. These belong in Ungrouped.
- A valid group should feel connected to at least one other group in the graph, not like an isolated curiosity.
- "Ungrouped" is not a failure — use it freely.

GRANULARITY & SPECIFICITY:
- Create only groups clearly supported by the data. Prefer the most specific name the evidence supports over broad buckets.
- Preserve meaningful distinctions in abstraction level when relevant to the prompt — don't merge a broad category with a narrower stable subtype.
- Use the "Promotes" field as a tiebreaker: features that promote clearly different tokens should be split; features promoting the same or related tokens can stay together.
- Distinctions that explain WHY the model chose one output over another should be preserved.

SEMANTIC ROLE — "SAY X" vs "X ITSELF" (highest-priority rule):
- A feature that introduces or frames a concept is different from a feature that IS the concept. Keep them in separate groups.
  - Highlighted tokens are function/structural words → the feature sets up what follows; name it "say [what]".
  - Highlighted tokens are content words → the feature represents the concept directly; name the concept.
- [SAY] / [CONCEPT] tags in descriptions are strong signals — respect them.
- "Say" is for genuine framing or introduction. Do not use it when features directly represent a concept.
- Surface overlap is never enough reason to merge groups with different semantic roles.

NAMING (STRICT — violating this is an error):
- HARD LIMIT: 5 words maximum. No exceptions.
- Natural, simple phrasing. If you need more than 5 words, the name is too specific — broaden it.
- Prefer "say a city" over "city mention", "city reference", or "mentioned city".
- Avoid parentheses and words like "mention", "reference", "entity", "concept", "topic", or "pattern" when a simpler phrase works.
- Include a specific named entity in the group name when that entity is the clear shared referent — don't collapse "Oakland" into "a city" if the features are specifically about Oakland.
- Prefer layman's vocabulary.

SPLITTING vs MERGING:
- Split when a group mixes semantic roles or abstraction levels.
- Prefer two narrow groups over one vague bucket — Phase 3 can merge, but cannot recover lost distinctions.
- Small groups are fine if they are interpretable and prompt-relevant.
"""

# All variants use a0 as the base philosophy.
# Per-phase extras are injected only into the specific phase each variant targets.
GROUPING_PHILOSOPHY = GROUPING_PHILOSOPHY_A0

# ---------------------------------------------------------------------------
# Per-phase variant extras (a1 tweaks Phase 1, a2 tweaks Phase 2, a3 tweaks Phase 3)
# ---------------------------------------------------------------------------

# a1 — Phase 1 extra: push for completeness and specificity in group discovery
PHASE1_EXTRAS: dict[str, str] = {
    "a0": "",
    "a1": """
PHASE 1 COMPLETENESS (variant a1):
Your primary goal is to surface EVERY meaningful semantic cluster, not just the most obvious ones.
Phase 2 can only assign features to groups that already exist here — missing a group in Phase 1 means it is lost.
- Actively look for groups you would expect to exist based on the prompt content and output, even if only 2–3 seed features represent them now.
- Prefer two narrow groups over one broad bucket when the evidence supports a real distinction. Phase 3 can merge, but cannot recover a distinction that was never captured.
- Be specific: a precise group name helps Phase 2 assign accurately. Vague bucket names produce wrong assignments.
- Before finalising, ask: is there any concept clearly present in the prompt or output that has no group yet? If seed features support it, create it.
""",
    "a2": "",
    "a3": "",
}

# a2 — Phase 2 extra: prioritise assignment accuracy over coverage
PHASE2_EXTRAS: dict[str, str] = {
    "a0": "",
    "a1": "",
    "a2": """
PHASE 2 ASSIGNMENT ACCURACY (variant a2):
Prioritise accuracy over coverage. A wrong assignment harms the graph more than leaving a feature Ungrouped.
- Only assign a feature to a group if the semantic match is clear and strong — not just the closest available option.
- When a feature fits two groups with equal confidence, choose Ungrouped over an arbitrary assignment.
- Only create a new group if the feature is clearly relevant to the prompt and you are confident 2+ additional features from the broader set would belong there.
""",
    "a3": "",
}

# a3 — Phase 3 extra: be decisive about compression and output-relevance
PHASE3_EXTRAS: dict[str, str] = {
    "a0": "",
    "a1": "",
    "a2": "",
    "a3": """
PHASE 3 COMPRESSION (variant a3):
Be decisive. The output-relevance principle above is not just a tiebreaker — apply it to every group.
- Default to merging when two groups play the same role relative to the output. Only preserve a distinction if you can articulate how it changes a reader's understanding of WHY the model predicted this specific output.
- Default to dropping when a group's connection to the output is indirect or unclear. Context that was present in the prompt but did not drive the prediction belongs in Ungrouped.
- A graph with 5 clear, causally relevant groups is strictly better than one with 10 groups where 4 are marginal.
- Drop any remaining "say X" groups where X is a function word, grammatical structure, or syntactic role (e.g., "say 'is'", "say a relative clause", "say 'of'"). These are never useful graph nodes.
""",
}

if GROUPING_VARIANT not in PHASE1_EXTRAS:
    log.warning("Unknown GROUPING_VARIANT '%s' — extras will be empty (a0 behaviour).", GROUPING_VARIANT)

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
        feat: dict = {
            "id": item.get("id"),
            "score": float(item.get("influence_score", 0.0)),
            "desc": item.get("generated_description", "No description"),
        }
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


def format_feature_list(batch: list[dict]) -> str:
    lines = []
    for f in batch:
        line = f"ID: {f['id']} | Desc: {f['desc']}"
        promotes = f.get("promotes")
        if promotes:
            line += f" | Promotes: {', '.join(promotes)}"
        lines.append(line)
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
{PHASE2_EXTRAS.get(GROUPING_VARIANT, "")}
Features:
{format_feature_list(batch)}
"""
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
            max_completion_tokens=4096,
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

Additional guidance for this phase:
- When in doubt between one broad group and two narrower groups, prefer the narrower ones — Phase 3 can merge, but cannot recover distinctions that were never captured.
- A single feature may form its own group only if it reflects a stable, reusable semantic pattern, not a one-off surface detail.
- Prefer names that make the graph easy to read over taxonomically tidy labels.
{PHASE1_EXTRAS.get(GROUPING_VARIANT, "")}
Features:
{format_feature_list(seed_features)}
"""

    response = await client.beta.chat.completions.parse(
        model=GROUPING_MODEL,
        messages=[{"role": "user", "content": phase1_prompt}],
        response_format=Phase1Output,
        max_completion_tokens=32768,
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

The pipeline produced {num_groups} groups from {len(final_assignments)} features. Your job is to clean up the result — rename unclear groups, split groups that are too broad, and reassign misplaced features.

{GROUPING_PHILOSOPHY}

OUTPUT-RELEVANCE PRINCIPLE:
Before any merge or drop decision, ask: does this group help explain WHY the model predicted the specific output above?
- A group is worth keeping if it plays a distinct role in the reasoning path to the output.
- A distinction between two groups is worth keeping only if the two groups relate to the output differently — e.g., one frames the output class while the other is the concept itself, or one is directly causal while the other is supporting context.
- If two groups are both relevant but play the same role relative to the output, merge them.
- If a group is real but its existence does not help a reader understand why the model chose this output, drop it.

REVIEW CHECKLIST (work through in order):
1. SAY vs CONCEPT MIXING: Does any group mix "say X" features with "X itself" features? → Highest-priority issue. Split them.
2. OVERLY BROAD: Does any group mix features with clearly different semantic roles or abstraction levels? → Split it.
3. OVER-MERGED SUBTYPES: Does any group combine a broad category with a narrower stable subtype? → Keep separate or split.
4. POLYSEMY / OFF-SENSE MEMBERS: Does any group contain features that activate on a different sense of the group's named concept than what this prompt requires? Features that belong to an irrelevant sense of the word should be reassigned to Ungrouped — they are real activations but not part of this prompt's reasoning. If the off-sense features are numerous and coherent, split them into their own group only if that group would itself pass the OUTPUT-RELEVANCE PRINCIPLE; otherwise Ungrouped.
5. IRRELEVANT GROUPS: Apply the OUTPUT-RELEVANCE PRINCIPLE. Drop groups that cannot be connected to the model's output prediction. Real distinctions that are irrelevant to this specific output should be collapsed or dropped.
6. SAME-ROLE MERGE: Two groups both pass relevance but play identical roles relative to the output? → Merge them.
7. MISASSIGNED: Are any features obviously in the wrong group? → Reassign.
8. NAMING: Are group names clear, natural, and ≤5 words? → Rename if needed.
9. DUPLICATES: Are any two groups identical in meaning with no useful distinction? → Merge (use sparingly).

Only make changes you are confident about. If the grouping looks good, return empty lists for all actions.
{PHASE3_EXTRAS.get(GROUPING_VARIANT, "")}
Current grouping:
{group_summary}
"""

    response3 = await client.beta.chat.completions.parse(
        model=GROUPING_MODEL,
        messages=[{"role": "user", "content": phase3_prompt}],
        response_format=Phase3Output,
        max_completion_tokens=8192,
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