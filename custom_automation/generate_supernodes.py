"""
Step 3 — Semantically group features into supernodes using OpenAI.

Three-phase approach:
- Phase 1 creates groups
- Phase 2 only assigns to existing groups / Ungrouped
- Phase 3 can split / rename / recover missed structure

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

# Phase 1 seed size
GROUPING_TOP_K_SEED = 50

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
    assignments: list[Assignment] = Field(
        description="Feature-to-group assignments. Must use an existing group name or 'Ungrouped'."
    )

class RenameAction(BaseModel):
    old_name: str = Field(description="The current group name.")
    new_name: str = Field(description="The new group name (must be <= 5 words).")



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
    splits: list[SplitAction] = Field(default_factory=list, description="Groups to split into subgroups.")
    reassignments: list[ReassignAction] = Field(default_factory=list, description="Individual features to move between groups.")
    dropped_groups: list[str] = Field(default_factory=list, description="Groups to dissolve entirely (members become Ungrouped).")


# ---------------------------------------------------------------------------
# Shared prompt preamble
# ---------------------------------------------------------------------------

GROUPING_PHILOSOPHY = """
Important principles:
- The goal is a **cohesive attribution graph** that highlights *intent and meaning*.
- Features encoding prepositions, articles, punctuation, conjunctions, or other
  purely grammatical / syntactic scaffolding (e.g. "of", "the", "is", ",") should
  mostly go to "Ungrouped" unless it promotes a word. In most cases however, they rarely carry attribution-relevant signal.
- Do NOT force a fixed number of groups. Create as few or as many groups as the
  data genuinely supports. Fewer, cleaner groups are usually better than many noisy ones. 
- If a feature is ambiguous, polysemantic, or very low-signal (no specificity), prefer "Ungrouped".
- Group names may be generated with context of embedding prompt
- If you are planning to generate a group name with many disparate parts (sports + players), try to instead split the group up for specificity.
- Group names should be <= 5 words and still fully represent the features you think fit inside.

SUBGROUP AWARENESS:
- Before finalizing a group, consider whether it contains meaningful subgroups.
  A group like "education" might actually be two distinct circuits. Similarly, "literary works" might split
  into output driven versus conceptual ideas. A location could be a state versus a city.
- These cases call for SEPARATE groups even if they share a topic.
- It is better to have 2-3 precise subgroups than one vague supergroup, balanced with an overall minimal number of groups in total.
- Do not prefer supergroups with too many features.
- AVOID this AND that. If they are somewhat different, you can have a group "this" and a group "that"

PROPER NOUNS & ENTITIES:
- Groups involving specific named entities or concepts should include those names or be placed in a new subgroup.
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
    features.sort(key=lambda x: x["score"])

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
    """Assign a single batch of features to existing groups or Ungrouped."""

    prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the prompt: {prompt_text}

Current groups and rationales:
{groups_context}

{GROUPING_PHILOSOPHY}
Task: For each feature below, assign it to one of the EXISTING groups if it strongly aligns.
If it is noisy, purely structural, ambiguous, or does not clearly fit an existing group,
assign it to "Ungrouped". Do NOT force fit features into weak matches.

Features:
{format_feature_list(batch)}
"""
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            log.warning("Phase 2 batch returned None — skipping batch.")
            return Phase2Output(assignments=[])
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
        if ra.feature_id not in final_assignments:
            continue

        if ra.to_group != "Ungrouped" and ra.to_group not in active_groups:
            active_groups[ra.to_group] = f"Created during Phase 3 reassignment from '{ra.from_group}'."
            log.info("Created new Phase 3 group: '%s'", ra.to_group)

        final_assignments[ra.feature_id] = ra.to_group
        log.info(
            "Reassigned: %s from '%s' → '%s'",
            ra.feature_id,
            ra.from_group,
            ra.to_group,
        )

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
      short descriptive noun phrase — e.g. "U.S. geography", "Celebrity Names".
    • Output-driving (proximal predictor steering toward a specific token):
      prefix with "say" — e.g. "say a fact", "say a capital".
  Ask: is the feature representing a fact (conceptual) or pushing a token (output-driving)?
  When in doubt, prefer the conceptual label.

Features:
{format_feature_list(seed_features)}
"""

    try:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": phase1_prompt}],
            response_format=Phase1Output,
        )
        p1 = response.choices[0].message.parsed
    except Exception as exc:
        log.error("Phase 1 request failed: %s", exc)
        return


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
            try:
                p2: Phase2Output = await coro
            except Exception as exc:
                log.warning("Phase 2 batch failed: %s", exc)
                continue

            for a in p2.assignments:
                if a.group_name == "Ungrouped" or a.group_name in active_groups:
                    final_assignments[a.feature_id] = a.group_name
                else:
                    log.warning(
                        "Phase 2 returned unknown group '%s' for feature %s; coercing to Ungrouped",
                        a.group_name,
                        a.feature_id,
                    )
                    final_assignments[a.feature_id] = "Ungrouped"



    # ==================================================================
    # PHASE 3 — Reconciliation
    # ==================================================================
    log.info("Phase 3: Reconciling groups…")

    group_summary = build_group_summary(final_assignments, features)
    num_groups = len({g for g in final_assignments.values() if g != "Ungrouped"})

    phase3_prompt = f"""You are an expert AI interpretability researcher reviewing the output of an automated feature grouping pipeline.

Context: The model was given the prompt: {prompt_text}

The pipeline produced {num_groups} groups from {len(final_assignments)} features. Your job is to
clean up the result — rename unclear groups, split groups that are too broad,
and reassign misplaced features.

{GROUPING_PHILOSOPHY}

Only make changes you are confident about. Do not restructure for the sake of it.

REVIEW CHECKLIST:
1. OVERLY BROAD: Does any group mix features that serve clearly different computational roles
   (e.g. conceptual knowledge vs output-driving)? → Split it. 
3. MISASSIGNED: Are any features obviously in the wrong group based on their description? → Reassign.
4. NAMING: Are group names clear, specific, and <= 5 words? → Rename if not.
5. MISSING GROUPS: Is there a coherent cluster of features currently in "Ungrouped" that deserves
   its own group? If so, reassign those features to a new group name or existing group.

IMPORTANT:
- Only make changes you are confident about. Do not restructure for the sake of it.
- If the grouping looks good, return empty lists for all actions.

Current grouping:
{group_summary}
"""

    try:
        response3 = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": phase3_prompt}],
            response_format=Phase3Output,
        )
        p3 = response3.choices[0].message.parsed
    except Exception as exc:
        log.warning("Phase 3 request failed: %s", exc)
        p3 = None


    if p3 is None:
        log.warning("Phase 3 parsing returned None — skipping reconciliation.")
    else:
        total_actions = (
            len(p3.renames) + len(p3.splits)
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