"""
Step 3 — Semantically group features into supernodes using OpenAI.

Three-phase approach:
  Phase 1: Discover groups from the top-50 most influential features.
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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    DESCRIPTION_VARIANT,
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    GROUPING_BATCH_SIZE,
    GROUPING_MODEL,
    GROUPING_VARIANT,
    PACKAGE_DIR,
    setup_logging,
)

GROUPING_LOG_FILE = PACKAGE_DIR / "artifacts" / "grouping_log.md"

log = setup_logging()

#Phase 1 seeds from features[:GROUPING_TOP_K_SEED];
# Phase 2 assigns features[GROUPING_TOP_K_SEED:] (the remaining lower-influence ones).
GROUPING_TOP_K_SEED = 50

# Top-p threshold for logit nodes: include output tokens until cumulative prob >= this.
# LOGIT_MAX_NODES caps the total even if top-p isn't reached.
LOGIT_TOP_P = 0.90
LOGIT_MAX_NODES = 3

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




class ReassignAction(BaseModel):
    feature_id: str = Field(description="The feature ID to reassign.")
    from_group: str = Field(description="Current group name.")
    to_group: str = Field(description="Target group name (can be 'Ungrouped' or a new name).")


class Phase3Output(BaseModel):
    renames: list[RenameAction] = Field(default_factory=list, description="Groups to rename.")
    merges: list[MergeAction] = Field(default_factory=list, description="Groups to merge together.")
    reassignments: list[ReassignAction] = Field(default_factory=list, description="Individual features to move between groups.")
    dropped_groups: list[str] = Field(default_factory=list, description="Groups to dissolve (members become Ungrouped). Use for: (1) Grammar kill — names describing syntactic roles, token patterns, prefix fragments, or suppression. (2) Relevance drop — concepts with no connection to the prompt's reasoning chain or predicted output.")

# ---------------------------------------------------------------------------
# Grouping prompt variants (a0–a3)
#
# All variants share the same say-X / X-itself hard constraint.
# a1–a3 each combine the same three properties at increasing smartness:
#   - borderline pull (assign plausible features rather than leaving Ungrouped)
#   - critical merge constraint (same-role groups that tell the same story → merge)
#   - description reading (scan member descriptions for specific named entities
#     before naming groups generically; never use a broad category when
#     descriptions name a specific entity)
#
#   a0 — neutral baseline
#   a1 — structural rules: borderline pull + critical merge constraint
#   a2 — a1 + description-aware: read descriptions for named entities before naming
#   a3 — a2 + first-principles: reader test on every group decision
# ---------------------------------------------------------------------------

# Single shared strictness applied to all variants.
_SAY_X_STRICTNESS = (
    "Treat say-X / X-itself separation as a hard constraint. Every say-X group must contain only framing features; "
    "every concept group must contain only content features. A single misplaced feature is enough to split or reassign."
)

_SPECIFICITY_BIAS_BASE = (
    "When in doubt between narrower and broader groups, use whichever granularity best explains the model's specific output and prompt. "
    "Consider both the prompt and the predicted output together: a distinction is worth keeping only if it is relevant to what was asked and what the model predicted. "
)

_STRUCTURAL_RULES = (
    "BORDERLINE FEATURES: prefer assigning a borderline feature to a plausible existing group over Ungrouped — "
    "reserve Ungrouped for features with no meaningful connection to the prompt or output. "
    "MERGE CONSTRAINT: when two same-role groups tell the same part of the story and their separation does not help a reader understand the reasoning differently, merge them. "
    "Never merge a say-X group with a concept group — framing and content always stay separate. Prefer keeping more proper noun specificity when relevant to the prompt and output"
)

_DESCRIPTION_READING = (
    "DESCRIPTION-AWARE NAMING: before naming any group, scan the member descriptions for recurring proper nouns or specific named entities. "
    "If a specific entity (a place, person, concept) appears consistently across descriptions, use that specific name — "
    "do not collapse to a generic category like 'a place' or 'a city' when the descriptions clearly name something specific. "
    "Apply the same rule to say-X groups: if descriptions consistently name a specific entity after the trigger, prefer 'say California' over 'say a place'. "
    "If features clearly relate to an alternate sense of a key prompt word, name the group with a sense qualifier (e.g. 'X (general)') rather than a domain label that is not applicable given the prompt and output context (e.g. 'economic X' or 'X (music)'). "
    "The last few words of a description often carry non-trivial specificity — use them as an additional signal when placing features into groups. "
    "Group names must come from member descriptions, not from the prompt and output context. Be faithful to feature descriptions in naming. "

)

_SPECIFICITY_BIAS: dict[str, str] = {
    "a0": _SPECIFICITY_BIAS_BASE + _SAY_X_STRICTNESS,
    "a1": _SPECIFICITY_BIAS_BASE + _STRUCTURAL_RULES + _SAY_X_STRICTNESS,
    "a2": _SPECIFICITY_BIAS_BASE + _STRUCTURAL_RULES + _DESCRIPTION_READING + _SAY_X_STRICTNESS,
    "a3": (
        _SPECIFICITY_BIAS_BASE
        + _STRUCTURAL_RULES
        + _DESCRIPTION_READING
        + "READER TEST: before finalising any group decision, ask — if a reader saw only this group name and its members, "
        "would they understand something specific about why the model predicted this output? "
        "A group that passes is worth keeping. A group that fails should be merged or dropped. "
        "For merges specifically: two groups that tell the same part of the reasoning story should become one; "
        "two groups that explain different steps or angles should stay separate even if semantically close. "
        + _SAY_X_STRICTNESS
    ),
}

_bias_phase1: str = _SPECIFICITY_BIAS.get(GROUPING_VARIANT, _SPECIFICITY_BIAS["a0"])
_bias_phase2: str = _SPECIFICITY_BIAS.get(GROUPING_VARIANT, _SPECIFICITY_BIAS["a0"])
_bias_phase3: str = _SPECIFICITY_BIAS.get(GROUPING_VARIANT, _SPECIFICITY_BIAS["a0"])
if GROUPING_VARIANT not in _SPECIFICITY_BIAS:
    log.warning("Unknown GROUPING_VARIANT '%s' — falling back to a0 behaviour.", GROUPING_VARIANT)

# Shared rules for all phases — no specificity bias here; injected per-phase below.
GROUPING_PHILOSOPHY = """
GOAL: Produce a cohesive attribution graph that highlights the main intent and meaning of the prompt.

RELEVANCE & UNGROUPED:
- Assign to "Ungrouped": weak, isolated, or noisy features; features not meaningfully connected to the main prompt and output semantics.
- Purely grammatical tokens (prepositions, articles, conjunctions, punctuation, copulas) go to "Ungrouped" unless they clearly promote a semantically meaningful role (this is very rare). Sentence structure-level grammar is mostly pointless, unless the prompt or output is about it.
- "say X" groups are ONLY valid when X is a meaningful content word or category (e.g., "say a city", "say a state"). Do NOT create "say X" groups when X is a function word, grammatical structure, or syntactic role — e.g., "say 'is'", "say a relative clause", "say 'of'", "say a preposition" are never valid groups. These belong in Ungrouped.
- A valid group should feel connected to at least one other group in the graph, not like an isolated curiosity.
- "Ungrouped" is not a failure.

GRANULARITY & SPECIFICITY:
- Create only groups clearly supported by the data. Prefer the most specific name the evidence supports over broad buckets.
- Preserve meaningful distinctions in abstraction level when relevant to the prompt — don't blindly merge a broad category with a narrower stable subtype.
- Distinctions that explain WHY the model chose one output over another should be preserved.
- The prompt commits to one active sense of every word in it. Features from alternate senses of a word (senses the prompt does not require) may form their own groups — Phase 3 will consolidate them.

SEMANTIC ROLE — "SAY X" vs "X ITSELF" (high-priority rule):
- A feature that introduces or frames a concept is different from a feature that IS the concept. Keep them in separate groups.
  - Highlighted tokens are function/structural words → the feature sets up what follows; name it "say [what]".
  - Highlighted tokens are content words → the feature represents the concept directly; name the concept.
- say tags in descriptions are strong signals — respect them.
- "Say" is for genuine framing or introduction. Do not use it when features directly represent a concept.
- Only use 'say' in a group name when member descriptions themselves use it. Do not add 'say' by inference.
- Surface overlap is never enough reason to merge groups with different semantic roles.

NAMING (STRICT):
- LIMIT: 5 words maximum. No exceptions.
- Natural, simple phrasing. If you need more than 5 words, the name is too specific.
- Before naming a group, read the member descriptions. If a specific named entity (place, person, concept) recurs across them, use that name — do not default to a generic category when the descriptions clearly point to something specific.
- This applies to say-X groups too: "say California" is better than "say a place" when descriptions consistently name California.
- Avoid parentheses and words like "mention", "reference", "entity", "concept", "topic", or "pattern" when a simpler phrase works.
- Prefer layman's vocabulary.

SPLITTING vs MERGING:
- Split when a group mixes semantic roles or abstraction levels.
- Small groups are fine if they are interpretable and prompt-relevant.
"""

# ---------------------------------------------------------------------------
# Grouping log
# ---------------------------------------------------------------------------

def append_grouping_log(
    prompt_text: str,
    phase1_groups: list[str],
    phase3: "Phase3Output | None",
    final_assignments: dict[str, str],
) -> None:
    """Append a human-readable entry to the shared grouping log."""
    GROUPING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_groups = sorted({g for g in final_assignments.values() if g != "Ungrouped"})
    ungrouped_count = sum(1 for g in final_assignments.values() if g == "Ungrouped")

    lines: list[str] = [
        "---",
        f"## {prompt_text} — {DESCRIPTION_VARIANT} — {GROUPING_VARIANT}",
        "",
        f"### Phase 1 ({len(phase1_groups)} groups)",
        ", ".join(sorted(phase1_groups)) or "—",
        "",
        "### Phase 3 changes",
    ]

    if phase3 is None:
        lines.append("Skipped (API error)")
    else:
        total = len(phase3.renames) + len(phase3.merges) + len(phase3.reassignments) + len(phase3.dropped_groups)
        if total == 0:
            lines.append("No changes")
        else:
            if phase3.renames:
                lines.append("Renames (%d): %s" % (len(phase3.renames), ", ".join(f'"{r.old_name}" → "{r.new_name}"' for r in phase3.renames)))
            if phase3.merges:
                lines.append("Merges  (%d): %s" % (len(phase3.merges), ", ".join(f'[{", ".join(m.groups_to_merge)}] → "{m.merged_name}"' for m in phase3.merges)))
            if phase3.dropped_groups:
                lines.append("Dropped (%d): %s" % (len(phase3.dropped_groups), ", ".join(f'"{g}"' for g in phase3.dropped_groups)))
            if phase3.reassignments:
                lines.append("Reassigned: %d features" % len(phase3.reassignments))

    lines += [
        "",
        f"### Final ({len(final_groups)} groups, {ungrouped_count} ungrouped)",
        ", ".join(final_groups) or "—",
        "",
    ]

    with open(GROUPING_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Grouping log appended → %s", GROUPING_LOG_FILE)


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
        features.append(feat)

    features.sort(key=lambda x: x["score"], reverse=False)

    # Extract prompt text and output tokens from graph metadata (single read).
    prompt_text = "Unknown Prompt"
    output_tokens_str = ""
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
    return "\n".join(f"ID: {f['id']} | Desc: {f['desc']}" for f in batch)


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

Task: Assign each feature below to the best matching existing group.
These are lower-influence features — they rarely introduce meaningfully new semantic concepts beyond what Phase 1 already captured. Default to an existing group or "Ungrouped".
Do not force a match: if no group fits clearly, "Ungrouped" is correct.
Only create a new group if the concept is genuinely absent from the existing groups, clearly relevant to the prompt, and specific enough that multiple features would share it — this should be rare.
{_bias_phase2}
Features:
{format_feature_list(batch)}
"""
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
            max_completion_tokens=4096,
            reasoning_effort="low",
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

    # Report ungrouped features with descriptions so Phase 3 can rescue misassigned ones
    ungrouped_ids = [fid for fid, g in final_assignments.items() if g == "Ungrouped"]
    lines.append(f"\n## Ungrouped ({len(ungrouped_ids)} features)")
    for fid in ungrouped_ids[:20]:
        desc = id_to_desc.get(fid, "no description")
        lines.append(f"  - {fid}: {desc}")
    if len(ungrouped_ids) > 20:
        lines.append(f"  ... and {len(ungrouped_ids) - 20} more")

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

    # 3. Individual reassignments
    for ra in phase3.reassignments:
        if ra.feature_id in final_assignments:
            final_assignments[ra.feature_id] = ra.to_group
            log.info("Reassigned: %s from '%s' → '%s'", ra.feature_id, ra.from_group, ra.to_group)

    # 4. Dropped groups
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
- A single feature may form its own group only if it reflects a stable, reusable semantic pattern, not a one-off surface detail.
- Prefer names that make the graph easy to read over taxonomically tidy labels.
- Prefer two narrow groups over one vague bucket — Phase 3 later can merge, but cannot recover lost distinctions easily. When a concept is clearly relevant to the prompt or output, err toward creating a group rather than Ungrouped.
- HARD RULE — do NOT create groups for grammatical or structural patterns under any circumstances. Assign those features directly to Ungrouped. This includes: prepositions and locational connectors (say 'of', say 'in', say 'after', say locational preposition), copulas and predicate framing (say 'is', predicate framing, copula, say noun after copula), sentence-completion or next-token patterns (say completion, say next noun), subword or token-prefix fragments, heading markers, where-clause framing, structural/relational patterns derived from words in the prompt itself (containment verbs, prepositional structures, syntactic connectors), typographic or capitalization patterns (title case, capitalized tokens, proper noun formatting), and word-onset or prefix fragments — these describe token shape or suppression, not meaning. The test: does this group name a semantic concept, or does it describe a syntactic role or sentence structure? Concept = valid group. Sentence structure = Ungrouped.
- Do not create "suppress X", "demote X", "avoid X", or "anti-X" groups. Features that suppress or down-weight a particular output belong in the relevant concept group or Ungrouped.
- Do not create groups named for prompt format or input structure (e.g. "fact prompt", "fill-in-the-blank") — these describe the wrapper, not the reasoning content. Assign to Ungrouped.

SPECIFICITY GUIDANCE:
{_bias_phase1}

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

    phase1_group_names = list(active_groups.keys())
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
                output_context,
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
    # PHASE 3 — Reconciliation
    # ==================================================================
    log.info("Phase 3: Reconciling groups…")

    group_summary = build_group_summary(final_assignments, features)
    # Strip control characters that can corrupt the JSON payload
    group_summary = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', group_summary)
    num_groups = len({g for g in final_assignments.values() if g != "Ungrouped"})

    phase3_prompt = f"""You are an expert AI interpretability researcher reviewing the output of an automated feature grouping pipeline.

Context: The model was given the prompt: {prompt_text}

{output_context}

The pipeline produced {num_groups} groups from {len(final_assignments)} features. Your job is to clean up the result — rename unclear groups, reassign misplaced features, and drop irrelevant groups as defined below.

{GROUPING_PHILOSOPHY}

Your job is limited to five things only:

1. GRAMMAR KILL: Any group whose name describes a syntactic role, sentence structure, token pattern, word-prefix fragment, or suppression — move its members to Ungrouped and dissolve it. Examples: "containment verb", "prefix 'ill'", "say location after of", "fill-in-the-blank", "[concept] prefix", "anti-X", "[X] relation", "[X] clause", "location clause". The test: does this name a concept or describe sentence structure / token shape / suppression? Structure/shape/suppression = dissolve. For borderline say-X groups, judge by X: if X names a concept relevant to the prompt or output reasoning chain, keep the group — the "say" framing does not make it irrelevant. Exception: if a word is clearly semantic and central to the prompt's reasoning chain, judge by its role in context rather than its word class alone.

2. ALTERNATE SENSE: A word has an alternate sense when it shares a surface form with the relevant concept but means something different given this prompt — this includes any domain (financial, architectural, political, etc.) that the prompt does not require. Example: "capital (finance)" or "economic capital" when the prompt asks about a state capital. If alternate-sense groups are present, merge them together into a single fallback group named "[concept] (general)" — do not touch the correct-sense group. Do not split the correct-sense group to create a (general) variant; only create "[concept] (general)" by merging existing alternate-sense groups. If no alternate-sense groups exist, take no action. This applies to genuine alternate senses only — do not use this to merge a specific named group into a broader same-sense group. '[concept] (general)' is strictly for features activating on a genuinely different dictionary definition (e.g. 'bank' as a financial institution vs. a riverbank) — not the same concept in different contexts or positions.

3. RENAME: Are any group names unclear, longer than 5 words, or use jargon? Rename for clarity. A rename must not lose specificity, introduce a structural name, or flip a concept group to a say-X group or vice versa. Do not drop intermediate reasoning steps from a group name.

4. REASSIGN: Are any individual features obviously in the wrong group given their description and the prompt? Move them. Only reassign with high confidence.

5. RELEVANCE DROP: If a group's concept has no clear connection to the prompt's reasoning chain or predicted output — it is not a named entity in the prompt, not an intermediate reasoning step, and not a framing pattern for the output — drop it (members to Ungrouped). Use the SPECIFICITY GUIDANCE to judge relevance.

SPECIFIC → BROAD PROTECTION: Before any merge or rename, check — is one group semantically more precise than the other (a named entity, specific concept, or something referenced in the prompt or output)? If yes, protect the specific group. "say capital" must not collapse into "say place name"; "say Texas" must not collapse into "say state". If the specific group is irrelevant to the reasoning chain, send it to Ungrouped — never collapse into a vaguer group.

Only make changes you are CONFIDENT about. If the grouping looks good, return empty lists for all actions.

SPECIFICITY GUIDANCE:
{_bias_phase3}

Current grouping:
{group_summary}
"""

    try:
        response3 = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": phase3_prompt}],
            response_format=Phase3Output,
            max_completion_tokens=8192,
        )
        p3 = response3.choices[0].message.parsed
    except Exception as e:
        log.warning("Phase 3 API call failed (%s) — skipping reconciliation.", e)
        p3 = None

    if p3 is None:
        log.warning("Phase 3 parsing returned None — skipping reconciliation.")
    else:
        total_actions = (
            len(p3.renames) + len(p3.merges)
            + len(p3.reassignments) + len(p3.dropped_groups)
        )
        if total_actions == 0:
            log.info("Phase 3: No changes needed — grouping looks clean.")
        else:
            log.info("Phase 3: Applying %d actions…", total_actions)
            apply_phase3(p3, final_assignments, active_groups)

    append_grouping_log(prompt_text, phase1_group_names, p3, final_assignments)

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