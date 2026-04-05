"""
Step 2 — Generate natural-language descriptions for pruned features.

Loads ``pruned_activations.json`` from the artifacts directory (fetch_all_activation_text.py), prompts
OpenAI's GPT-5-mini model concurrently for each feature, and writes
``feature_descriptions.json``.

If the output file already exists, features
that already have a description are skipped.

Usage:
    export OPENAI_API_KEY="sk-..."
    python add_description.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

from config import (
    CHECKPOINT_INTERVAL,
    DESCRIPTION_MODEL,
    DESCRIPTION_VARIANT,
    FEATURE_DESCRIPTIONS_FILE,
    PRUNED_ACTIVATIONS_FILE,
    GRAPH_FILE,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = DESCRIPTION_MODEL
CONCURRENCY_LIMIT = 167
CHUNK_SIZE = 50 # How many to process before saving a checkpoint

# DESCRIPTION_VARIANT is imported from config (set via DESCRIPTION_VARIANT env var, default "v1").
# This controls both the system prompt used here and the artifact filenames throughout the pipeline.

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

# Shared preamble: evidence types and how to weight them.
_EVIDENCE_BLOCK = (
    "You will receive three types of evidence:\n"
    "1. Overall Prompt Context: the original prompt the model was processing.\n"
    "2. Input Activations: text excerpts where the neuron activated strongly. "
    "The most relevant tokens are delimited by <<<>>>.\n"
    "3. Global Output Tokens: tokens this neuron tends to push toward or away from in the output.\n\n"

    "Use input activations as the primary evidence. "
    "Use prompt context only for disambiguation, not as proof by itself. "
    "Output tokens can be noisy — only factor them in when either show a clear, consistent pattern. If they are consistent, they likely reveal a lot of information.\n\n"

    "STYLE: Write in short, direct fragments — not full sentences. "
    "Get to the point immediately. No filler, no hedging, no grammatical padding. "
    "'Capital of Texas' is better than 'This feature represents the capital of Texas.' "
    "'Say a location' is better than 'This feature is used when a location is about to be mentioned.'\n\n"
)

# ── V0 — original concise (1-4 words) ─────────────────────────────────────
SYSTEM_PROMPT_V0 = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. Use natural, human-readable phrasing for labels."
    "Your task is to produce a short, natural label for the main semantic pattern this feature represents.\n\n"

    + _EVIDENCE_BLOCK +

    "Prefer a short, graph-friendly label over a long explanation. "
    "Choose the one that would look most natural and useful as a node label in an attribution graph. "
    "Preserve important domain-specific information when it is consistently supported by the activations, "
    "even if a broader label would also be technically true.\n"

    "If a narrower subtype is clearly supported, keep it rather than collapsing to a broader generic label. "
    "If the evidence is mixed or weak, prefer a broader but still meaningful label.\n"

    "For function words, prepositions, punctuation, and other structural features: "
    "if they mainly serve a local semantic role in this prompt, label that role rather than the raw token class. Avoid labeling purely grammatical categories unless that is clearly the main feature (Do not include words like preposition, article, demonyms, puncutation, verb, noun, etc.). "
    "The subsequent (or prior) text surrounding a preposition or connector is often what the neuron is describing. For function words, label the semantic slot they say rather than the word itself, often in a short phrase using ‘say,’ for example ‘say a location’ or ‘say a method.’"
    " A pattern does not need to be universal, but ignore one-off details unless they are highly relevant to the prompt and main idea.\n"

    "Include a specific named entity if it appears consistently across the evidence and is clearly related to or is the main shared concept, "
    "not just a side detail (such as a proper name).\n\n"

    "Return only the label text, with no explanation. "
    "Keep it concise, ideally 1-4 words. Prefer specifics (proper nouns, locations, events, methods) over generic labels. "
    "Avoid explanatory phrases, quotes, examples, and parentheses."
)

# ── Shared v2 core ─────────────────────────────────────────────────────────
# All variants below (v1, v2, v3) use the same format: SHORT_LABEL — elaboration.
# Each tests a different emphasis. v0 is kept unchanged as a legacy reference.

_V2_CORE = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. "
    "Your task is to produce a label and brief description for this feature.\n\n"

    + _EVIDENCE_BLOCK +

    "FEATURE TYPES — use this to guide your description style:\n"
    "Features tend to fall into three types. Figure out which one fits, then describe accordingly.\n\n"
    "1. Input features — activate on a specific token or category of tokens.\n"
    "   Describe what they activate on: ‘activates on X’ or just name the pattern directly.\n"
    "   If they activate on a range of related things, describe the category.\n\n"
    "2. Output features — consistently promote a specific next token or category.\n"
    "   Label as ‘say X’ when a clear next-token pattern exists.\n"
    "   Prepositions may fall under this category, where the important words are subsequent to the thing it is referencing."
    "3. Abstract/middle features — neither cleanly input nor output.\n"
    "   Describe the context pattern: what kind of text, what situation, what role it plays.\n"
    "   These often need the surrounding context of activations, not just the highlighted token.\n\n"

    "’SAY X’ vs ‘X ITSELF’:\n"
    "Features either represent a concept directly, or signal that a concept is about to appear "
    "(activating on structural words right before it — prepositions, articles, punctuation).\n"
    "- Highlighted tokens are content words → SHORT_LABEL is the concept itself.\n"
    "- Highlighted tokens are structural words setting up content → SHORT_LABEL is ‘say [what]’.\n"
    "- Unsure? Ask: does the highlighted part carry meaning on its own?\n"
    "- Also check what follows the trigger across activations: if a specific concept X (e.g. a proper noun, a method name) "
    "consistently appears right after the trigger token, that supports ‘say X’. "
    "When unclear, prefer naming the concept directly — ‘say X’ is a stronger claim and needs consistent evidence.\n\n"

    "PROPER NOUNS:\n"
    "If a specific name, place, or entity recurs across the activations — even in a minority of them — "
    "include it in the SHORT_LABEL or elaboration. Don’t collapse to a generic label when a specific one is clearly supported. "
    "Recurring proper nouns are a signal of specificity, not noise.\n\n"

    "AVOID:\n"
    "- Linguistic or technical jargon: copula, lemma, morpheme, orthogonal, syntactic, "
    "prepositional phrase, noun phrase, etc. Prefer layman's vocabulary.\n"
    "- Broad labels when something more specific is clearly supported.\n"
    "- Full sentences. Filler. Hedging.\n\n"
)

# ── V1 — v2 variant: concise, label-first ──────────────────────────────────
# Tests: tighter elaboration budget, label does most of the work.
SYSTEM_PROMPT_V1 = (
    _V2_CORE +

    "OUTPUT FORMAT: SHORT_LABEL — elaboration\n\n"
    "- SHORT_LABEL: 1-4 words. The most specific natural label the evidence supports.\n"
    "- After ‘ — ‘: one tight fragment only. The single most useful extra detail "
    "(context, subpattern, or what it promotes). Omit if the label is already self-explanatory.\n"
    "- Total: 5-20 words.\n\n"

    "Return only the formatted line, nothing else."
)

# ── V2 — base variant ──────────────────────────────────────────────────────
# Tests: balanced label + elaboration with room for specifics.
SYSTEM_PROMPT_V2 = (
    _V2_CORE +

    "OUTPUT FORMAT: SHORT_LABEL — elaboration\n\n"
    "- SHORT_LABEL: 1-5 words. Natural graph node name — specific over generic.\n"
    "- After ‘ — ‘: 1-2 tight fragments. Add context, what it promotes, or consistent subpatterns. "
    "Skip if the label already says it all.\n"
    "- Total: 10-35 words.\n\n"

    "Return only the formatted line, nothing else."
)

# ── V3 — v2 variant: explicit [SAY]/[CONCEPT] tag ─────────────────────────
# Tests: machine-readable classification prefix + same label — elaboration format.
# Format: [SAY] SHORT_LABEL — elaboration  OR  [CONCEPT] SHORT_LABEL — elaboration
SYSTEM_PROMPT_V3 = (
    _V2_CORE +

    "OUTPUT FORMAT: [TAG] SHORT_LABEL — elaboration\n\n"
    "- [TAG]: either [SAY] or [CONCEPT] based on the highlighted tokens.\n"
    "- SHORT_LABEL: 1-5 words. Same as v2 — most specific natural label.\n"
    "- After ‘ — ‘: 1-2 tight fragments, same as v2.\n"
    "- Total (excluding tag): 10-35 words.\n\n"

    "Return only the formatted line, nothing else."
)

# Map variant names to prompts.
DESCRIPTION_VARIANTS: dict[str, str] = {
    "v0": SYSTEM_PROMPT_V0,
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
}

def _get_system_prompt() -> str:
    """Return the system prompt for the configured variant."""
    variant = DESCRIPTION_VARIANT.lower()
    if variant not in DESCRIPTION_VARIANTS:
        log.warning(
            "Unknown DESCRIPTION_VARIANT=%r, falling back to v1. Valid: %s",
            variant, ", ".join(DESCRIPTION_VARIANTS),
        )
        variant = "v1"
    log.info("Using description prompt variant: %s", variant)
    return DESCRIPTION_VARIANTS[variant]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_excerpt(context: str, trigger: str) -> str:
    """Wrap *trigger* in ``<<<>>>`` within *context*."""
    clean = trigger.strip()
    if clean and clean in context:
        return context.replace(clean, f"<<<{clean}>>>", 1)
    return f"{context} [Activates on: <<<{clean}>>>]"

def _build_user_prompt(feature: dict, prompt_text: str) -> str:
    """Compose the user-turn content including prompt context, inputs, and output logits."""
    lines = [f"Neuron {feature.get('id', 'Unknown')}:\n"]

    lines.append("--- OVERALL PROMPT CONTEXT ---")
    lines.append(prompt_text)

    lines.append("\n--- INPUT ACTIVATIONS ---")
    for i, act in enumerate(feature.get("top_activations", [])[:10], 1):
        formatted = _format_excerpt(act.get("context", ""), act.get("trigger", ""))
        lines.append(f"Excerpt {i}: {formatted}")

    lines.append("\n--- GLOBAL OUTPUT TOKENS ---")
    promotes = feature.get("promotes", [])
    demotes = feature.get("demotes", [])

    lines.append(f"Top Promoted Tokens: {', '.join(promotes) if promotes else 'None available'}")
    lines.append(f"Top Demoted Tokens: {', '.join(demotes) if demotes else 'None available'}")

    return "\n".join(lines)
def _load_existing_descriptions(path: Path) -> dict[str, dict]:
    """Return successful prior descriptions only."""
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)

        return {
            item["id"]: item
            for item in data
            if item.get("generated_description")
            and item.get("generated_description") != "Error generating description"
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}

def _save_checkpoint(features: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(features, f, indent=2)
    log.info("💾 Checkpoint saved (%d features) → %s", len(features), path)

# ---------------------------------------------------------------------------
# Async Generation
# ---------------------------------------------------------------------------

async def process_feature(
    feature: dict,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    prompt_text: str,
    system_prompt: str,
) -> None:
    fid = feature.get("id", "?")
    max_retries = 3

    async with sem:
        for attempt in range(1, max_retries + 1):
            log.info(
                "[%d/%d] Requesting GPT-5.4-mini for %s (attempt %d/%d)...",
                idx, total, fid, attempt, max_retries
            )
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _build_user_prompt(feature, prompt_text)},
                    ],
                    reasoning_effort="low",
                    max_completion_tokens=4096,
                )

                desc = response.choices[0].message.content.strip()  # type: ignore

                if "[DESCRIPTION]:" in desc:
                    desc = desc.split("[DESCRIPTION]:", 1)[-1].strip()

                feature["generated_description"] = desc
                log.info("  → %s: %s", fid, desc[:60])
                return

            except Exception as exc:
                log.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, max_retries, fid, exc
                )

                if attempt < max_retries:
                    wait_time = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    log.info("Retrying %s in %ds...", fid, wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    feature["generated_description"] = "Error generating description"
                    log.warning("Giving up on %s after %d attempts.", fid, max_retries)

def _load_prompt_text() -> str:
    """Load original prompt text from graph metadata, if available."""
    if not GRAPH_FILE.exists():
        return "Unknown Prompt"

    try:
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)

        metadata = graph.get("metadata", {})
        prompt_text = metadata.get("prompt", "")

        if not prompt_text:
            input_tokens = metadata.get("prompt_tokens", [])
            if input_tokens:
                prompt_text = "".join(str(t) for t in input_tokens).replace("\n", " ")

        return prompt_text or "Unknown Prompt"
    except Exception:
        return "Unknown Prompt"
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async() -> None:
    if not PRUNED_ACTIVATIONS_FILE.exists():
        log.error("Input file not found: %s", PRUNED_ACTIVATIONS_FILE)
        log.error("Run fetch_all_activating_text.py first (Step 1).")
        sys.exit(1)

    with open(PRUNED_ACTIVATIONS_FILE, "r") as f:
        features: list[dict] = json.load(f)
        prompt_text = _load_prompt_text()
        log.info("Prompt context: %s", prompt_text)

    total = len(features)
    log.info("Loaded %d features from %s", total, PRUNED_ACTIVATIONS_FILE)

    # Select description prompt variant
    system_prompt = _get_system_prompt()

    features_to_process = features
    remaining = len(features_to_process)

    # Initialize Async Client and Semaphore
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    log.info("Starting async generation for %d features...", remaining)

    # Process in chunks to save checkpoints periodically
    for i in range(0, remaining, CHUNK_SIZE):
        chunk = features_to_process[i:i + CHUNK_SIZE]

        # Calculate the absolute index for logging purposes
        tasks = [
            process_feature(
                feat,
                client,
                sem,
                (total - remaining) + i + j + 1,
                total,
                prompt_text,
                system_prompt,
            )
            for j, feat in enumerate(chunk)
        ]

        await asyncio.gather(*tasks)
        _save_checkpoint(features, FEATURE_DESCRIPTIONS_FILE)

    log.info("Done — Descriptions fully generated and saved to %s", FEATURE_DESCRIPTIONS_FILE)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()