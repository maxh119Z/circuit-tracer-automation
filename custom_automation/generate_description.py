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
    FEATURE_DESCRIPTIONS_FILE,
    PRUNED_ACTIVATIONS_FILE,
    GRAPH_FILE,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-5-mini"
CONCURRENCY_LIMIT = 67
CHUNK_SIZE = 50 # How many to process before saving a checkpoint

# Which description prompt variant to use.  Set DESCRIPTION_VARIANT env var
# to one of: v0, v1, v2, v3.  Default is v1 (longer, more specific).
DESCRIPTION_VARIANT: str = os.environ.get("DESCRIPTION_VARIANT", "v1")

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

# Shared preamble: evidence types and how to weight them.
_EVIDENCE_BLOCK = (
    "You will receive three types of evidence:\n"
    "1. Overall Prompt Context: the original prompt the model was processing.\n"
    "2. Input Activations: text excerpts where the neuron activated strongly. "
    "The triggering tokens are delimited by <<<>>>.\n"
    "3. Global Output Tokens: tokens the neuron tends to promote and demote across the vocabulary.\n\n"

    "Use input activations as the primary evidence. "
    "Use prompt context only for disambiguation, not as proof by itself. "
    "Use output tokens only as supporting evidence (when outputs exhibits clear patterns "
    "or if other information is noisy), since they may be noisy, polysemantic, or tokenizer artifacts.\n\n"
)

# ── V0 — original concise (1-4 words) ─────────────────────────────────────
SYSTEM_PROMPT_V0 = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. Treat neurons as humans for verb choosing."
    "Your task is to produce a short, natural label for the main semantic pattern this feature represents.\n\n"

    + _EVIDENCE_BLOCK +

    "Prefer a short, graph-friendly label over a long explanation. "
    "Choose the one that would look most natural and useful as a node label in an attribution graph. "
    "Preserve important domain-specific information when it is consistently supported by the activations, "
    "even if a broader label would also be technically true.\n"

    "If a narrower subtype is clearly supported, keep it rather than collapsing to a broader generic label. "
    "If the evidence is mixed or weak, prefer a broader but still meaningful label; if text includes words meaningful to the prompt, could make educated guesses without full consistency. \n"

    "For function words, prepositions, punctuation, and other structural features: "
    "if they mainly serve a local semantic role in this prompt, label that role rather than the raw token class. Avoid labeling purely grammatical categories unless that is clearly the main feature (Do not include words like preposition, article, demonyms, puncutation, verb, noun, etc.). "
    "The subsequent (or prior) text surrounding a preposition or connector is often what the neuron is describing. For function words, label the semantic slot they say rather than the word itself, often in a short phrase using ‘say,’ for example ‘say a location’ or ‘say a method.’"
    " A pattern does not need to be universal, but ignore one-off details unless they are highly relevant to the prompt and main idea.\n"

    "Include a specific named entity if it appears consistently across the evidence and is clearly related to or is the main shared concept, "
    "not just a side detail (such as a proper name).\n\n"

    "Return only the label text, with no explanation. "
    "Prefer specificity over something too general! Descriptions of specific proper nouns, names, places, methods, etc are important. "
    "Consider surrounding context of input activations as secondary evidence if important words pop up relevant to the prompt; do not rely only on the trigger tokens. "
    "Avoid explanatory phrases, quotes, examples, and parentheses in your label text unless truly necessary. "
    "Keep it concise, ideally 1-4 words. Prefer layman’s terms that catches specifics (proper nouns, locations, events, etc.) but without complex terminology unless it is relevant to the prompt."
)

# ── V1 — detailed descriptive (1-2 sentences) ─────────────────────────────
# Goal: longer descriptions that preserve information for downstream grouping.
# Explicitly classifies "say X" vs "represents X".
SYSTEM_PROMPT_V1 = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. "
    "Your task is to produce a detailed, specific description of the semantic pattern this feature represents.\n\n"

    + _EVIDENCE_BLOCK +

    "CRITICAL DISTINCTION — ‘say X’ vs ‘X itself’:\n"
    "A feature can either (a) represent the concept X itself, or (b) represent the "
    "act of mentioning / introducing / referring to X in text. These are fundamentally different.\n"
    "- If the feature fires on tokens that ARE the concept (e.g., the word ‘Paris’ in a "
    "feature about Paris), it represents the concept itself. Describe it as the concept.\n"
    "- If the feature fires on structural tokens (prepositions, articles, punctuation, "
    "connectors) that SET UP or INTRODUCE a concept, it is a ‘say’ feature. "
    "Describe it as ‘say [what it introduces]’ — e.g., ‘say a location’, ‘say a method’, "
    "’say a comparison’.\n"
    "- Look at what tokens trigger the feature (the <<<>>> markers). If the triggers are "
    "content words (nouns, verbs, adjectives carrying meaning), it’s likely the concept itself. "
    "If the triggers are function words, punctuation, or structural tokens, it’s likely a ‘say’ feature.\n"
    "- When in doubt, look at the promoted output tokens: if they are content words related "
    "to a domain, the feature likely represents that domain concept. If promoted tokens are "
    "structural or varied, it’s more likely a ‘say’ feature.\n\n"

    "DESCRIPTION GUIDELINES:\n"
    "- Write 1-2 sentences that summarize ALL the evidence about this feature, not just the strongest signal.\n"
    "- Be specific: include named entities, domain terms, and concrete details that appear "
    "consistently across activations.\n"
    "- Mention the key trigger pattern: what kind of tokens activate this feature?\n"
    "- If the feature is clearly about a specific concept (person, place, method, event), name it.\n"
    "- If the feature captures a discourse role (introducing, comparing, listing), say so explicitly.\n"
    "- Preserve important nuances rather than collapsing to a generic label.\n"
    "- Do NOT pad with filler or hedging. Every word should carry information.\n"
    "- Aim for 10-30 words. Longer is fine if the extra words carry genuine information.\n\n"

    "Return only the description text, with no explanation or meta-commentary."
)

# ── V2 — structured label + explanation ────────────────────────────────────
# Goal: a short label AND a longer explanation, separated by " — ".
# This lets downstream code use either the label or the full description.
SYSTEM_PROMPT_V2 = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. "
    "Your task is to produce a label and description for this feature.\n\n"

    + _EVIDENCE_BLOCK +

    "CRITICAL DISTINCTION — ‘say X’ vs ‘X itself’:\n"
    "Determine whether this feature represents a concept directly, or represents the "
    "linguistic act of mentioning/introducing that concept:\n"
    "- Concept features fire on content words (nouns, verbs, domain terms) that ARE the concept.\n"
    "- ‘Say’ features fire on function words, punctuation, or structural tokens that "
    "set up or introduce a concept. Label these as ‘say [what]’.\n"
    "- Check the <<<>>> trigger tokens: content words → concept; function words → ‘say’.\n"
    "- Check promoted output tokens for confirmation.\n\n"

    "OUTPUT FORMAT:\n"
    "Return a single line in this format:\n"
    "  SHORT_LABEL — Detailed explanation of what this feature captures, what tokens "
    "trigger it, and what semantic role it plays.\n\n"
    "- SHORT_LABEL: 1-5 words, suitable as a graph node label.\n"
    "- After the ‘ — ‘: 1-2 sentences summarizing all evidence. Be specific about "
    "trigger patterns, promoted tokens, and the feature’s role in the prompt’s meaning.\n"
    "- Include named entities, domain specifics, and concrete patterns.\n"
    "- Total length: 15-40 words.\n\n"

    "Return only the formatted description, nothing else."
)

# ── V3 — say-classification focused ───────────────────────────────────────
# Goal: maximally clear on say vs concept, with a classification tag.
# Output starts with [SAY] or [CONCEPT] for easy downstream parsing.
SYSTEM_PROMPT_V3 = (
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. "
    "Your task is to classify and describe this feature.\n\n"

    + _EVIDENCE_BLOCK +

    "STEP 1 — CLASSIFY the feature as [SAY] or [CONCEPT]:\n\n"
    "[CONCEPT] features represent the concept itself:\n"
    "- The <<<>>> trigger tokens are content words (nouns, adjectives, verbs, proper names) "
    "that directly embody the concept.\n"
    "- Promoted output tokens relate to the same semantic domain.\n"
    "- Example: a feature triggering on ‘Einstein’, ‘relativity’, ‘physics’ → [CONCEPT] physics / Einstein.\n\n"

    "[SAY] features represent the act of mentioning or introducing a concept:\n"
    "- The <<<>>> trigger tokens are function words (prepositions, articles, conjunctions, "
    "punctuation, connectors) that SET UP a slot for content.\n"
    "- The feature fires on structural tokens adjacent to meaningful content.\n"
    "- Promoted output tokens may be the content that fills the slot.\n"
    "- Example: a feature triggering on ‘in’ before location names → [SAY] say a location.\n\n"

    "AMBIGUOUS CASES:\n"
    "- If trigger tokens are a mix of content and function words, classify based on the majority.\n"
    "- If the feature seems to fire on a transition or framing pattern, prefer [SAY].\n"
    "- If the feature fires on domain-specific content words, prefer [CONCEPT].\n\n"

    "STEP 2 — DESCRIBE the feature:\n"
    "- After the tag, write a specific description (10-30 words).\n"
    "- Summarize what tokens trigger it, what it promotes, and its role in the prompt.\n"
    "- Include named entities, specific patterns, and domain details.\n"
    "- For [SAY] features, specify what kind of content the feature introduces.\n"
    "- For [CONCEPT] features, specify what concept and any important sub-patterns.\n\n"

    "OUTPUT FORMAT:\n"
    "Return a single line: [SAY] or [CONCEPT] followed by the description.\n"
    "Example: [SAY] Introduces location references, firing on prepositions like ‘in’ and ‘at’ before city and country names.\n"
    "Example: [CONCEPT] Represents Paris and French geography, triggering on ‘Paris’, ‘France’, and related terms.\n\n"
    "Return only the classification and description, nothing else."
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
                "[%d/%d] Requesting GPT-5-mini for %s (attempt %d/%d)...",
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
                    max_completion_tokens=2048,
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

    # --- Checkpoint resume ---
    existing = _load_existing_descriptions(FEATURE_DESCRIPTIONS_FILE)
    if existing:
        log.info("Resuming — %d features already described, skipping those.", len(existing))
        for feat in features:
            prev = existing.get(feat.get("id")) # type: ignore
            if prev and "generated_description" in prev:
                feat["generated_description"] = prev["generated_description"]

    # Filter out features that already have descriptions
    features_to_process = [f for f in features if "generated_description" not in f]
    remaining = len(features_to_process)

    if remaining == 0:
        log.info("All features already described. Exiting.")
        return

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