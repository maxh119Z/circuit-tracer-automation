"""
Step 2 — Generate natural-language descriptions for pruned features.

Loads ``pruned_activations.json`` from the artifacts directory, prompts
OpenAI's GPT-5-mini model concurrently for each feature, and writes
``feature_descriptions.json``.

Supports **checkpoint resume**: if the output file already exists, features
that already have a description are skipped.

Usage:
    export OPENAI_API_KEY="sk-..."
    python add_description.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

from config import (
    CHECKPOINT_INTERVAL,
    FEATURE_DESCRIPTIONS_FILE,
    PRUNED_ACTIVATIONS_FILE,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-5-mini"
CONCURRENCY_LIMIT = 50  # How many simultaneous requests to send to OpenAI
CHUNK_SIZE = 50         # How many to process before saving a checkpoint

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into a specific neuron inside a language model. Your task is to concisely "
    "describe the concept or feature that this neuron represents.\n\n"
    "You will receive two types of evidence:\n"
    "1. Input Activations: Text excerpts where the neuron activated strongly. "
    "The specific tokens causing activation are delimited by {{ }}. "
    "Remember that activations only depend on preceding tokens, never subsequent ones.\n"
    "2. Global Output Tokens: A list of tokens the neuron intrinsically promotes "
    "(increases probability) and demotes (decreases probability) across the vocabulary.\n\n"
    "CRITICAL WARNING: The output tokens (promoted/demoted) can often be noisy, "
    "polysemantic, or artifacts of the tokenizer. Be wary of this noise. "
    "Use the output tokens merely as directional hints to support the context seen "
    "in the input activations, not as absolute truth.\n"
    "In some cases, if input activations are noisy but output tokens follow clear patterns, listen to the tokens.""\n"
    "IMPORTANT:\n"
    "If the neuron consistently activates on or near a specific named entity "
    "(person, place, organization, brand, etc.), Include that entity in "
    "the description.\n"
    "SPECIFICITY OVER GENERALITY:\n"
    "Prefer the most specific accurate description over a vague general one."
    "Keep your final description as concise as possible — ideally <= 5 words. "
    "There is no need for much grammatical correctness."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_excerpt(context: str, trigger: str) -> str:
    """Wrap *trigger* in ``{{ }}`` within *context*."""
    clean = trigger.strip()
    if clean and clean in context:
        return context.replace(clean, f"{{{{{clean}}}}}", 1)
    return f"{context} [Activates on: {{{{{clean}}}}}]"

def _build_user_prompt(feature: dict) -> str:
    """Compose the user-turn content including inputs and output logits."""
    lines = [f"Neuron {feature.get('id', 'Unknown')}:\n"]
    
    lines.append("--- INPUT ACTIVATIONS ---")
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
    """Return a mapping of feature-id → feature-dict from a prior run."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {item["id"]: item for item in data if "generated_description" in item}
    except (json.JSONDecodeError, KeyError):
        return {}

def _save_checkpoint(features: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(features, f, indent=2)
    log.info("💾 Checkpoint saved (%d features) → %s", len(features), path)

# ---------------------------------------------------------------------------
# Async Generation
# ---------------------------------------------------------------------------

async def process_feature(feature: dict, client: AsyncOpenAI, sem: asyncio.Semaphore, idx: int, total: int) -> None:
    fid = feature.get("id", "?")
    
    async with sem:
        log.info("[%d/%d] Requesting GPT-5-mini for %s ...", idx, total, fid)
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(feature)},
                ],
                reasoning_effort="low",
                max_completion_tokens=1024
            )
            desc = response.choices[0].message.content.strip() # type: ignore
            
            if "[DESCRIPTION]:" in desc:
                desc = desc.split("[DESCRIPTION]:")[-1].strip()
                
            feature["generated_description"] = desc
            log.info("  → %s: %s", fid, desc[:60])
            
        except Exception as exc:
            log.warning("  ✗ Failed for %s: %s", fid, exc)
            feature["generated_description"] = "Error generating description"

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

    total = len(features)
    log.info("Loaded %d features from %s", total, PRUNED_ACTIVATIONS_FILE)

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
            process_feature(feat, client, sem, (total - remaining) + i + j + 1, total) 
            for j, feat in enumerate(chunk)
        ]
        
        await asyncio.gather(*tasks)
        _save_checkpoint(features, FEATURE_DESCRIPTIONS_FILE)

    log.info("Done — Descriptions fully generated and saved to %s", FEATURE_DESCRIPTIONS_FILE)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()