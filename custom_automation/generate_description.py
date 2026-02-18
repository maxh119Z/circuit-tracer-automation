"""
Step 2 — Generate natural-language descriptions for pruned features.

Loads ``pruned_activations.json`` from the artifacts directory, prompts a
local Transluce Llama-3 explainer model for each feature, and writes
``feature_descriptions.json``.

Supports **checkpoint resume**: if the output file already exists, features
that already have a description are skipped.

Usage:
    python add_description.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CHECKPOINT_INTERVAL,
    EXPLAINER_MODEL_ID,
    FEATURE_DESCRIPTIONS_FILE,
    PRUNED_ACTIVATIONS_FILE,
    setup_logging,
)

log = setup_logging()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into a specific neuron inside a language model that activates in response "
    "to text excerpts. Your overall task is to describe features of text "
    "excerpts that cause the neuron to strongly activate.\n\n"
    "You will receive a list of text excerpts on which the neuron activates. "
    "Tokens causing activation will appear between delimiters like {{this}}. "
    "Consecutive activating tokens will also be accordingly delimited "
    "{{just like this}}. If no tokens are highlighted with {{}}, then the "
    "neuron does not activate on any tokens in the excerpt.\n\n"
    "Note: Neurons activate on a word-by-word basis. Also, neuron activations "
    "can only depend on words before the word it activates on, so the "
    "description cannot depend on words that come after, and should only "
    "depend on words that come before the activation. Note: make your final "
    "descriptions as concise as possible, using as few words as possible to "
    "describe text features that activate the neuron."
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
    """Compose the user-turn content from a feature's activation examples."""
    lines = [f"Neuron {feature.get('id', 'Unknown')}:\n"]
    for i, act in enumerate(feature.get("top_activations", [])[:10], 1):
        formatted = _format_excerpt(act.get("context", ""), act.get("trigger", ""))
        lines.append(f"Excerpt {i}: {formatted}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(model_id: str = EXPLAINER_MODEL_ID):
    """Load tokenizer and model (float16, auto device-map)."""
    log.info("Loading explainer model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_description(feature: dict, tokenizer, model) -> str:
    """Prompt the explainer model and return a concise description string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(feature)},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids, max_new_tokens=60, do_sample=False)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    # The model often prefixes with "[DESCRIPTION]: …"
    if "[DESCRIPTION]:" in response:
        return response.split("[DESCRIPTION]:")[-1].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


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
    log.debug("Checkpoint saved (%d features) → %s", len(features), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
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
            prev = existing.get(feat.get("id"))
            if prev and "generated_description" in prev:
                feat["generated_description"] = prev["generated_description"]

    tokenizer, model = load_model()

    processed = 0
    for idx, feature in enumerate(features, 1):
        fid = feature.get("id", "?")

        if "generated_description" in feature:
            log.debug("[%d/%d] Skipping %s (already described)", idx, total, fid)
            continue

        log.info("[%d/%d] Processing %s …", idx, total, fid)
        try:
            desc = generate_description(feature, tokenizer, model)
            feature["generated_description"] = desc
            processed += 1
            log.info("  → %s", desc[:60])
        except Exception as exc:
            log.warning("  ✗ Failed: %s", exc)
            feature["generated_description"] = "Error generating description"

        # Periodic checkpoint
        if processed > 0 and processed % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(features, FEATURE_DESCRIPTIONS_FILE)

    # --- Final save ---
    _save_checkpoint(features, FEATURE_DESCRIPTIONS_FILE)
    log.info("Done — %d new descriptions written to %s", processed, FEATURE_DESCRIPTIONS_FILE)


if __name__ == "__main__":
    main()
