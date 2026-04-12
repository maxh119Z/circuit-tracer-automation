"""
Generate 50 Wikipedia next-token-prediction prompts for circuit tracing.

Strategy:
  1. Fetch random Wikipedia articles across diverse categories.
  2. Extract sentences, truncate each to a natural mid-sentence prefix (8-15 words).
  3. Run each candidate through Gemma-2-2B and compute the top-1 softmax probability
     for the next token.
  4. Keep candidates whose top-1 probability falls in the "interesting" window
     (MIN_CONF, MAX_CONF) — confident enough to trace, uncertain enough to be non-trivial.
  5. Cap at MAX_PER_ARTICLE per article for diversity.
  6. Write the top TARGET_N results to prompts_wikipedia.csv.

Usage (on the desktop with the model):
    pip install wikipedia transformer_lens torch
    python prompts/generate_wikipedia_prompts.py
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wikipedia
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_N = 50
CANDIDATES_TO_FETCH = 300       # how many article sentences to collect before scoring
MAX_PER_ARTICLE = 2             # diversity cap per article
MIN_WORDS = 8                   # minimum words in a truncated prefix
MAX_WORDS = 15                  # maximum words in a truncated prefix
MIN_CONF = 0.15                 # lower bound on top-1 softmax prob (filter boring/random)
MAX_CONF = 0.65                 # upper bound on top-1 softmax prob (filter trivially easy)
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"
BATCH_SIZE = 16                 # prompts scored at once (tune to your GPU VRAM)
SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_FILE = Path(__file__).parent / "prompts_wikipedia.csv"

# Diverse Wikipedia categories to seed article selection
SEED_TOPICS = [
    "History of science", "World War II", "Ancient Rome", "Renaissance art",
    "Quantum mechanics", "Human evolution", "Climate change", "Jazz music",
    "Olympic Games", "Byzantine Empire", "French Revolution", "Solar System",
    "Medieval philosophy", "Amazon rainforest", "Industrial Revolution",
    "Chinese dynasties", "Neuroscience", "Architecture", "Linguistics",
    "Astronomy", "Ecology", "Economics", "Ancient Greece", "Mathematics",
    "Literature", "Geography", "Medicine", "Philosophy", "Sociology", "Music theory",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLEANUP = re.compile(r"\s+")


def _slug(article_title: str, word_idx: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", article_title.lower()).strip("-")
    return f"{base}-{word_idx}"


def extract_candidates(article_title: str, text: str, seen_slugs: set[str]) -> list[dict]:
    """Return truncated sentence prefixes from one article."""
    candidates = []
    sentences = _SENTENCE_SPLIT.split(text)
    for sent in sentences:
        sent = _CLEANUP.sub(" ", sent).strip()
        words = sent.split()
        if len(words) < MIN_WORDS + 2:
            continue
        # Try two truncation lengths per sentence for variety
        for trunc in range(MIN_WORDS, min(MAX_WORDS + 1, len(words) - 1)):
            prefix = " ".join(words[:trunc])
            slug = _slug(article_title, trunc)
            if slug in seen_slugs:
                continue
            # Basic quality filters
            if not prefix[-1].isalnum():   # avoid trailing punctuation
                continue
            if any(c in prefix for c in ["[", "]", "{", "}"]):  # skip wiki markup
                continue
            seen_slugs.add(slug)
            candidates.append({"slug": slug, "prompt": prefix, "article": article_title})
    return candidates


def fetch_candidates(n: int) -> list[dict]:
    """Collect ~n candidate prompts from random Wikipedia articles."""
    candidates: list[dict] = []
    seen_slugs: set[str] = set()
    tried_titles: set[str] = set()
    article_counts: dict[str, int] = {}

    # First pass: use seed topics to get good article titles
    import random
    random.shuffle(SEED_TOPICS)
    titles_queue: list[str] = []

    for topic in SEED_TOPICS:
        try:
            results = wikipedia.search(topic, results=5)
            titles_queue.extend(results)
        except Exception:
            pass

    # Pad with random articles
    for _ in range(30):
        try:
            titles_queue.extend(wikipedia.random(pages=5))
        except Exception:
            pass

    for title in titles_queue:
        if len(candidates) >= n:
            break
        if title in tried_titles:
            continue
        if article_counts.get(title, 0) >= MAX_PER_ARTICLE:
            continue
        tried_titles.add(title)
        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content[:4000]  # first ~4000 chars is enough
            new_cands = extract_candidates(title, text, seen_slugs)
            # Limit per article
            new_cands = new_cands[: MAX_PER_ARTICLE * 6]  # keep some to score, trim later
            candidates.extend(new_cands)
            article_counts[title] = article_counts.get(title, 0) + len(new_cands)
            time.sleep(0.1)  # be polite to Wikipedia API
        except Exception:
            continue

    return candidates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prompts(prompts: list[str], model: HookedTransformer) -> list[float]:
    """Return top-1 softmax probability for the last token of each prompt."""
    scores = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=True)  # (B, L)
        with torch.no_grad():
            logits = model(tokens)  # (B, L, V)
        last_logits = logits[:, -1, :]  # (B, V)
        probs = F.softmax(last_logits, dim=-1)
        top1 = probs.max(dim=-1).values  # (B,)
        scores.extend(top1.tolist())
        print(f"  scored {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Wikipedia prompt generator ===")
    print(f"Target: {TARGET_N} prompts | confidence window: [{MIN_CONF}, {MAX_CONF}]")

    # --- Step 1: collect candidates ---
    print(f"\n[1/3] Fetching ~{CANDIDATES_TO_FETCH} candidate prompts from Wikipedia...")
    candidates = fetch_candidates(CANDIDATES_TO_FETCH)
    print(f"  Collected {len(candidates)} candidates from Wikipedia.")

    # --- Step 2: score with Gemma ---
    print(f"\n[2/3] Loading {MODEL_NAME} on {SCORE_DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=SCORE_DEVICE)
    model.eval()

    print(f"  Scoring {len(candidates)} prompts...")
    prompt_texts = [c["prompt"] for c in candidates]
    scores = score_prompts(prompt_texts, model)

    for cand, score in zip(candidates, scores):
        cand["top1_prob"] = score

    # --- Step 3: filter, deduplicate per article, select top 50 ---
    print(f"\n[3/3] Filtering and selecting {TARGET_N} prompts...")
    interesting = [
        c for c in candidates
        if MIN_CONF <= c["top1_prob"] <= MAX_CONF
    ]
    print(f"  {len(interesting)} candidates in confidence window [{MIN_CONF}, {MAX_CONF}].")

    # Sort by distance from midpoint of confidence window (most "interesting" first)
    midpoint = (MIN_CONF + MAX_CONF) / 2
    interesting.sort(key=lambda c: abs(c["top1_prob"] - midpoint))

    # Enforce per-article diversity cap
    article_used: dict[str, int] = {}
    selected = []
    for cand in interesting:
        article = cand["article"]
        if article_used.get(article, 0) >= MAX_PER_ARTICLE:
            continue
        article_used[article] = article_used.get(article, 0) + 1
        selected.append(cand)
        if len(selected) == TARGET_N:
            break

    if len(selected) < TARGET_N:
        print(
            f"  WARNING: only found {len(selected)} prompts in confidence window. "
            "Consider lowering MIN_CONF or raising MAX_CONF."
        )

    # --- Write CSV ---
    with OUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slug", "prompt", "transcoder_set"])
        for cand in selected:
            writer.writerow([cand["slug"], cand["prompt"], TRANSCODER_SET])

    print(f"\nWrote {len(selected)} prompts to {OUT_FILE}")
    print("\nSample prompts:")
    for cand in selected[:5]:
        print(f"  [{cand['top1_prob']:.2f}] {cand['prompt']!r}")


if __name__ == "__main__":
    main()
