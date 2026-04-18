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
CANDIDATES_TO_FETCH = 3500       # how many article sentences to collect before scoring
MAX_PER_ARTICLE = 3             # max prompts per article in final selection
MIN_WORDS = 8                   # minimum words in a truncated prefix
MAX_WORDS = 15                  # maximum words in a truncated prefix
MIN_CONF = 0.15                 # lower bound on top-1 softmax prob (filter boring/random)
MAX_CONF = 0.65                 # upper bound on top-1 softmax prob (filter trivially easy)
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"
BATCH_SIZE = 16                 # prompts scored at once (tune to your GPU VRAM)
SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_FILE = Path(__file__).parent / "ground_truth_wikipedia.csv"
PROMPTS_FILE = Path(__file__).parent / "prompts_wikipedia.csv"

# Diverse Wikipedia categories to seed article selection
SEED_TOPICS = [
    # Science & technology
    "History of science", "Quantum mechanics", "Human evolution", "Neuroscience",
    "Astronomy", "Ecology", "Mathematics", "Medicine", "Computer science",
    "Physics", "Chemistry", "Biology", "Genetics", "Climate change",
    "Artificial intelligence", "Space exploration", "Geology", "Oceanography",
    # History & civilization
    "World War II", "Ancient Rome", "Byzantine Empire", "French Revolution",
    "Industrial Revolution", "Chinese dynasties", "Ancient Greece", "Medieval history",
    "World War I", "Roman Empire", "Cold War", "Age of Exploration",
    "Egyptian civilization", "Mongol Empire", "Ottoman Empire", "British Empire",
    # Arts & culture
    "Renaissance art", "Jazz music", "Classical music", "Impressionism",
    "Architecture", "Literature", "Film history", "Photography", "Opera",
    "Baroque art", "Modernist literature", "Ancient philosophy",
    # Geography & nature
    "Amazon rainforest", "Solar System", "African geography", "Mountain ranges",
    "Major rivers", "Island nations", "Desert ecosystems", "Arctic exploration",
    # Social sciences
    "Linguistics", "Economics", "Sociology", "Anthropology", "Psychology",
    "Political philosophy", "Medieval philosophy", "Music theory",
    # Sports & other
    "Olympic Games", "History of chess", "History of sport",
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
    """Collect ~n candidate prompts from random Wikipedia articles.

    Seeds from topic searches then keeps pulling random pages until we hit n.
    Random pages avoid disambiguation failures that plague topic searches.
    """
    import random

    candidates: list[dict] = []
    seen_slugs: set[str] = set()
    tried_titles: set[str] = set()
    queue: list[str] = []
    consec_refill_failures = 0

    # Seed from topic searches
    random.shuffle(SEED_TOPICS)
    for topic in SEED_TOPICS:
        try:
            queue.extend(wikipedia.search(topic, results=5))
        except Exception:
            pass

    while len(candidates) < n:
        # Refill with random pages (actual articles, not disambiguation pages)
        if not queue:
            try:
                queue.extend(wikipedia.random(pages=50))
                consec_refill_failures = 0
            except Exception:
                consec_refill_failures += 1
                if consec_refill_failures >= 10:
                    print("  Wikipedia API unresponsive — stopping fetch early.")
                    break
                time.sleep(2)
                continue

        title = queue.pop(0)
        if title in tried_titles:
            continue
        tried_titles.add(title)

        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content[:6000]
            new_cands = extract_candidates(title, text, seen_slugs)
            candidates.extend(new_cands[:8])
            time.sleep(0.05)
        except Exception:
            continue

        if len(tried_titles) % 100 == 0:
            print(f"  [{len(tried_titles)} articles tried] {len(candidates)} candidates so far...")

    return candidates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prompts(prompts: list[str], model: HookedTransformer) -> list[dict]:
    """Return top-1 softmax probability and predicted token for each prompt."""
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=True)  # (B, L)
        with torch.no_grad():
            logits = model(tokens)  # (B, L, V)
        last_logits = logits[:, -1, :]  # (B, V)
        probs = F.softmax(last_logits, dim=-1)
        top1_probs, top1_indices = probs.max(dim=-1)  # (B,)
        top1_tokens = model.to_str_tokens(top1_indices)
        for prob, tok in zip(top1_probs.tolist(), top1_tokens):
            results.append({"top1_prob": prob, "top1_token": tok.strip()})
        print(f"  scored {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return results


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
    results = score_prompts(prompt_texts, model)

    for cand, result in zip(candidates, results):
        cand["top1_prob"] = result["top1_prob"]
        cand["correct_answer"] = result["top1_token"]

    # --- Step 3: filter, deduplicate per article, select top 50 ---
    print(f"\n[3/3] Filtering and selecting {TARGET_N} prompts...")

    # Same list as generate_supernodes.py FUNCTION_WORDS
    FUNCTION_WORDS = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "over",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "it", "its", "he", "she", "they", "them", "his", "her", "we", "you",
        "i", "me", "my", "your", "our", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might",
        "not", "no", "if", "then", "than", "as", "which", "who", "whom",
        "what", "when", "where", "how", "that",
        ",", ".", ":", ";", "!", "?", "'", '"', "(", ")", "-", "—", "",
        "<bos>", "<eos>", "<pad>", "<s>", "</s>",
        "containing",
    })

    def _is_clean_token(tok: str) -> bool:
        t = tok.strip().lower()
        if not t or len(t) < 3:                     # filter single chars and very short fragments
            return False
        if not t[0].isalpha():                      # no punctuation, digits, markup
            return False
        if t in FUNCTION_WORDS:
            return False
        # Filter subword fragments: real words don't start with lowercase after a capital
        # and don't look like truncated tokens (e.g. 'Hoo', 'psych', 'kháu')
        if len(tok.strip()) < 4 and tok.strip()[0].isupper():
            return False                             # 3-char capitalised fragments like 'Hoo'
        # Filter non-ASCII tokens (markup artifacts, non-English subwords)
        if not tok.strip().isascii():
            return False
        # Filter camelCase/PascalCase markup artifacts like 'UnknownFieldSet', 'SourceChecksum'
        if re.search(r'[a-z][A-Z]', tok):
            return False
        # Filter tokens with suspicious substrings
        if any(s in tok for s in ["\\", "{", "}", "<", ">", "Field", "Checksum", "rawDesc"]):
            return False
        return True

    def _is_clean_prompt(prompt: str) -> bool:
        if not prompt or not prompt[0].isupper():   # no mid-sentence fragments
            return False
        # Avoid context-free pronoun openers
        for opener in ("It ", "This ", "They ", "He ", "She ", "Also ", "However "):
            if prompt.startswith(opener):
                return False
        # Avoid prompts with non-ASCII (Japanese, Chinese, etc. mixed in)
        if not prompt.isascii():
            return False
        return True

    interesting = [
        c for c in candidates
        if MIN_CONF <= c["top1_prob"] <= MAX_CONF
        and _is_clean_token(c["correct_answer"])
        and _is_clean_prompt(c["prompt"])
    ]
    print(f"  {len(interesting)} candidates after quality + confidence filtering.")

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

    # --- Write ground truth CSV ---
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slug", "prompt", "intermediate_concept", "correct_answer", "hop_type", "notes"])
        for cand in selected:
            writer.writerow([
                cand["slug"],
                cand["prompt"],
                "N/A",
                cand["correct_answer"],
                "next-token prediction",
                f"wikipedia article: {cand['article']} | top1_prob: {cand['top1_prob']:.3f}",
            ])

    # --- Write prompts CSV (for attribution batch) ---
    with PROMPTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slug", "prompt", "transcoder_set"])
        for cand in selected:
            writer.writerow([cand["slug"], cand["prompt"], "gemma"])

    print(f"\nWrote {len(selected)} prompts to {OUT_FILE}")
    print(f"Wrote {len(selected)} prompts to {PROMPTS_FILE}")
    print("\nSample prompts:")
    for cand in selected[:5]:
        print(f"  [{cand['top1_prob']:.2f}] {cand['prompt']!r} -> {cand['correct_answer']!r}")


if __name__ == "__main__":
    main()
