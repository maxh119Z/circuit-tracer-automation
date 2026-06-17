"""
Generate 100 Wikipedia next-token-prediction prompts for circuit tracing.

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

import argparse
import csv
import random
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

TARGET_N = 100
CANDIDATES_TO_FETCH = 5000       # how many article sentences to collect before scoring
MAX_PER_ARTICLE = 5             # max prompts per article in final selection
MIN_WORDS = 8                   # minimum words in a truncated prefix
MAX_WORDS = 15                  # maximum words in a truncated prefix
MIN_CONF = 0.0                  # lower bound on top-1 softmax prob (no floor — model_correct already implies non-trivial mass)
MAX_CONF = 0.8                  # upper bound on top-1 softmax prob (filter trivially easy)
MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"
BATCH_SIZE = 4                  # prompts scored at once (tune to your GPU VRAM)
SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Wikipedia now returns HTTP 403 ("Please set a user-agent") for the wikipedia
# library's generic default UA, so EVERY fetch silently fails and the run yields
# only a handful of prompts regardless of --candidates. A descriptive User-Agent
# fixes it. Override with the WIKIPEDIA_USER_AGENT env var or --user-agent.
import os as _os
WIKIPEDIA_USER_AGENT = _os.environ.get(
    "WIKIPEDIA_USER_AGENT",
    "circuit-tracer-automation/0.1 (research; +https://github.com/safety-research/circuit-tracer)",
)

# When True, each prompt is built as: <full previous sentence> + <truncated current sentence>.
# The truncated portion still has length MIN_WORDS..MAX_WORDS; the prepended sentence
# only adds context. Output filenames get a "_context" suffix in this mode so the
# original no-context outputs are not overwritten.
PREPEND_PREVIOUS_SENTENCE = True

# Appended to every slug (e.g. "ctx" -> "...-s26-11-ctx"). Lets plain and context
# runs share one graph pool without filename collisions. Set per-run in main().
SLUG_TAG = ""

_suffix = "_context" if PREPEND_PREVIOUS_SENTENCE else ""
OUT_FILE = Path(__file__).parent / f"ground_truth_wikipedia{_suffix}.csv"
PROMPTS_FILE = Path(__file__).parent / f"prompts_wikipedia{_suffix}.csv"

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
_WORD_NORMALIZE = re.compile(r"^[^\w]+|[^\w]+$")
# Wikipedia section headers like "== Ancient history ==" or "=== Subsection ==="
# don't end in .!? so they get glued onto the next sentence by _SENTENCE_SPLIT.
# Strip them out before splitting.
_WIKI_HEADER = re.compile(r"^=+\s*[^=\n]+?\s*=+\s*$", re.MULTILINE)


def _normalize_word(w: str) -> str:
    """Strip surrounding punctuation/whitespace and lowercase."""
    return _WORD_NORMALIZE.sub("", w.strip()).lower()


def _slug(article_title: str, sent_idx: int, word_idx: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", article_title.lower()).strip("-")
    slug = f"{base}-s{sent_idx}-{word_idx}"
    return f"{slug}-{SLUG_TAG}" if SLUG_TAG else slug


def extract_candidates(article_title: str, text: str, seen_slugs: set[str]) -> list[dict]:
    """Return truncated sentence prefixes from one article (shuffled, so the
    caller's [:N] cap samples random sentences instead of always the first few).

    If PREPEND_PREVIOUS_SENTENCE is True, each prompt is the full previous
    sentence + the truncated current-sentence prefix. The truncation length
    still respects MIN_WORDS..MAX_WORDS for the current sentence only.
    """
    candidates = []
    text = _WIKI_HEADER.sub("", text)
    sentences = [_CLEANUP.sub(" ", s).strip() for s in _SENTENCE_SPLIT.split(text)]
    for sent_idx, sent in enumerate(sentences):
        if PREPEND_PREVIOUS_SENTENCE and sent_idx == 0:
            continue                        # need a previous sentence as context
        words = sent.split()
        if len(words) < MIN_WORDS + 2:
            continue

        prev_sent = ""
        if PREPEND_PREVIOUS_SENTENCE:
            # A complete prior sentence is REQUIRED — drop the pair otherwise.
            prev_sent = sentences[sent_idx - 1]
            if (
                not prev_sent
                or not prev_sent.endswith((".", "!", "?"))   # must be a complete sentence
                or not prev_sent[0].isupper()                # must look like a real sentence start
                or not prev_sent.isascii()
                or any(c in prev_sent for c in ["[", "]", "{", "}", "="])
            ):
                continue

        for trunc in range(MIN_WORDS, min(MAX_WORDS + 1, len(words) - 1)):
            prefix = " ".join(words[:trunc])
            expected = _normalize_word(words[trunc])
            if not expected:                # next word is pure punctuation
                continue
            slug = _slug(article_title, sent_idx, trunc)
            if slug in seen_slugs:
                continue
            # Basic quality filters (applied to the truncated current portion)
            if not prefix[-1].isalnum():   # avoid trailing punctuation
                continue
            if any(c in prefix for c in ["[", "]", "{", "}", "="]):  # skip wiki markup
                continue
            full_prompt = (prev_sent + " " + prefix) if PREPEND_PREVIOUS_SENTENCE else prefix
            seen_slugs.add(slug)
            candidates.append({
                "slug": slug,
                "prompt": full_prompt,
                "article": article_title,
                "expected_answer": expected,
            })
    random.shuffle(candidates)
    return candidates


def fetch_candidates(n: int) -> list[dict]:
    """Collect ~n candidate prompts from random Wikipedia articles.

    Seeds from topic searches then keeps pulling random pages until we hit n.
    Random pages avoid disambiguation failures that plague topic searches.
    """
    candidates: list[dict] = []
    seen_slugs: set[str] = set()
    tried_titles: set[str] = set()
    queue: list[str] = []
    consec_refill_failures = 0
    page_failures = 0
    page_successes = 0

    # Seed from topic searches
    random.shuffle(SEED_TOPICS)
    for topic in SEED_TOPICS:
        try:
            queue.extend(wikipedia.search(topic, results=5))
        except Exception as e:
            print(f"  seed search failed for {topic!r}: {type(e).__name__}: {e}")

    while len(candidates) < n:
        # Refill with random pages (actual articles, not disambiguation pages)
        if not queue:
            try:
                queue.extend(wikipedia.random(pages=50))
                consec_refill_failures = 0
            except Exception as e:
                consec_refill_failures += 1
                backoff = min(2 ** consec_refill_failures, 60)  # exp backoff up to 60s
                print(f"  random refill failed ({consec_refill_failures}/30): "
                      f"{type(e).__name__}: {e} — sleeping {backoff}s")
                if consec_refill_failures >= 30:
                    print("  Wikipedia API unresponsive after 30 retries — stopping fetch early.")
                    break
                time.sleep(backoff)
                continue

        title = queue.pop(0)
        if title in tried_titles:
            continue
        tried_titles.add(title)

        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content[:6000]
            new_cands = extract_candidates(title, text, seen_slugs)
            candidates.extend(new_cands[:40])
            page_successes += 1
            time.sleep(0.05)
        except Exception as e:
            page_failures += 1
            # Log a sample failure every 200 errors so we can see what's happening
            # without spamming the console with disambiguation/missing-page noise.
            if page_failures % 200 == 1:
                print(f"  page failure #{page_failures} for {title!r}: {type(e).__name__}: {e}")

        if len(tried_titles) % 100 == 0:
            ok_rate = page_successes / max(1, page_successes + page_failures)
            print(f"  [{len(tried_titles)} articles tried | {page_successes} ok, "
                  f"{page_failures} failed ({ok_rate:.0%})] "
                  f"{len(candidates)} candidates so far...")

    print(f"  Final: {page_successes} successful pages, {page_failures} failed, "
          f"{len(candidates)} candidates.")
    return candidates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prompts(prompts: list[str], model: HookedTransformer) -> list[dict]:
    """Return top-1 softmax probability and predicted token for each prompt."""
    results = []
    pad_id = model.tokenizer.pad_token_id
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=True)  # (B, L)
        # Right-padding means logits[:, -1] reads from a PAD position for every
        # prompt that isn't the longest in the batch. Find each row's real last
        # non-pad index and read the logit there.
        if pad_id is not None:
            non_pad = (tokens != pad_id).long()
            last_idx = non_pad.sum(dim=-1) - 1  # (B,)
        else:
            last_idx = torch.full((tokens.size(0),), tokens.size(1) - 1,
                                  device=tokens.device, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens)  # (B, L, V)
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_idx, last_idx, :].float()  # (B, V)
            probs = F.softmax(last_logits, dim=-1)
            top1_probs, top1_indices = probs.max(dim=-1)  # (B,)
            # Move to CPU before extending so we can free GPU memory immediately
            top1_probs_cpu = top1_probs.tolist()
            top1_indices_cpu = top1_indices.cpu()
        top1_tokens = model.to_str_tokens(top1_indices_cpu)
        for prob, tok in zip(top1_probs_cpu, top1_tokens):
            results.append({"top1_prob": prob, "top1_token": tok.strip()})
        del tokens, logits, last_logits, probs, top1_probs, top1_indices
        if SCORE_DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  scored {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_excluded_slugs(paths: list[str]) -> set[str]:
    """Collect slugs already used in earlier prompt/ground-truth CSVs so a new
    run produces a DISJOINT set (e.g. 800 new on top of the existing 200)."""
    excluded: set[str] = set()
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            print(f"  WARNING: --exclude file not found, skipping: {fp}")
            continue
        with fp.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                slug = (row.get("slug") or "").strip()
                if slug:
                    excluded.add(slug)
    return excluded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Wikipedia next-token prompts for circuit tracing.")
    p.add_argument("--target", type=int, default=TARGET_N,
                   help=f"Number of prompts to select (default {TARGET_N}). Use 800 to scale up.")
    p.add_argument("--candidates", type=int, default=CANDIDATES_TO_FETCH,
                   help=f"Candidate sentences to fetch before scoring (default {CANDIDATES_TO_FETCH}). "
                        "Scale roughly with --target; selection yield is only a few percent.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Prompts scored per forward pass (default {BATCH_SIZE}). Raise on bigger GPUs.")
    p.add_argument("--max-per-article", type=int, default=MAX_PER_ARTICLE,
                   help=f"Max prompts kept per article for diversity (default {MAX_PER_ARTICLE}).")
    p.add_argument("--mode", choices=["plain", "context"],
                   default="context" if PREPEND_PREVIOUS_SENTENCE else "plain",
                   help="plain = bare truncated sentence prefix; context = one complete "
                        "sentence prepended before the prefix. Sets the output filename "
                        "(_context suffix) and default slug tag.")
    p.add_argument("--slug-tag", type=str, default=None,
                   help="Suffix appended to every slug so plain/context runs can share one "
                        "graph pool. Default: 'pl' for plain, 'ctx' for context. Pass '' to "
                        "reproduce the original untagged convention.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed Python's RNG for reproducible candidate selection (default: unseeded).")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated CSV(s) with a 'slug' column whose slugs to skip, so the new "
                        "set is disjoint from an existing one (e.g. the original 200 prompts).")
    p.add_argument("--out-suffix", type=str, default="",
                   help="Append this to the output filename stems, e.g. '_800' -> prompts_wikipedia_800.csv, "
                        "so an existing set is not overwritten.")
    p.add_argument("--score-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
                   help="dtype for the scoring model (default bfloat16). float32 needs ~10GB just "
                        "for weights and OOMs on longer context-mode prompts on a 10GB GPU.")
    p.add_argument("--user-agent", type=str, default=WIKIPEDIA_USER_AGENT,
                   help="User-Agent sent to the Wikipedia API. A descriptive UA is REQUIRED "
                        "(the library default is rejected with HTTP 403). Default: "
                        f"{WIKIPEDIA_USER_AGENT!r}")
    return p.parse_args()


def main():
    global TARGET_N, CANDIDATES_TO_FETCH, BATCH_SIZE, MAX_PER_ARTICLE
    global OUT_FILE, PROMPTS_FILE, PREPEND_PREVIOUS_SENTENCE, SLUG_TAG
    args = parse_args()
    TARGET_N = args.target
    CANDIDATES_TO_FETCH = args.candidates
    BATCH_SIZE = args.batch_size
    MAX_PER_ARTICLE = args.max_per_article
    if args.seed is not None:
        random.seed(args.seed)

    # REQUIRED: Wikipedia rejects the library's default UA with HTTP 403, which
    # silently caps the candidate pool. Set a descriptive one before any fetch.
    wikipedia.set_user_agent(args.user_agent)

    # Mode controls prompt construction, output filename, and the default slug tag.
    PREPEND_PREVIOUS_SENTENCE = (args.mode == "context")
    SLUG_TAG = args.slug_tag if args.slug_tag is not None else ("ctx" if PREPEND_PREVIOUS_SENTENCE else "pl")

    # Recompute output paths from the chosen mode (not the module-load default).
    prompts_dir = OUT_FILE.parent
    suffix = ("_context" if PREPEND_PREVIOUS_SENTENCE else "") + args.out_suffix
    OUT_FILE = prompts_dir / f"ground_truth_wikipedia{suffix}.csv"
    PROMPTS_FILE = prompts_dir / f"prompts_wikipedia{suffix}.csv"

    excluded = load_excluded_slugs([s.strip() for s in args.exclude.split(",")]) if args.exclude else set()

    print("=== Wikipedia prompt generator ===")
    print(f"Target: {TARGET_N} prompts | mode={args.mode} | slug_tag={SLUG_TAG!r} "
          f"| confidence window: [{MIN_CONF}, {MAX_CONF}]"
          + (f" | excluding {len(excluded)} existing slugs" if excluded else ""))

    # --- Step 1: collect candidates ---
    print(f"\n[1/3] Fetching ~{CANDIDATES_TO_FETCH} candidate prompts from Wikipedia...")
    candidates = fetch_candidates(CANDIDATES_TO_FETCH)
    print(f"  Collected {len(candidates)} candidates from Wikipedia.")

    if excluded:
        before = len(candidates)
        candidates = [c for c in candidates if c["slug"] not in excluded]
        print(f"  Dropped {before - len(candidates)} candidates already in the excluded set "
              f"({len(candidates)} remain).")

    # --- Step 2: score with Gemma ---
    # Load in bfloat16: float32 weights are ~10GB for Gemma-2-2b, which leaves no
    # room for activations on a 10GB card and OOMs on longer (context-mode) prompts.
    # bf16 is plenty for top-1 next-token scoring.
    score_dtype = getattr(torch, args.score_dtype)
    print(f"\n[2/3] Loading {MODEL_NAME} on {SCORE_DEVICE} ({args.score_dtype})...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=SCORE_DEVICE, dtype=score_dtype)
    model.eval()

    print(f"  Scoring {len(candidates)} prompts...")
    prompt_texts = [c["prompt"] for c in candidates]
    results = score_prompts(prompt_texts, model)

    for cand, result in zip(candidates, results):
        cand["top1_prob"] = result["top1_prob"]
        cand["top1_token"] = result["top1_token"]
        # The Wikipedia next-word is the ground truth; we keep the prompt only
        # if the model also picked it (see model_correct check below).
        cand["correct_answer"] = cand["expected_answer"]
        cand["model_correct"] = (
            _normalize_word(result["top1_token"]) == cand["expected_answer"]
        )

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

    # Words that syntactically force an article/preposition/pronoun next
    _FUNCTION_FORCING_ENDINGS = frozenset({
        "and", "or", "but", "of", "in", "to", "for", "with", "on", "at",
        "from", "by", "about", "into", "through", "is", "was", "are", "were",
        "a", "an", "the", "that", "which", "as", "than", "not", "also",
        "both", "either", "neither", "between", "among",
    })

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
        # Reject prompts whose last word syntactically forces a function word next
        last_word = prompt.rstrip().rsplit(" ", 1)[-1].lower().strip(".,;:")
        if last_word in _FUNCTION_FORCING_ENDINGS:
            return False
        return True

    n_correct = sum(1 for c in candidates if c["model_correct"])
    n_in_window = sum(1 for c in candidates
                      if c["model_correct"] and MIN_CONF <= c["top1_prob"] <= MAX_CONF)
    interesting = [
        c for c in candidates
        if c["model_correct"]                       # (2) model's top-1 == Wikipedia next word
        and MIN_CONF <= c["top1_prob"] <= MAX_CONF  # (3) probability ≤ 0.8
        and _is_clean_token(c["correct_answer"])    # (1) not a stopword + other quality filters
        and _is_clean_prompt(c["prompt"])
    ]
    print(f"  {len(candidates)} candidates -> {n_correct} model_correct "
          f"-> {n_in_window} in [{MIN_CONF}, {MAX_CONF}] -> {len(interesting)} after token/prompt cleanup.")

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
            f"  WARNING: only found {len(selected)} prompts. "
            "Consider raising MAX_CONF, raising MAX_PER_ARTICLE, or fetching more candidates."
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
