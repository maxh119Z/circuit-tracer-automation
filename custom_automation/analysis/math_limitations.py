"""
math_limitations.py — Quantify how the grouping pipeline is limited on arithmetic.

Motivation
----------
Addition is *known* to be mechanistically rich (Anthropic's "Biology of an LLM"
shows Gemma-2-2b does parallel approximate-magnitude + lookup-table computation,
with modular/periodic features and explicit sum construction). So if our pipeline
turns those graphs into generic surface-level supernodes ("digit 1", "plus sign",
"say a number"), that is a concrete, demonstrable limitation: the grouping
abstraction is too coarse to express the mechanism that is actually there.

This script produces one markdown report with five analyses, each explained
inline in the report:

  1. Group-name taxonomy   — are supernodes surface-token buckets or real
                             mechanism? (LLM-classified)
  2. Interestingness gap   — judge math vs wikipedia graphs on the paper-grounded
                             criteria; math should sit near the floor. (LLM)
  3. Coherence ≠ usefulness— autointerp validation (M1/M2/D1/D2) on 20 graphs,
                             shown next to interestingness: vague groups can be
                             internally COHERENT yet mechanistically empty.
  4. Feature→group vagueness— does the group NAME throw away structure that is
                             present in its member feature descriptions? (LLM)
  5. By problem type        — does grouping ever represent a CARRY? do 3-digit /
                             multi-carry problems yield richer groups, or just
                             more "digit X" buckets?

Model parts use gpt-5.4 by default (override with --model / EXPLORE_JUDGE_MODEL).

Usage:
    OPENAI_API_KEY=sk-... python custom_automation/analysis/math_limitations.py
    # bound cost while iterating:
    ... --judge-sample 20 --vague-sample 30 --skip-validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# UTF-8 stdout for Windows consoles.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))
import explore_interesting_graphs as explore  # noqa: E402  (reuse criteria + judge)

PACKAGE_DIR = ANALYSIS_DIR.parent                 # custom_automation/
REPO_ROOT = PACKAGE_DIR.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MATH_DIR = REPO_ROOT / "test_graphs_math"
WIKI_DIR = REPO_ROOT / "test_graphs_wiki400_plain"
GROUND_TRUTH = REPO_ROOT / "prompts" / "ground_truth_math.csv"

OUT_MD = RESULTS_DIR / "math_limitations_report.md"

MODEL = os.environ.get("EXPLORE_JUDGE_MODEL", "gpt-5.4")

# ---------------------------------------------------------------------------
# Item-level result cache — every expensive LLM result is written to disk as it
# completes, so a crash (or running out of quota) never discards finished work,
# and a re-run only does what's missing. Caches are namespaced by model.
# ---------------------------------------------------------------------------

CACHE_DIR = RESULTS_DIR / "cache"
REFRESH = False  # set by --refresh to ignore existing caches


def _cache_path(part: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", MODEL)
    return CACHE_DIR / f"{part}.{safe_model}.json"


def load_cache(part: str) -> dict:
    if REFRESH:
        return {}
    p = _cache_path(part)
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(part: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(part).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(_cache_path(part))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _client():
    from openai import OpenAI
    return OpenAI()


def _chat(client, system: str, user: str, max_tokens: int = 4096) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(text: str):
    m = re.search(r"[\[{][\s\S]*[\]}]", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def graph_paths(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*.json") if p.name != "graph-metadata.json")


def supernode_entries(graph: dict) -> list[list]:
    """Raw [name, fid, fid, ...] entries, excluding embedding/output groups."""
    raw = graph.get("qParams", {}).get("supernodes", "[]")
    items = raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
    out = []
    for e in items:
        if isinstance(e, list) and e and not str(e[0]).startswith(("Emb:", "Output:")):
            out.append(e)
    return out


def id_to_clerp(graph: dict) -> dict[str, str]:
    return {str(n.get("node_id")): (n.get("clerp") or "") for n in graph.get("nodes", [])}


def load_ground_truth() -> dict[str, dict]:
    """slug -> {operand_digits, num_carries, category, correct_answer}."""
    gt: dict[str, dict] = {}
    if not GROUND_TRUTH.exists():
        return gt
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            note = r.get("notes", "")
            md = dict(re.findall(r"(\w+)=([\w]+)", note))
            gt[r["slug"]] = {
                "operand_digits": int(md.get("operand_digits", 0) or 0),
                "num_carries": int(md.get("num_carries", 0) or 0),
                "category": md.get("category", "?"),
                "correct_answer": r.get("correct_answer", ""),
            }
    return gt


# ===========================================================================
# Part 1 — Group-name taxonomy
# ===========================================================================

CLASSIFY_SYSTEM = (
    "You are a mechanistic-interpretability researcher classifying the NAMES of "
    "supernode groups produced by an automated pipeline for addition prompts "
    "(e.g. '13+27='). Put each name in exactly one bucket:\n"
    "  surface_token  = names which literal token/character a feature reads or "
    "writes (e.g. 'digit 1', 'plus sign', 'equals sign', 'Arabic numerals', "
    "'digit tokens', 'numeric context').\n"
    "  say_number     = output-promotion of a number/number-word with no specific "
    "computation (e.g. 'say number', 'say digit', 'say small digit', 'say 3').\n"
    "  mechanism      = an actual arithmetic COMPUTATION step: carrying, magnitude/"
    "approximation, a specific sum or addend lookup, place-value addition, modular "
    "structure (e.g. 'carry the tens', 'sum near 40', 'add units', 'lookup 6+7', "
    "'magnitude estimate').\n"
    "  other          = anything that fits none of the above.\n"
    "Be strict: a name only counts as 'mechanism' if it refers to how the addition "
    "is computed, not merely to a number or token being present."
)


def classify_names(names: list[str], client) -> dict[str, str]:
    """LLM-classify unique names into the four buckets, batched. Cached per name."""
    cache = load_cache("name_classification")            # name -> category
    todo = [n for n in names if n not in cache]
    if not todo:
        print(f"  all {len(names)} names cached — skipping classification")
        return {n: cache.get(n, "other") for n in names}
    print(f"  {len(names) - len(todo)} cached, classifying {len(todo)} new names")
    BATCH = 40
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        numbered = "\n".join(f"{j}. {n}" for j, n in enumerate(chunk))
        user = (
            f"Classify each name. Reply with a JSON object mapping the index "
            f"(as a string) to one of: surface_token, say_number, mechanism, other.\n\n{numbered}"
        )
        parsed = _extract_json(_chat(client, CLASSIFY_SYSTEM, user, max_tokens=2048)) or {}
        for j, n in enumerate(chunk):
            cat = str(parsed.get(str(j), "other")).strip()
            cache[n] = cat if cat in ("surface_token", "say_number", "mechanism", "other") else "other"
        save_cache("name_classification", cache)         # incremental — survives a crash
        print(f"  classified {min(i + BATCH, len(todo))}/{len(todo)} names", flush=True)
    return {n: cache.get(n, "other") for n in names}


def rule_classify(name: str) -> str:
    """Keyword fallback when no API key is available."""
    n = name.lower()
    if any(k in n for k in ("carry", "magnitude", "approx", "lookup", "sum of", "addend",
                            "place value", "tens place", "units place", "modular")):
        return "mechanism"
    if n.startswith("say"):
        return "say_number"
    if any(k in n for k in ("digit", "plus sign", "equals", "numeral", "number", "numeric",
                            "token", "arabic", "sign")):
        return "surface_token"
    return "other"


def collect_math(paths: list[Path]):
    """One streaming pass over math graphs collecting only the small data each
    part needs, so we never hold the big graph dicts in memory.

    Returns: (name_counts, per_graph_names{slug:[names]}, vague_pool[(slug,name,[clerps])]).
    """
    name_counts: Counter = Counter()
    per_graph_names: dict[str, list[str]] = {}
    vague_pool: list[tuple[str, str, list[str]]] = []
    for i, p in enumerate(paths, 1):
        g = explore.load_graph(p)
        if not g:
            continue
        slug = g.get("metadata", {}).get("slug") or p.stem
        names = explore.get_supernode_names(g)
        per_graph_names[slug] = names
        for nm in names:
            name_counts[nm] += 1
        id2c = id_to_clerp(g)
        for e in supernode_entries(g):
            nm = str(e[0])
            clerps = [id2c.get(str(x), "") for x in e[1:] if id2c.get(str(x))]
            if len(clerps) >= 2:
                vague_pool.append((slug, nm, clerps))
        del g
        if i % 50 == 0:
            print(f"  scanned {i}/{len(paths)} math graphs", flush=True)
    return name_counts, per_graph_names, vague_pool


def part1_name_taxonomy(name_counts: Counter, n_graphs: int, use_llm: bool, client) -> tuple[str, dict[str, str]]:
    uniq = sorted(name_counts)

    if use_llm:
        print(f"[Part 1] LLM-classifying {len(uniq)} unique group names ({MODEL})…")
        name_cat = classify_names(uniq, client)
        method = f"gpt model `{MODEL}`"
    else:
        name_cat = {n: rule_classify(n) for n in uniq}
        method = "keyword rules (no API key — run with OPENAI_API_KEY for LLM classification)"

    # Aggregate by instances (weighted by how often each name appears).
    inst = Counter()
    uniq_by_cat = Counter()
    for n, c in name_cat.items():
        inst[c] += name_counts[n]
        uniq_by_cat[c] += 1
    total_inst = sum(inst.values()) or 1

    # Near-duplicate detection: names that normalize to the same string.
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    norm_groups = defaultdict(set)
    for n in uniq:
        norm_groups[norm(n)].add(n)
    dup_clusters = {k: v for k, v in norm_groups.items() if len(v) > 1}

    L = [
        "## 1. Group-name taxonomy — surface tokens vs. mechanism",
        "",
        "**What this measures.** Every supernode the pipeline produced on the math "
        "graphs is classified by what its *name* refers to: a literal token "
        "(`surface_token`), generic number output (`say_number`), an actual "
        "arithmetic computation step (`mechanism`), or `other`. **Why it shows a "
        "limitation:** addition has real internal mechanism (carrying, magnitude "
        "estimation, sum lookup); if almost no groups are named for any of that, the "
        "grouping is only re-describing the input/output tokens, not the computation.",
        "",
        f"Classification method: {method}.  "
        f"Graphs: {n_graphs} · group instances: {total_inst} · unique names: {len(uniq)}.",
        "",
        "| Bucket | Group instances | Share | Unique names |",
        "|:--|--:|--:|--:|",
    ]
    for c in ("surface_token", "say_number", "mechanism", "other"):
        L.append(f"| {c} | {inst[c]} | {inst[c]/total_inst:.0%} | {uniq_by_cat[c]} |")
    L += [
        "",
        f"**Mechanism groups: {inst['mechanism']} / {total_inst} "
        f"({inst['mechanism']/total_inst:.0%}).** "
        + ("None of the named supernodes describe how the sum is computed."
           if inst["mechanism"] == 0 else
           "Examples of the rare mechanism-named groups: "
           + ", ".join(f"`{n}`" for n, c in name_cat.items() if c == "mechanism")[:400]),
        "",
        "Most common supernode names (all surface/say, by design of the pipeline):",
        "",
        "| count | name | bucket |",
        "|--:|:--|:--|",
    ]
    for n, cnt in name_counts.most_common(15):
        L.append(f"| {cnt} | {n} | {name_cat.get(n, '?')} |")
    L += [
        "",
        f"**Naming instability.** {len(uniq)} distinct names for {n_graphs} near-identical "
        f"addition prompts, including {len(dup_clusters)} clusters of case/punctuation "
        "near-duplicates that the pipeline never canonicalizes. Examples: "
        + "; ".join("/".join(sorted(v)) for v in list(dup_clusters.values())[:6]),
        "",
    ]
    return "\n".join(L), name_cat


# ===========================================================================
# Part 2 — Interestingness contrast (math vs wikipedia)
# ===========================================================================

def judge_sample(paths: list[Path], client, min_conf: float, cache_part: str) -> tuple[dict[str, float], list[dict]]:
    """Judge a set of graph PATHS (loaded one at a time), cached per graph.

    Returns (means, rows): `means` is the mean per-criterion score (+ '_n'); each
    row has slug, prompt, predicted, the per-criterion `scores`, the standout
    criterion + reason, and `total` (sum of criteria, used to rank "most
    interesting"). Cached by filename stem, so cached graphs are not even reloaded.
    """
    cache = load_cache(cache_part)                       # stem -> row
    rows: list[dict] = []
    for p in paths:
        stem = p.stem
        if stem in cache:
            rows.append(cache[stem])
            continue
        g = explore.load_graph(p)
        if not g:
            continue
        slug = g.get("metadata", {}).get("slug") or stem
        m = explore.compute_metrics(g, influence_threshold=0.1)
        del g
        if m["model_confidence"] < min_conf:
            continue
        verdict = explore.call_llm_judge(
            m["prompt"], m["predicted"], m.get("correct_answer", ""),
            m["model_confidence"], m["supernode_names"], m["top_clerps"],
            m["num_supernode_groups"], client,
        )
        scores = {key: verdict[key]["score"] for key, _, _ in explore.LLM_CRITERIA}
        top_key, top_label = max(
            ((k, lbl) for k, lbl, _ in explore.LLM_CRITERIA), key=lambda kl: scores[kl[0]]
        )
        row = {
            "slug": slug,
            "prompt": m["prompt"],
            "predicted": m["predicted"],
            "scores": scores,
            "total": sum(scores.values()),
            "top_label": top_label,
            "top_score": scores[top_key],
            "top_reason": verdict[top_key]["reason"],
            "supernodes": " | ".join(m["supernode_names"][:6]),
        }
        cache[stem] = row
        save_cache(cache_part, cache)                    # incremental — survives a crash
        rows.append(row)
        print(f"  judged {len(rows)} graphs", end="\r", flush=True)
    print()
    n = len(rows)
    means = {key: (sum(r["scores"][key] for r in rows) / n if n else 0.0)
             for key, _, _ in explore.LLM_CRITERIA} | {"_n": n}
    return means, rows


def part2_interestingness(math_paths, wiki_paths, client, sample: int, min_conf: float) -> str:
    import random
    rng = random.Random(0)
    msamp = rng.sample(math_paths, min(sample, len(math_paths)))
    print(f"[Part 2] Judging {len(msamp)} math graphs ({MODEL})…")
    math_scores, math_rows = judge_sample(msamp, client, min_conf, "interest_math")

    wiki_scores = None
    if wiki_paths:
        wsamp = rng.sample(wiki_paths, min(sample, len(wiki_paths)))
        print(f"[Part 2] Judging {len(wsamp)} wikipedia graphs for contrast…")
        wiki_scores, _ = judge_sample(wsamp, client, min_conf, "interest_wiki")

    n_crit = len(explore.LLM_CRITERIA)
    L = [
        "## 2. Interestingness gap — math vs. Wikipedia",
        "",
        "**What this measures.** The same paper-grounded judge (criteria distilled "
        "from Anthropic's attribution-graph examples) scores a sample of math graphs "
        "and, for contrast, a sample of Wikipedia graphs, 1–10 per criterion. **Why "
        "it shows a limitation:** if math scores near the floor — especially on "
        "`hidden_or_multihop`, `abstraction_beyond_tokens`, and `unexpected_mechanism` "
        "— the pipeline is surfacing nothing a "
        "researcher would find notable, even though the addition mechanism itself is "
        "interesting. The Wikipedia column is the reference for what 'normal' scores "
        "look like.",
        "",
        f"Judge model: `{MODEL}` · math n={math_scores['_n']}"
        + (f" · wikipedia n={wiki_scores['_n']}" if wiki_scores else " · wikipedia: skipped"),
        "",
        "### Average scores (math graphs)",
        "",
        "| Criterion | Math mean | Wikipedia mean | Δ (wiki−math) |",
        "|:--|--:|--:|--:|",
    ]
    for key, label, _ in explore.LLM_CRITERIA:
        mv = math_scores[key]
        if wiki_scores:
            wv = wiki_scores[key]
            L.append(f"| {label} | {mv:.1f} | {wv:.1f} | {wv-mv:+.1f} |")
        else:
            L.append(f"| {label} | {mv:.1f} | — | — |")
    overall_math = sum(math_scores[k] for k, _, _ in explore.LLM_CRITERIA) / n_crit
    L.append(f"| **overall (mean of criteria)** | **{overall_math:.1f}** | "
             + (f"**{sum(wiki_scores[k] for k,_,_ in explore.LLM_CRITERIA)/n_crit:.1f}**"
                if wiki_scores else "—") + " | |")

    # Top-5 most interesting math graphs (by summed criteria score).
    top = sorted(math_rows, key=lambda r: -r["total"])[:5]
    L += [
        "",
        f"### Top 5 most interesting math graphs (of {math_scores['_n']} judged)",
        "",
        f"Ranked by total interestingness (sum of {n_crit} criteria, max {n_crit*10}). "
        "Even the most interesting math graphs score low in absolute terms — that's the point.",
        "",
        "| rank | total | prompt | predicted | standout criterion | why |",
        "|--:|--:|:--|:--|:--|:--|",
    ]
    for i, r in enumerate(top, 1):
        why = (r["top_reason"] or "")[:140].replace("|", "/")
        prompt = r["prompt"].replace("<bos>", "").strip()
        L.append(f"| {i} | {r['total']}/{n_crit*10} | `{prompt}` | {r['predicted']} "
                 f"| {r['top_label']} ({r['top_score']}) | {why} |")
    L += ["", ""]
    return "\n".join(L)


# ===========================================================================
# Part 4 — Feature→group vagueness
# ===========================================================================

VAGUE_SYSTEM = (
    "You are a mechanistic-interpretability researcher. You are given a supernode "
    "GROUP NAME from an addition-prompt attribution graph, plus the descriptions of "
    "the individual features inside that group. Rate, 1-10, how much SPECIFIC, "
    "computation-relevant structure the group NAME throws away relative to what the "
    "member descriptions actually contain.\n"
    "  1-2  = the name captures the members well; little is lost.\n"
    "  3-4  = minor detail lost.\n"
    "  5-6  = the members show specific structure (particular numeric ranges, "
    "operand roles, positions, relations) that the generic name flattens.\n"
    "  7-8  = the name is a generic bucket while members encode clearly different, "
    "computation-relevant behavior.\n"
    "  9-10 = the name is near-meaningless given how specific/varied the members are.\n"
    'Reply as JSON: {"vagueness": <1-10>, "lost": "<the specific structure the name '
    'drops, or \'none\'>"}.'
)


def part4_vagueness(vague_pool: list, client, sample: int) -> str:
    import random
    rng = random.Random(1)
    pool = list(vague_pool)
    rng.shuffle(pool)
    pool = pool[:sample]
    print(f"[Part 4] Scoring vagueness of {len(pool)} groups ({MODEL})…")

    cache = load_cache("vagueness")                      # content-hash -> record
    scored = []
    for i, (slug, name, clerps) in enumerate(pool, 1):
        key = hashlib.sha1(("|".join([slug, name] + clerps)).encode("utf-8")).hexdigest()
        if key in cache:
            rec = cache[key]
        else:
            members = "\n".join(f"  - {c[:160]}" for c in clerps[:8])
            user = f'Group name: "{name}"\n\nMember feature descriptions:\n{members}'
            parsed = _extract_json(_chat(client, VAGUE_SYSTEM, user, max_tokens=512)) or {}
            try:
                v = max(1, min(10, int(parsed.get("vagueness", 0))))
            except (TypeError, ValueError):
                v = 0
            rec = {"v": v, "name": name, "lost": str(parsed.get("lost", "")).strip(), "slug": slug}
            cache[key] = rec
            save_cache("vagueness", cache)               # incremental — survives a crash
        scored.append((rec["v"], rec["name"], rec["lost"], rec["slug"]))
        print(f"  scored {i}/{len(pool)}", end="\r", flush=True)
    print()
    vals = [v for v, *_ in scored if v]
    mean_v = sum(vals) / len(vals) if vals else 0.0
    worst = sorted(scored, key=lambda x: -x[0])[:8]

    L = [
        "## 4. Feature→group vagueness — does the NAME lose what the features encode?",
        "",
        "**What this measures.** For a sample of groups, the model is shown the group "
        "name and its member feature descriptions, and rates 1–10 how much specific, "
        "computation-relevant structure the *name* discards relative to the *members*. "
        "**Why it shows a limitation:** this is the crux of 'groups too vague to "
        "capture the behavior' — if individual features encode specific structure "
        "(particular numeric ranges, operand roles, positions) but get bucketed under "
        "a flat label like 'digit 1' or 'say number', the abstraction step is where "
        "the interesting behavior is lost, *not* the features themselves.",
        "",
        f"Judge model: `{MODEL}` · groups scored: {len(vals)} · "
        f"**mean vagueness: {mean_v:.1f}/10**.",
        "",
        "Most vague groups (name vs. what the members actually encode):",
        "",
        "| vagueness | group name | structure the name drops |",
        "|--:|:--|:--|",
    ]
    for v, name, lost, slug in worst:
        L.append(f"| {v} | `{name}` | {lost[:160].replace('|','/')} |")
    L += ["", ""]
    return "\n".join(L)


# ===========================================================================
# Part 5 — By problem type (carries / digits)
# ===========================================================================

def part5_by_type(per_graph_names: dict[str, list[str]], gt: dict, name_cat: dict[str, str]) -> str:
    rows = []  # (slug, digits, carries, n_groups, n_surface, n_say, n_mech, has_carry_group)
    for slug, names in per_graph_names.items():
        info = gt.get(slug)
        if not info:
            continue
        cats = [name_cat.get(n, "other") for n in names]
        has_carry = any("carry" in n.lower() for n in names)
        rows.append((slug, info["operand_digits"], info["num_carries"], len(names),
                     cats.count("surface_token"), cats.count("say_number"),
                     cats.count("mechanism"), has_carry))

    def agg(subset):
        if not subset:
            return None
        ng = [r[3] for r in subset]
        mech = sum(r[6] for r in subset)
        carry_graphs = sum(1 for r in subset if r[7])
        tot = sum(ng) or 1
        return {
            "n": len(subset),
            "avg_groups": sum(ng) / len(subset),
            "pct_surface": sum(r[4] for r in subset) / tot,
            "pct_say": sum(r[5] for r in subset) / tot,
            "mech": mech,
            "carry_graphs": carry_graphs,
        }

    L = [
        "## 5. By problem type — does grouping ever represent a carry?",
        "",
        "**What this measures.** Group statistics broken down by operand size "
        "(2- vs 3-digit) and by number of column carries. **Why it shows a "
        "limitation:** carrying is the central sub-computation of multi-digit "
        "addition. If harder problems (more carries, more digits) produce only *more* "
        "surface 'digit X' buckets — and essentially zero groups that name a carry — "
        "then the grouping does not scale with the difficulty of the underlying "
        "computation; it just tiles more token groups.",
        "",
        "By operand size:",
        "",
        "| operand digits | graphs | avg #groups | % surface | % say-number | mechanism groups | graphs w/ a 'carry' group |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for d in (2, 3):
        a = agg([r for r in rows if r[1] == d])
        if a:
            L.append(f"| {d}-digit | {a['n']} | {a['avg_groups']:.1f} | {a['pct_surface']:.0%} "
                     f"| {a['pct_say']:.0%} | {a['mech']} | {a['carry_graphs']} |")
    L += [
        "",
        "By number of carries:",
        "",
        "| carries | graphs | avg #groups | % surface | % say-number | mechanism groups | graphs w/ a 'carry' group |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for c in sorted({r[2] for r in rows}):
        a = agg([r for r in rows if r[2] == c])
        if a:
            L.append(f"| {c} | {a['n']} | {a['avg_groups']:.1f} | {a['pct_surface']:.0%} "
                     f"| {a['pct_say']:.0%} | {a['mech']} | {a['carry_graphs']} |")
    total_carry_graphs = sum(1 for r in rows if r[7])
    L += [
        "",
        f"**Across all {len(rows)} graphs, {total_carry_graphs} contain a group whose "
        f"name mentions a carry.** Harder problems add token buckets, not mechanism.",
        "",
    ]
    return "\n".join(L)


# ===========================================================================
# Part 3 — Validation on 20 graphs (coherence vs usefulness)
# ===========================================================================

def select_validation_slugs(gt: dict, available: set, n_each: int = 10) -> list[str]:
    """10 two-digit + 10 three-digit, spread across carry counts for coverage."""
    chosen = []
    for d in (2, 3):
        cand = sorted((s for s, i in gt.items() if i["operand_digits"] == d and s in available),
                      key=lambda s: (gt[s]["num_carries"], s))
        if not cand:
            continue
        step = max(1, len(cand) // n_each)
        picked = cand[::step][:n_each]
        chosen.extend(picked)
    return chosen


def _macro(rep, *path):
    cur = rep
    for k in path:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur.get("mean_accuracy", 0.0) if isinstance(cur, dict) else 0.0


def report_is_good(rep: dict) -> bool:
    """A validation report is real only if all four macro scores are > 0.

    validate_groups.py swallows per-call API errors and scores them 0, so a run
    that hit a quota/rate-limit 'completes' with all-zero scores. Those are
    garbage and must be re-run, not averaged in.
    """
    m1 = _macro(rep, "auto", "easy", "method1", "macro_avg")
    m2 = _macro(rep, "auto", "easy", "method2", "macro_avg")
    d1 = _macro(rep, "description_accuracy", "macro_avg")
    d2 = _macro(rep, "description_snippet_accuracy", "macro_avg")
    return all(v > 0.0 for v in (m1, m2, d1, d2))


def run_validation(slugs: list[str], force: bool = False) -> dict[str, dict]:
    """Run validate_groups.py per slug (gpt-5-mini autointerp graders) and read reports.

    Skips slugs whose existing report is already valid (so re-runs after a quota
    top-up only redo the failed ones), and drops all-zero reports as failures.
    """
    results: dict[str, dict] = {}
    for i, slug in enumerate(slugs, 1):
        report = PACKAGE_DIR / "artifacts" / slug / "validation_report_v2_a2.json"
        if report.exists() and not force:
            try:
                cached = json.load(open(report, encoding="utf-8"))
            except Exception:
                cached = None
            if cached and report_is_good(cached):
                print(f"[Part 3] ({i}/{len(slugs)}) {slug}: cached valid report — skipping")
                results[slug] = cached
                continue
        env = os.environ.copy()
        env["CURRENT_SLUG"] = slug
        env["GRAPH_DIR"] = "test_graphs_math"
        env.setdefault("DESCRIPTION_VARIANT", "v2")
        env.setdefault("GROUPING_VARIANT", "a2")
        print(f"[Part 3] ({i}/{len(slugs)}) validating {slug} …", flush=True)
        proc = subprocess.run(
            [sys.executable, str(PACKAGE_DIR / "pipeline" / "validate_groups.py")],
            env=env, cwd=str(PACKAGE_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0 or not report.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"    FAILED ({tail[0][:120]})")
            continue
        with open(report, encoding="utf-8") as f:
            rep = json.load(f)
        if not report_is_good(rep):
            print("    WARNING: all-zero scores — likely API/quota failure; not counted.")
            continue
        results[slug] = rep
    return results


def part3_validation(reports: dict[str, dict], gt: dict) -> str:
    if not reports:
        return ("## 3. Coherence ≠ usefulness — validation on 20 graphs\n\n"
                "_Validation produced no reports (run with OPENAI_API_KEY, or it was "
                "skipped via --skip-validation)._\n")
    rows = []
    for slug, rep in reports.items():
        m1 = _macro(rep, "auto", "easy", "method1", "macro_avg")
        m2 = _macro(rep, "auto", "easy", "method2", "macro_avg")
        d1 = _macro(rep, "description_accuracy", "macro_avg")
        d2 = _macro(rep, "description_snippet_accuracy", "macro_avg")
        rows.append((slug, gt.get(slug, {}).get("operand_digits", "?"), m1, m2, d1, d2))

    def mean(i):
        vals = [r[i] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    L = [
        "## 3. Coherence ≠ usefulness — autointerp validation on 20 graphs",
        "",
        "**What this measures.** Standard autointerp validation on 20 math graphs "
        "(10 two-digit, 10 three-digit): M1 = can a grader pick a group's feature out "
        "of 10 from the group name (chance 10%); M2 = match 5-of-10 snippets to a "
        "group (chance 50%); D1/D2 = description quality. **Why it matters here:** a "
        "vague group like 'digit 1' is *easy* to identify, so it can score HIGH on "
        "coherence while carrying no mechanistic content. High M1/M2 next to the "
        "rock-bottom interestingness of Part 2 is the point: the groups are "
        "internally consistent but mechanistically empty — coherence is not the same "
        "as capturing the behavior.",
        "",
        f"Validation grader: `gpt-5-mini` (the project's standard autointerp grader). "
        f"Graphs validated: {len(rows)}.",
        "",
        "| metric | mean | chance |",
        "|:--|--:|--:|",
        f"| M1 feature identification | {mean(2):.0%} | 10% |",
        f"| M2 snippet matching | {mean(3):.0%} | 50% |",
        f"| D1 description accuracy | {mean(4):.0%} | 10% |",
        f"| D2 description snippets | {mean(5):.0%} | 50% |",
        "",
        "Per-graph M1 / M2:",
        "",
        "| slug | digits | M1 | M2 |",
        "|:--|:--|--:|--:|",
    ]
    for slug, d, m1, m2, _, _ in rows:
        L.append(f"| {slug} | {d} | {m1:.0%} | {m2:.0%} |")
    L += ["", ""]
    return "\n".join(L)


# ===========================================================================
# CSV emission — alongside the markdown report, dump the per-item data as CSVs
# for plotting/analysis. Reads the on-disk caches directly (so it works even on
# --refresh runs and partial runs), plus name counts + validation from memory.
# ===========================================================================

def _read_cache_file(part: str) -> dict:
    p = _cache_path(part)
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write_csvs(name_counts: Counter, name_cat: dict, reports: dict, gt: dict) -> None:
    crit = [k for k, _, _ in explore.LLM_CRITERIA]
    written: list[str] = []

    for side in ("math", "wiki"):
        d = _read_cache_file(f"interest_{side}")
        if not d:
            continue
        path = RESULTS_DIR / f"math_interestingness_{side}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slug", "prompt", "predicted", "total"] + crit
                       + ["top_label", "top_score", "top_reason", "supernodes"])
            for r in sorted(d.values(), key=lambda r: -r.get("total", 0)):
                sc = r.get("scores", {})
                w.writerow([r.get("slug"), r.get("prompt"), r.get("predicted"), r.get("total")]
                           + [sc.get(k, "") for k in crit]
                           + [r.get("top_label"), r.get("top_score"), r.get("top_reason"), r.get("supernodes")])
        written.append(f"interestingness_{side}")

    # Prefer the cached LLM classification (so a --no-llm run can't overwrite the
    # good gpt-5.4 categories with keyword-rule ones).
    cat_lookup = _read_cache_file("name_classification") or name_cat
    if cat_lookup:
        path = RESULTS_DIR / "math_name_classification.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "category", "count"])
            for n, c in name_counts.most_common():
                w.writerow([n, cat_lookup.get(n, "?"), c])
        written.append("name_classification")

    vg = _read_cache_file("vagueness")
    if vg:
        path = RESULTS_DIR / "math_vagueness.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slug", "group_name", "vagueness", "lost"])
            for rec in sorted(vg.values(), key=lambda r: -r.get("v", 0)):
                w.writerow([rec.get("slug"), rec.get("name"), rec.get("v"), rec.get("lost")])
        written.append("vagueness")

    if reports:
        path = RESULTS_DIR / "math_validation.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slug", "operand_digits", "M1", "M2", "D1", "D2"])
            for slug, rep in sorted(reports.items()):
                w.writerow([slug, gt.get(slug, {}).get("operand_digits", ""),
                            _macro(rep, "auto", "easy", "method1", "macro_avg"),
                            _macro(rep, "auto", "easy", "method2", "macro_avg"),
                            _macro(rep, "description_accuracy", "macro_avg"),
                            _macro(rep, "description_snippet_accuracy", "macro_avg")])
        written.append("validation")

    if written:
        print("CSVs written → " + ", ".join(f"math_{w}.csv" for w in written))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    global MODEL, REFRESH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=f"LLM for model-based parts (default {MODEL}).")
    ap.add_argument("--judge-sample", type=int, default=40, help="Graphs per side for the interestingness contrast.")
    ap.add_argument("--vague-sample", type=int, default=60, help="Groups to score for feature→group vagueness.")
    ap.add_argument("--val-each", type=int, default=10, help="Graphs per digit-size for validation (×2 total).")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--force-validation", action="store_true",
                    help="Re-run validation even for slugs that already have a valid report.")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore all caches and recompute every LLM part from scratch.")
    ap.add_argument("--no-llm", action="store_true", help="Skip all model-based parts (free; rule-based Part 1).")
    args = ap.parse_args()

    if args.model:
        MODEL = args.model
    REFRESH = args.refresh

    if not MATH_DIR.exists():
        sys.exit(f"Math graph dir not found: {MATH_DIR}")
    math_paths = graph_paths(MATH_DIR)
    wiki_paths = graph_paths(WIKI_DIR) if WIKI_DIR.exists() else []
    gt = load_ground_truth()

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    use_llm = has_key and not args.no_llm
    client = _client() if use_llm else None
    if not use_llm:
        print("NOTE: running without LLM (no OPENAI_API_KEY or --no-llm). "
              "Parts 1/5 use keyword rules; Parts 2/3/4 are skipped.\n")

    # One streaming pass collects the small data every part needs.
    print(f"Scanning {len(math_paths)} math graphs (streaming)…")
    name_counts, per_graph_names, vague_pool = collect_math(math_paths)
    n_math = len(per_graph_names)
    print(f"Done. {n_math} math graphs, {len(name_counts)} unique group names, "
          f"{len(vague_pool)} describable groups; {len(wiki_paths)} wiki graphs; "
          f"{len(gt)} ground-truth rows.\n")

    sections = []
    # Part 1 (needed before 5 — provides name classification)
    s1, name_cat = part1_name_taxonomy(name_counts, n_math, use_llm, client)
    sections.append(s1)
    # Part 2
    if use_llm:
        sections.append(part2_interestingness(math_paths, wiki_paths, client,
                                               args.judge_sample, args.min_confidence))
    else:
        sections.append("## 2. Interestingness gap — math vs. Wikipedia\n\n_Skipped (needs OPENAI_API_KEY)._\n")
    # Part 3
    reports: dict = {}
    if use_llm and not args.skip_validation:
        slugs = select_validation_slugs(gt, set(per_graph_names), args.val_each)
        reports = run_validation(slugs, force=args.force_validation)
        sections.append(part3_validation(reports, gt))
    else:
        sections.append("## 3. Coherence ≠ usefulness — validation on 20 graphs\n\n"
                        "_Skipped (--skip-validation or no OPENAI_API_KEY)._\n")
    # Part 4
    if use_llm:
        sections.append(part4_vagueness(vague_pool, client, args.vague_sample))
    else:
        sections.append("## 4. Feature→group vagueness\n\n_Skipped (needs OPENAI_API_KEY)._\n")
    # Part 5
    sections.append(part5_by_type(per_graph_names, gt, name_cat))

    header = [
        "# Math prompts — where the grouping pipeline is limited",
        "",
        "Addition is *known* to be mechanistically rich (Anthropic 2025: parallel "
        "approximate-magnitude + lookup-table computation). This report tests whether "
        "our pipeline's supernodes capture any of that, or collapse it into generic "
        "surface buckets. Five analyses, each explained in its section.",
        "",
        f"Math graphs: {n_math} · model for LLM parts: `{MODEL}`.",
        "",
        "---",
        "",
    ]
    OUT_MD.write_text("\n".join(header) + "\n\n---\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"\nReport written → {OUT_MD}")
    write_csvs(name_counts, name_cat, reports, gt)


if __name__ == "__main__":
    main()
