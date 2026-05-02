"""
Print clean samples of the actual validation tasks per condition, so you can
eyeball whether the random baseline (or any other condition) is being given
unfair tasks.

For one or more slugs, replays the validator's own task-generation logic at the
same seed (RANDOM_SEED, run_idx=0) and dumps a small number of M1 and M2 tasks
per condition into a markdown file. The correct answer is marked in each task.

Usage:
    # Default: pick a few representative slugs.
    python custom_automation/analysis/inspect_validation_tasks.py

    # Specific slugs, more samples per condition:
    python custom_automation/analysis/inspect_validation_tasks.py \
        --slugs gemma-G,gemma-addition,gemma-dollar --samples 3

Outputs (in custom_automation/analysis/validation_inspection_results/):
    <slug>_inspection_min{N}.md   one file per (slug, min_group_size)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "neuronpedia_graphs"
sys.path.insert(0, str(PIPE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-use the validator's helpers so the tasks we print are byte-identical to
# what the LLM saw during validation (same RNG seed, same negative pools, etc.).
from validate_neuropedia_groups import (  # noqa: E402
    BASE_CONDITIONS,
    MAX_FEATURES_M1,
    N_NEG_FEATURES_M1,
    N_NEG_SNIPPETS_M2,
    N_POS_SNIPPETS_M2,
    RANDOM_SEED,
    _OURS_CAP_RE,
    build_group_index,
    build_medium_neg_pool,
    create_random_group_index,
    discover_cap_conditions,
    load_condition_groups,
    _format_snippet,
)
from config import DESCRIPTION_VARIANT, GROUPING_VARIANT  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RESULTS_DIR = Path(__file__).resolve().parent / "validation_inspection_results"


# ---------------------------------------------------------------------------
# Task-replay helpers (mirror generate_m1_tasks / generate_m2_tasks, but
# return RICH structures we can render — not opaque prompt strings).
# ---------------------------------------------------------------------------

def replay_m1_tasks(group_index, neg_pool, seed, min_group_size, max_groups, max_per_group):
    rng = random.Random(seed)
    tasks: list[dict] = []
    for group_name, pos_features in group_index.items():
        if len(pos_features) < min_group_size:
            continue
        if max_groups is not None and len(tasks) >= max_groups * max_per_group:
            break
        if not neg_pool:
            continue
        test = list(pos_features) if len(pos_features) <= MAX_FEATURES_M1 else rng.sample(pos_features, MAX_FEATURES_M1)
        local_taken = 0
        for pos in test:
            if local_taken >= max_per_group:
                break
            n_neg = min(len(neg_pool), N_NEG_FEATURES_M1)
            negs = rng.sample(neg_pool, n_neg)
            items = [(pos.get("generated_description", "<no desc>"), True, pos.get("id", "?"))]
            items += [(f.get("generated_description", "<no desc>"), False, f.get("id", "?")) for f in negs]
            rng.shuffle(items)
            tasks.append({
                "group_name": group_name,
                "items": items,
                "correct_idx": next(i for i, (_, is_pos, _) in enumerate(items) if is_pos) + 1,
                "n_items": len(items),
            })
            local_taken += 1
    return tasks


def replay_m2_tasks(group_index, neg_pool, seed, min_group_size, max_groups):
    rng = random.Random(seed)
    tasks: list[dict] = []
    for group_name, pos_features in group_index.items():
        if len(pos_features) < min_group_size:
            continue
        if max_groups is not None and len(tasks) >= max_groups:
            break
        all_pos_snippets = []
        for f in pos_features:
            for s in _get_snippets(f):
                all_pos_snippets.append((s, f.get("id", "?")))
        if len(all_pos_snippets) < N_POS_SNIPPETS_M2:
            continue
        pos_snips = rng.sample(all_pos_snippets, N_POS_SNIPPETS_M2)

        if neg_pool is not None:
            neg_pool_snips = []
            for f in neg_pool:
                for s in _get_snippets(f):
                    neg_pool_snips.append((s, f.get("id", "?")))
        else:
            neg_pool_snips = []
            for gname, feats in group_index.items():
                if gname == group_name:
                    continue
                for f in feats:
                    for s in _get_snippets(f):
                        neg_pool_snips.append((s, f.get("id", "?")))

        n_neg = min(len(neg_pool_snips), N_NEG_SNIPPETS_M2)
        if n_neg == 0:
            continue
        neg_snips = rng.sample(neg_pool_snips, n_neg)

        items = [(s, True, fid) for s, fid in pos_snips] + [(s, False, fid) for s, fid in neg_snips]
        rng.shuffle(items)
        tasks.append({
            "group_name": group_name,
            "items": items,
            "n_pos": len(pos_snips),
            "n_neg": len(neg_snips),
        })
    return tasks


def _get_snippets(feat: dict, n: int = 5) -> list[str]:
    return [
        _format_snippet(act)
        for act in feat.get("top_activations", [])[:n]
        if act.get("context", "").strip()
    ]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 280) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_m1(task: dict) -> str:
    lines = [f"**Group:** `{task['group_name']}`", ""]
    lines.append(f"_Pick the 1 description (out of {task['n_items']}) that belongs to this group._")
    lines.append("")
    for i, (desc, is_pos, fid) in enumerate(task["items"], 1):
        marker = "✓" if is_pos else " "
        lines.append(f"  {marker} **{i:>2}.** `{fid}` — {_truncate(desc)}")
    lines.append("")
    return "\n".join(lines)


def render_m2(task: dict) -> str:
    lines = [f"**Group:** `{task['group_name']}`", ""]
    lines.append(f"_Pick the {task['n_pos']} snippets (out of {task['n_pos']+task['n_neg']}) that belong to this group._")
    lines.append("")
    for i, (snip, is_pos, fid) in enumerate(task["items"], 1):
        marker = "✓" if is_pos else " "
        lines.append(f"  {marker} **{i:>2}.** `{fid}` — {_truncate(snip)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-slug driver
# ---------------------------------------------------------------------------

def all_conditions_for_slug(slug: str) -> list[str]:
    # Re-use validator's discovery, but rooted at this slug's artifacts dir.
    import config as _cfg
    _cfg.CURRENT_SLUG = slug  # noqa
    return list(BASE_CONDITIONS) + discover_cap_conditions()


def inspect_slug(slug: str, min_group_size: int, samples_per_condition: int) -> Path | None:
    # Hot-swap config.CURRENT_SLUG / paths so load_condition_groups sees this slug.
    import importlib
    import config
    config.CURRENT_SLUG = slug  # type: ignore
    config.GRAPH_FILE = config.REPO_ROOT / "test_graphs" / f"{slug}.json"
    config.ARTIFACTS_DIR = config.PACKAGE_DIR / "artifacts" / slug
    config.FEATURE_DESCRIPTIONS_FILE = config.ARTIFACTS_DIR / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"
    config.FEATURE_GROUPS_FILE       = config.ARTIFACTS_DIR / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    config.FEATURE_GROUPS_PRE3_FILE  = config.ARTIFACTS_DIR / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_pre3.json"
    config.MANUAL_GROUPS_FILE        = config.ARTIFACTS_DIR / "manual_groups.json"
    # Force the validator module to pick up the new paths.
    import validate_neuropedia_groups as v
    importlib.reload(v)

    desc_file = config.FEATURE_DESCRIPTIONS_FILE
    if not desc_file.exists():
        print(f"  [{slug}] no feature_descriptions file — skipping")
        return None
    with open(desc_file, encoding="utf-8") as f:
        features: list[dict] = json.load(f)

    conds = list(BASE_CONDITIONS) + v.discover_cap_conditions()
    if "ours-full" not in conds:
        print(f"  [{slug}] no ours-full file — skipping")
        return None

    out_lines: list[str] = []
    out_lines.append(f"# Validation task inspection — {slug} (min_group_size={min_group_size})")
    out_lines.append("")
    out_lines.append(f"Showing **up to {samples_per_condition}** M1 and M2 tasks per condition. "
                     "`✓` marks the positive (correct) feature/snippet. Tasks use the same "
                     f"RNG seed as the validator (run_idx=0, seed={RANDOM_SEED}), so what you "
                     "see is exactly what the LLM saw on the first of its 5 runs.")
    out_lines.append("")

    seed = RANDOM_SEED  # run_idx=0

    # Resolve the random condition specially: it uses ours-full's group structure
    # but features shuffled from the entire post-prune pool.
    base = v.load_condition_groups("ours-full", features)
    if base is None:
        return None
    full_groups_raw, full_index = base
    medium_neg_pool = v.build_medium_neg_pool(features, full_groups_raw)

    for cond in conds:
        if cond == "random":
            rng_for_random = random.Random(seed)
            random_index = v.create_random_group_index(features, full_index, rng_for_random)
            cond_index = random_index
            cond_neg_pool = medium_neg_pool
        else:
            loaded = v.load_condition_groups(cond, features)
            if loaded is None:
                out_lines.append(f"## Condition: `{cond}` — _skipped (no data)_\n")
                continue
            _, cond_index = loaded
            cond_neg_pool = medium_neg_pool

        m1_tasks = replay_m1_tasks(cond_index, cond_neg_pool, seed, min_group_size,
                                    max_groups=samples_per_condition, max_per_group=1)
        m2_tasks = replay_m2_tasks(cond_index, cond_neg_pool, seed, min_group_size,
                                    max_groups=samples_per_condition)

        out_lines.append(f"## Condition: `{cond}`")
        out_lines.append("")
        out_lines.append(f"_{len(cond_index)} groups in this condition; "
                         f"{sum(1 for v in cond_index.values() if len(v) >= min_group_size)} "
                         f"large enough to score (>= {min_group_size} members)._")
        out_lines.append("")

        out_lines.append("### M1 — feature identification (1-in-10 description match)")
        out_lines.append("")
        if not m1_tasks:
            out_lines.append("_No M1 tasks at this min_group_size._\n")
        else:
            for t in m1_tasks[:samples_per_condition]:
                out_lines.append(render_m1(t))
                out_lines.append("---")
                out_lines.append("")

        out_lines.append("### M2 — text snippet matching (5-in-10)")
        out_lines.append("")
        if not m2_tasks:
            out_lines.append("_No M2 tasks at this min_group_size._\n")
        else:
            for t in m2_tasks[:samples_per_condition]:
                out_lines.append(render_m2(t))
                out_lines.append("---")
                out_lines.append("")

    out_path = RESULTS_DIR / f"{slug}_inspection_min{min_group_size}.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  wrote {out_path}")
    return out_path


def parse_str_list(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slugs", default="gemma-G,gemma-addition,gemma-dollar",
                        help="Comma-separated slugs to inspect. Default: gemma-G,gemma-addition,gemma-dollar.")
    parser.add_argument("--min-size", type=int, default=3,
                        help="min_group_size threshold to use. Default: 3.")
    parser.add_argument("--samples", type=int, default=2,
                        help="Sample tasks per condition per method. Default: 2.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for slug in parse_str_list(args.slugs):
        inspect_slug(slug, args.min_size, args.samples)


if __name__ == "__main__":
    main()