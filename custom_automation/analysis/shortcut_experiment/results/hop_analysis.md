# Intermediate Hop Analysis

Variants analysed: a2  
Prompts analysed: 10  
Total graph runs: 10

## Results by Hop Count

| Hop type | Cases | Correct | In top-5 | Mean score | Hop found | All int. hops found | Partial | None found |
|----------|------:|-------:|--------:|----------:|----------:|--------------------:|--------:|-----------:|
| **Overall** | 10 | 5 (50%) | 9 (90%) | 0.76 | 1 (10%) | 1 (10%) | 0 (0%) | 9 (90%) |
| 1-hop | 4 | 2 (50%) | 4 (100%) | 0.85 | 0 (0%) | 0 (0%) | 0 (0%) | 4 (100%) |
| 2-hop | 5 | 3 (60%) | 5 (100%) | 0.84 | 1 (20%) | 1 (20%) | 0 (0%) | 4 (80%) |
| 3-hop | 1 | 0 (0%) | 0 (0%) | 0.00 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (100%) |

## Correctness × Hop Presence

| | Hop found | No hop found |
|---|---|---|
| **Model correct** | 1 | 4 |
| **Model wrong** | 0 | 5 |

> When model is **wrong**: hop found in 0.0% of cases  
> When model is **correct**: hop found in 20.0% of cases

## Correctness × Hop Presence by Hop Count

| Hop type | n | Correct + Hop found | Correct + Hop missing | Wrong + Hop found | Wrong + Hop missing |
|----------|--:|--------------------:|----------------------:|------------------:|--------------------:|
| **Overall** | 10 | 1 | 4 | 0 | 5 |
| 1-hop | 4 | 0 | 2 | 0 | 2 |
| 2-hop | 5 | 1 | 2 | 0 | 2 |
| 3-hop | 1 | 0 | 0 | 0 | 1 |

## Per-Prompt Results — variant a2

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| mquake-2012-skip | skip-missed (from mquake-2012) | The Beatles, Brian Epstein | `Asia` (13.1%) | 2 | 0.8 | ✗ (0 feat, groups: —) ✗The Beatles, Brian Epstein | 0 (0.0%) | 0.0000 |
| mquake-2749-skip | skip-missed (from mquake-2749) | The Beatles, Brian Epstein | `London` (13.5%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗The Beatles, Brian Epstein | 0 (0.0%) | 0.0000 |
| mquake-2847-skip | skip-missed (from mquake-2847) | England | `The` (5.3%) | — | 0.0 | ✗ (0 feat, groups: —) ✗England | 0 (0.0%) | 0.0000 |
| mquake-2965-skip | skip-missed (from mquake-2965) | Messerschmitt, Willy Messerschmitt, Germany | `Africa` (11.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) ✗Messerschmitt, Willy Messerschmitt, Germany | 0 (0.0%) | 0.0000 |
| mquake-2471-skip | skip-missed (from mquake-2471) | Manchester United F.C., Ole Gunnar Solskjær, Norway | `Africa` (26.7%) | 3 | 0.6 | ✗ (0 feat, groups: —) ✗Manchester United F.C., Ole Gunnar Solskjær, Norway | 0 (0.0%) | 0.0000 |
| mquake-2185-skip | skip-missed (from mquake-2185) | midfielder, association football, England | `London` (13.7%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗midfielder, association football, England | 0 (0.0%) | 0.0000 |
| mquake-2431-skip | skip-missed (from mquake-2431) | association football manager, association football, England | `London` (9.0%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗association football manager, association football, England | 0 (0.0%) | 0.0000 |
| mquake-2557-skip | skip-missed (from mquake-2557) | Terry Nation, Doctor Who | `London` (16.2%) | 1 | 1.0 | ✓ (4 feat, groups: Doctor Who) ✓Terry Nation, Doctor Who | 4 (0.2%) | 0.3508 |
| mquake-2170-skip | skip-missed (from mquake-2170) | A. A. Milne, Christopher Robin Milne | `The` (7.2%) | 4 | 0.4 | ✗ (0 feat, groups: —) ✗A. A. Milne, Christopher Robin Milne | 0 (0.0%) | 0.0000 |
| mquake-2718-skip | skip-missed (from mquake-2718) | A. A. Milne, Christopher Robin Milne | `English` (17.4%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗A. A. Milne, Christopher Robin Milne | 0 (0.0%) | 0.0000 |

