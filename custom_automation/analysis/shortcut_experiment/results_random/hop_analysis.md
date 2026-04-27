# Intermediate Hop Analysis

Variants analysed: a2  
Prompts analysed: 6  
Total graph runs: 6

## Results by Hop Count

| Hop type | Cases | Correct | In top-5 | Mean score | Hop found | All int. hops found | Partial | None found |
|----------|------:|-------:|--------:|----------:|----------:|--------------------:|--------:|-----------:|
| **Overall** | 6 | 3 (50%) | 5 (83%) | 0.63 | 1 (17%) | 1 (17%) | 0 (0%) | 5 (83%) |
| 2-hop | 5 | 2 (40%) | 4 (80%) | 0.56 | 1 (20%) | 1 (20%) | 0 (0%) | 4 (80%) |
| 3-hop | 1 | 1 (100%) | 1 (100%) | 1.00 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (100%) |

## Correctness × Hop Presence

| | Hop found | No hop found |
|---|---|---|
| **Model correct** | 0 | 3 |
| **Model wrong** | 1 | 2 |

> When model is **wrong**: hop found in 33.3% of cases  
> When model is **correct**: hop found in 0.0% of cases

## Correctness × Hop Presence by Hop Count

| Hop type | n | Correct + Hop found | Correct + Hop missing | Wrong + Hop found | Wrong + Hop missing |
|----------|--:|--------------------:|----------------------:|------------------:|--------------------:|
| **Overall** | 6 | 0 | 3 | 1 | 2 |
| 2-hop | 5 | 0 | 2 | 1 | 2 |
| 3-hop | 1 | 0 | 1 | 0 | 0 |

## Per-Prompt Results — variant a2

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| mquake-2012-random | skip-random-2 (from mquake-2012) | The Beatles, United Kingdom | `Africa` (7.4%) | 4 | 0.4 | ✓ (1 feat, groups: say United States) ✓The Beatles, United Kingdom | 1 (0.0%) | 0.3495 |
| mquake-2749-random | skip-random-2 (from mquake-2749) | The Beatles, United Kingdom | `London` (10.0%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗The Beatles, United Kingdom | 0 (0.0%) | 0.0000 |
| mquake-2847-random | skip-random-1 (from mquake-2847) | midfielder | `London` (7.2%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗midfielder | 0 (0.0%) | 0.0000 |
| mquake-2557-random | skip-random-2 (from mquake-2557) | Terry Nation, Doctor Who | `London` (23.7%) | 1 | 1.0 | ✗ (0 feat, groups: —) ✗Terry Nation, Doctor Who | 0 (0.0%) | 0.0000 |
| mquake-2170-random | skip-random-2 (from mquake-2170) | Christopher Robin Milne, United Kingdom | `Africa` (11.2%) | 4 | 0.4 | ✗ (0 feat, groups: —) ✗Christopher Robin Milne, United Kingdom | 0 (0.0%) | 0.0000 |
| mquake-2718-random | skip-random-2 (from mquake-2718) | United Kingdom, A. A. Milne | `What` (13.3%) | — | 0.0 | ✗ (0 feat, groups: —) ✗United Kingdom, A. A. Milne | 0 (0.0%) | 0.0000 |

## Spotlight: Model Wrong but Intermediate Hop Present

These are the most interpretability-interesting cases — the model encoded the intermediate concept but still predicted incorrectly.

### mquake-2012-random (a2)
- Prompt type: skip-random-2 (from mquake-2012)
- Intermediate concept: **The Beatles, United Kingdom**
- Correct answer: Europe
- Model predicted: `Africa` (7.4%)
- Hop features: 1 (0.0% of transcoder nodes)
- Hop in supernode groups: ['say United States']
- Mean influence of hop features: 0.3495
- **Concepts found:** The Beatles, United Kingdom
- **Concepts missed:** none
  - `The Beatles, United Kingdom`: matched via terms: ['united']

