# Validation Report

**Prompt:** <bos>The girl that the teacher sees
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:04:54Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 22.7% ± 2.8% | 53.6% ± 4.7% | 24.7% | 5/5 |
| **human** | 42.5% ± 10.0% | 56.7% ± 4.3% | 2.2% | 6/6 |
| **ours-no-reconciliation** | 79.8% ± 10.1% | 74.7% ± 3.8% | 25.4% | 6/6 |
| **ours-full** | 90.4% ± 5.5% | 76.8% ± 6.7% | 24.7% | 5/5 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 68.8% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 81.8% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| say see | 100.0% | 0.0% | 15 |
| teacher | 100.0% | 0.0% | 25 |
| vision verbs | 96.0% | 4.0% | 25 |
| girl | 84.0% | 7.5% | 25 |
| interpretation verbs | 72.0% | 8.0% | 25 |

---

*Generated 2026-05-06T03:04:54Z · desc=`v2` · group=`a2` · min_group_size=2*
