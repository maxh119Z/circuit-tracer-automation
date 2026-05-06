# Validation Report

**Prompt:** <bos>Fait: Michael Jordan joue au
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:41:46Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 19.2% ± 2.0% | 52.0% ± 4.0% | 39.2% | 8/10 |
| **human** | 72.5% ± 7.5% | 92.0% ± 2.5% | 2.5% | 6/6 |
| **ours-no-reconciliation** | 95.0% ± 2.7% | 77.5% ± 4.3% | 39.2% | 8/10 |
| **ours-full** | 94.5% ± 2.6% | 73.5% ± 4.5% | 39.2% | 8/10 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 73.2% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 81.9% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Basketball | 100.0% | 0.0% | 25 |
| Basketball team mention | 100.0% | 0.0% | 10 |
| Football | 100.0% | 0.0% | 10 |
| Playing/games | 100.0% | 0.0% | 25 |
| French | 96.0% | 4.0% | 25 |
| Sports | 92.0% | 4.9% | 25 |
| Person name | 88.0% | 4.9% | 25 |
| Say sport | 80.0% | 6.3% | 25 |

---

*Generated 2026-05-06T02:41:46Z · desc=`v2` · group=`a2` · min_group_size=2*
