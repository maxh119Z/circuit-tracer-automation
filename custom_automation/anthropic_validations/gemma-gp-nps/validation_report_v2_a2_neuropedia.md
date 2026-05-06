# Validation Report

**Prompt:** <bos>The guitarist knew the song
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:09:39Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 18.1% ± 5.4% | 49.7% ± 3.8% | 36.5% | 7/8 |
| **human** | 41.3% ± 16.9% | 66.0% ± 5.4% | 3.2% | 6/6 |
| **ours-no-reconciliation** | 91.0% ± 2.9% | 72.0% ± 5.0% | 36.7% | 7/10 |
| **ours-full** | 90.7% ± 3.1% | 66.3% ± 5.4% | 36.5% | 7/8 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 76.0% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 80.5% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| say well | 100.0% | 0.0% | 20 |
| song / music | 100.0% | 0.0% | 25 |
| musical instruments | 96.0% | 4.0% | 25 |
| say would | 90.0% | 6.1% | 20 |
| degree adverbs | 85.0% | 6.1% | 20 |
| know / knew | 84.0% | 7.5% | 25 |
| content nouns | 80.0% | 8.9% | 25 |

---

*Generated 2026-05-06T03:09:39Z · desc=`v2` · group=`a2` · min_group_size=2*
