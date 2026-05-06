# Validation Report

**Prompt:** <bos>2 + 1 = 
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:36:40Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 22.7% ± 4.2% | 54.0% ± 3.5% | 39.1% | 6/6 |
| **human** | 34.8% ± 15.2% | 69.6% ± 5.1% | 3.3% | 5/5 |
| **ours-no-reconciliation** | 80.6% ± 6.1% | 66.9% ± 5.0% | 50.6% | 7/7 |
| **ours-full** | 89.3% ± 6.0% | 62.7% ± 6.3% | 39.1% | 6/6 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 49.6% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 76.8% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| plus sign | 100.0% | 0.0% | 25 |
| say 3 | 100.0% | 0.0% | 25 |
| say small number word | 100.0% | 0.0% | 20 |
| digit 2 | 92.0% | 4.9% | 25 |
| say digit | 80.0% | 8.9% | 25 |
| result value | 64.0% | 9.8% | 25 |

---

*Generated 2026-05-06T02:36:40Z · desc=`v2` · group=`a2` · min_group_size=2*
