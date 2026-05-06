# Validation Report

**Prompt:** <bos>The girls that the teacher sees
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:06:57Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 15.2% ± 2.3% | 53.6% ± 2.0% | 22.8% | 5/5 |
| **human** | 44.7% ± 9.2% | 68.0% ± 8.5% | 1.3% | 5/5 |
| **ours-no-reconciliation** | 91.4% ± 3.3% | 66.9% ± 7.0% | 23.2% | 7/7 |
| **ours-full** | 90.4% ± 4.5% | 79.2% ± 5.0% | 22.8% | 5/5 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 74.4% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 84.4% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| female nouns | 100.0% | 0.0% | 25 |
| say see | 100.0% | 0.0% | 10 |
| teacher | 92.0% | 4.9% | 25 |
| children (plural) | 80.0% | 6.3% | 25 |
| see (perception) | 80.0% | 10.9% | 25 |

---

*Generated 2026-05-06T03:06:57Z · desc=`v2` · group=`a2` · min_group_size=2*
