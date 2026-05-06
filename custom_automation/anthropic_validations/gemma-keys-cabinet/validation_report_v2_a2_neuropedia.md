# Validation Report

**Prompt:** <bos>The keys on the cabinet
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:11:53Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 15.7% ± 2.4% | 55.3% ± 2.4% | 24.7% | 6/6 |
| **human** | 69.4% ± 22.2% | 70.7% ± 16.4% | 1.3% | 3/4 |
| **ours-no-reconciliation** | 89.3% ± 4.3% | 77.3% ± 6.6% | 25.5% | 6/6 |
| **ours-full** | 80.7% ± 8.2% | 79.3% ± 6.8% | 24.7% | 6/6 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 74.0% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 82.8% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Open/unlock doors | 100.0% | 0.0% | 25 |
| say door | 100.0% | 0.0% | 10 |
| Keys (physical) | 88.0% | 8.0% | 25 |
| plural nouns | 80.0% | 8.9% | 25 |
| key (general) | 68.0% | 12.0% | 25 |
| cabinet | 48.0% | 10.2% | 25 |

---

*Generated 2026-05-06T03:11:53Z · desc=`v2` · group=`a2` · min_group_size=2*
