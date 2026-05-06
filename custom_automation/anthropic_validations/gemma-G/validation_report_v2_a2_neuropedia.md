# Validation Report

**Prompt:** <bos>The International Advanced Security Group (IAS
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:27:02Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 12.7% ± 3.7% | 53.3% ± 2.7% | 15.6% | 3/3 |
| **human** | 60.3% ± 11.0% | 72.4% ± 4.4% | 9.4% | 9/10 |
| **ours-no-reconciliation** | 89.4% ± 3.9% | 72.0% ± 4.4% | 63.6% | 12/17 |
| **ours-full** | 78.7% ± 14.1% | 74.7% ± 13.1% | 15.6% | 3/3 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 79.2% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 78.3% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Security | 100.0% | 0.0% | 10 |
| group | 84.0% | 7.5% | 25 |
| Proper names & titles | 52.0% | 4.9% | 25 |

---

*Generated 2026-05-06T02:27:02Z · desc=`v2` · group=`a2` · min_group_size=2*
