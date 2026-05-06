# Validation Report

**Prompt:** <bos>3 + 5 = 
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:32:11Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 29.5% ± 4.0% | 56.5% ± 3.2% | 44.5% | 8/9 |
| **human** | 45.1% ± 14.1% | 71.2% ± 8.5% | 3.6% | 5/5 |
| **ours-no-reconciliation** | 95.6% ± 3.1% | 67.6% ± 2.8% | 46.7% | 10/11 |
| **ours-full** | 94.0% ± 4.3% | 67.5% ± 1.9% | 44.5% | 8/9 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 55.6% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 78.8% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Digit 3 | 100.0% | 0.0% | 25 |
| Digit 5 | 100.0% | 0.0% | 25 |
| Digit 8 | 100.0% | 0.0% | 25 |
| say Arabic numerals | 100.0% | 0.0% | 10 |
| Digits and numerals | 96.0% | 4.0% | 25 |
| Plus sign | 96.0% | 4.0% | 25 |
| say small number words | 96.0% | 4.0% | 25 |
| say number | 64.0% | 7.5% | 25 |

---

*Generated 2026-05-06T02:32:11Z · desc=`v2` · group=`a2` · min_group_size=2*
