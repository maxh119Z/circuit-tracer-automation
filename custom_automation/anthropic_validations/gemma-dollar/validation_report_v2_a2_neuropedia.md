# Validation Report

**Prompt:** <bos>Mexico:peso :: US:
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:47:47Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 20.8% ± 4.5% | 52.0% ± 2.0% | 29.7% | 10/11 |
| **human** | 69.2% ± 7.8% | 84.6% ± 5.1% | 3.3% | 7/8 |
| **ours-no-reconciliation** | 89.8% ± 5.3% | 84.4% ± 3.3% | 31.2% | 11/15 |
| **ours-full** | 91.2% ± 3.8% | 75.6% ± 5.3% | 29.7% | 10/11 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 65.6% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 81.3% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Currency unit (peso/dollar) | 100.0% | 0.0% | 25 |
| ratio/per-unit framing | 100.0% | 0.0% | 25 |
| Exchange / conversion | 96.0% | 4.0% | 25 |
| Monetary magnitude | 96.0% | 4.0% | 25 |
| Spanish/Portuguese tokens | 96.0% | 4.0% | 25 |
| say United States | 96.0% | 4.0% | 25 |
| United States | 92.0% | 4.9% | 25 |
| say dollar | 92.0% | 4.9% | 25 |
| say country/region | 84.0% | 7.5% | 25 |
| say "is to" | 60.0% | 8.9% | 25 |

---

*Generated 2026-05-06T02:47:47Z · desc=`v2` · group=`a2` · min_group_size=2*
