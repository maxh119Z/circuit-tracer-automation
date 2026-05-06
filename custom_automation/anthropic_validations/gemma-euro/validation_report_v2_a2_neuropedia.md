# Validation Report

**Prompt:** <bos>Mexico:peso :: Europe:
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:00:28Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 20.0% ± 2.8% | 51.1% ± 1.8% | 25.7% | 13/14 |
| **human** | 84.8% ± 2.9% | 88.8% ± 3.4% | 4.8% | 5/5 |
| **ours-no-reconciliation** | 90.6% ± 5.4% | 80.9% ± 2.9% | 30.4% | 14/15 |
| **ours-full** | 95.2% ± 1.6% | 80.6% ± 2.3% | 25.7% | 13/14 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 62.8% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 82.8% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Currency unit | 100.0% | 0.0% | 25 |
| Europe / European | 100.0% | 0.0% | 25 |
| Monetary/finance terms | 100.0% | 0.0% | 10 |
| Relations/links | 100.0% | 0.0% | 10 |
| say European | 100.0% | 0.0% | 15 |
| say exchange | 100.0% | 0.0% | 25 |
| say location | 100.0% | 0.0% | 10 |
| Country mention | 92.0% | 4.9% | 25 |
| say country/region | 92.0% | 4.9% | 25 |
| say place name | 92.0% | 4.9% | 25 |
| Country/demonym | 90.0% | 10.0% | 10 |
| Spanish/Portuguese tokens | 88.0% | 4.9% | 25 |
| say coin | 84.0% | 7.5% | 25 |

---

*Generated 2026-05-06T03:00:28Z · desc=`v2` · group=`a2` · min_group_size=2*
