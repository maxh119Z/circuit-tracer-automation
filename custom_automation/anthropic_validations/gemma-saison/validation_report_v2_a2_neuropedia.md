# Validation Report

**Prompt:** <bos>La saison après le printemps s'apelle l'
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:28:45Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 22.1% ± 2.5% | 55.6% ± 1.7% | 37.1% | 11/11 |
| **human** | 70.9% ± 8.1% | 83.2% ± 2.9% | 2.6% | 5/10 |
| **ours-no-reconciliation** | 93.0% ± 2.2% | 78.9% ± 2.3% | 38.6% | 14/15 |
| **ours-full** | 93.9% ± 1.6% | 78.5% ± 2.7% | 37.1% | 11/11 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 33.2% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 71.7% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| autumn | 100.0% | 0.0% | 25 |
| say season | 100.0% | 0.0% | 25 |
| say summer | 100.0% | 0.0% | 15 |
| French elision l' | 96.0% | 4.0% | 25 |
| season names | 96.0% | 4.0% | 25 |
| demote summer | 93.3% | 6.7% | 15 |
| French articles | 92.0% | 4.9% | 25 |
| month names | 92.0% | 4.9% | 25 |
| say month | 92.0% | 4.9% | 25 |
| French language | 88.0% | 4.9% | 25 |
| say French | 84.0% | 7.5% | 25 |

---

*Generated 2026-05-06T03:28:45Z · desc=`v2` · group=`a2` · min_group_size=2*
