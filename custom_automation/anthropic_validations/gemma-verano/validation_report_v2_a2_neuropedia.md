# Validation Report

**Prompt:** <bos>La estación después de la primavera se llama el
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:29:44Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 0.0% ± 0.0% | 0.0% ± 0.0% | 17.4% | 5/5 |
| **human** | 0.0% ± 0.0% | 0.0% ± 0.0% | 6.5% | 8/8 |
| **ours-no-reconciliation** | 0.0% ± 0.0% | 0.0% ± 0.0% | 41.4% | 8/8 |
| **ours-full** | 0.0% ± 0.0% | 0.0% ± 0.0% | 17.4% | 5/5 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 0.0% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 0.0% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| month names | 0.0% | 0.0% | 25 |
| say season | 0.0% | 0.0% | 25 |
| say summer | 0.0% | 0.0% | 25 |
| season names | 0.0% | 0.0% | 25 |
| time expressions | 0.0% | 0.0% | 25 |

---

*Generated 2026-05-06T03:29:44Z · desc=`v2` · group=`a2` · min_group_size=2*
