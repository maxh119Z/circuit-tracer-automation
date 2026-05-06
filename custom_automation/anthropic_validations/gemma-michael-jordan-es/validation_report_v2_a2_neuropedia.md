# Validation Report

**Prompt:** <bos>Hecho: Michael Jordan juega al
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:22:44Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 19.5% ± 1.4% | 59.0% ± 2.8% | 37.5% | 8/9 |
| **human** | 71.6% ± 9.8% | 89.3% ± 3.0% | 2.0% | 6/6 |
| **ours-no-reconciliation** | 88.4% ± 4.7% | 76.8% ± 5.3% | 39.7% | 10/11 |
| **ours-full** | 94.0% ± 2.1% | 82.5% ± 4.9% | 37.5% | 8/9 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 68.0% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 82.6% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Basketball | 100.0% | 0.0% | 25 |
| Play/playing | 100.0% | 0.0% | 25 |
| Spanish-language context | 100.0% | 0.0% | 25 |
| Ball sports | 96.0% | 4.0% | 25 |
| say Spanish noun | 92.0% | 4.9% | 25 |
| say sport name | 92.0% | 4.9% | 25 |
| Person name | 88.0% | 4.9% | 25 |
| Sports topic | 84.0% | 4.0% | 25 |

---

*Generated 2026-05-06T03:22:44Z · desc=`v2` · group=`a2` · min_group_size=2*
