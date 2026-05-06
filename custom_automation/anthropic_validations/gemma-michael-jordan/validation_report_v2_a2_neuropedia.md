# Validation Report

**Prompt:** <bos>Fact: Michael Jordan plays the sport of
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T03:17:18Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 29.5% ± 4.2% | 52.0% ± 3.1% | 35.4% | 9/10 |
| **human** | 86.9% ± 6.4% | 90.7% ± 1.7% | 2.8% | 6/7 |
| **ours-no-reconciliation** | 94.4% ± 2.1% | 82.8% ± 4.1% | 35.5% | 10/10 |
| **ours-full** | 97.3% ± 1.1% | 84.9% ± 3.8% | 35.4% | 9/10 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 75.6% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 80.9% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| Olympics | 100.0% | 0.0% | 10 |
| ball sports | 100.0% | 0.0% | 25 |
| combat martial arts | 100.0% | 0.0% | 10 |
| say play/ball | 100.0% | 0.0% | 25 |
| say sport/activity | 100.0% | 0.0% | 15 |
| Basketball | 96.0% | 4.0% | 25 |
| say sport name | 96.0% | 4.0% | 25 |
| proper names | 92.0% | 4.9% | 25 |
| sports | 92.0% | 4.9% | 25 |

---

*Generated 2026-05-06T03:17:18Z · desc=`v2` · group=`a2` · min_group_size=2*
