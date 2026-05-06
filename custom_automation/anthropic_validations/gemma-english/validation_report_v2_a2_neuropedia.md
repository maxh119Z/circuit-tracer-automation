# Validation Report

**Prompt:** <bos>Mexico:Spanish :: US:
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Min group size:** 2  |  **Difficulty:** medium
**Date:** 2026-05-06T02:54:15Z  |  **Runs:** 5

---

## Method Scores by Condition

M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). Scores are mean ± stderr across runs.

| Condition | M1 (medium) | M2 (medium) | Coverage | Groups |
|:--|---:|---:|---:|---:|
| **random** | 28.2% ± 4.0% | 56.0% ± 1.8% | 33.0% | 12/13 |
| **human** | 80.2% ± 7.5% | 77.0% ± 10.0% | 2.7% | 4/8 |
| **ours-no-reconciliation** | 89.9% ± 2.6% | 78.9% ± 3.2% | 36.1% | 15/18 |
| **ours-full** | 93.8% ± 2.1% | 81.7% ± 2.8% | 33.0% | 12/13 |

---

## Description Quality (D1 / D2)

Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. Slug-level (does not depend on grouping condition or min_group_size).

| Method | Macro accuracy |
|:--|---:|
| **D1** — feature evidence → description (1-in-10, chance 10%) | 61.2% ± 0.0% |
| **D2** — description → activating snippets (5-in-10, chance 50%) | 84.7% ± 0.0% |

---

## Per-Group Detail — ours-full / medium

| Group | M1 Accuracy | ±Stderr | Trials |
|:--|---:|---:|---:|
| nationality adjectives | 100.0% | 0.0% | 25 |
| say 'in' + language | 100.0% | 0.0% | 20 |
| say United States | 100.0% | 0.0% | 10 |
| say ethnicity | 100.0% | 0.0% | 25 |
| Country/place names | 96.0% | 4.0% | 25 |
| Language names | 96.0% | 4.0% | 25 |
| Spanish (language) | 96.0% | 4.0% | 25 |
| United States / America | 96.0% | 4.0% | 25 |
| say nationality adjective | 90.0% | 10.0% | 10 |
| say country | 88.0% | 4.9% | 25 |
| say non-US country | 88.0% | 8.0% | 25 |
| English varieties | 76.0% | 7.5% | 25 |

---

*Generated 2026-05-06T02:54:15Z · desc=`v2` · group=`a2` · min_group_size=2*
