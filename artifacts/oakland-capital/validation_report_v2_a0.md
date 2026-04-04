# Validation Report

**Prompt:** <bos>The capital of the state containing Oakland is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:13:32Z  |  **Runs:** 5
**Auto Attribution Coverage:** 55.0%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 86.7% ± 7.1% | 100.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 72.7% ± 8.8% | 79.3% ± 6.3% | 0.0% ± 0.0% | 50% |
| *Random M1* | 7.5% ± 3.1% | — | — | 10% |
| *Random M2* | 58.7% ± 2.7% | — | — | 50% |

*Easy: 100 M1 trials · 30 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 95.7% ± 0.0% | 10% | 235 |
| **D2** Description → pick correct snippets (5-in-10) | 86.2% ± 0.0% | 50% | 235 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| California | 100.0% | 0.0% | 5 | 10 |
| say Contra | 100.0% | 0.0% | 5 | 10 |
| say Sacramento | 100.0% | 0.0% | 5 | 10 |
| say a place name | 84.0% | 4.0% | 5 | 25 |
| Capital city | 80.0% | 5.0% | 5 | 20 |
| say a capital | 56.0% | 4.0% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| California | 92.0% | 4.9% | 5 | 5 |
| say Contra | 92.0% | 4.9% | 5 | 5 |
| Capital city | 80.0% | 0.0% | 5 | 5 |
| say a capital | 76.0% | 4.0% | 5 | 5 |
| say a place name | 60.0% | 8.9% | 5 | 5 |
| say Sacramento | 36.0% | 14.7% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 79.3% ± 6.3% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:13:32Z · desc=`v2` · group=`a0`*
