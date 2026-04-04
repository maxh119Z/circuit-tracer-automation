# Validation Report

**Prompt:** <bos>The country containing Toronto is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:17:47Z  |  **Runs:** 5
**Auto Attribution Coverage:** 65.9%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 88.8% ± 5.6% | 96.8% ± 3.2% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 79.2% ± 9.9% | 82.4% ± 8.1% | 0.0% ± 0.0% | 50% |
| *Random M1* | 7.9% ± 1.4% | — | — | 10% |
| *Random M2* | 52.0% ± 4.7% | — | — | 50% |

*Easy: 90 M1 trials · 25 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 89.7% ± 0.0% | 10% | 165 |
| **D2** Description → pick correct snippets (5-in-10) | 85.0% ± 0.0% | 50% | 165 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| Canadian city names | 100.0% | 0.0% | 5 | 15 |
| say alternate name | 100.0% | 0.0% | 5 | 15 |
| say a location | 92.0% | 4.9% | 5 | 25 |
| country | 80.0% | 12.2% | 5 | 10 |
| Canada | 72.0% | 4.9% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| country | 100.0% | 0.0% | 5 | 5 |
| say alternate name | 92.0% | 4.9% | 5 | 5 |
| Canada | 88.0% | 4.9% | 5 | 5 |
| Canadian city names | 72.0% | 4.9% | 5 | 5 |
| say a location | 44.0% | 7.5% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 96.8% ± 3.2% | 82.4% ± 8.1% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:17:47Z · desc=`v2` · group=`a0`*
