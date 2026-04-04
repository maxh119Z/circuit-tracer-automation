# Validation Report

**Prompt:** <bos>The animal associated with the country whose flag has a maple leaf is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:18:49Z  |  **Runs:** 5
**Auto Attribution Coverage:** 26.1%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 81.7% ± 18.3% | 100.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 77.3% ± 11.8% | 93.3% ± 2.7% | 0.0% ± 0.0% | 50% |
| *Random M1* | 4.7% ± 0.3% | — | — | 10% |
| *Random M2* | 53.3% ± 3.5% | — | — | 50% |

*Easy: 65 M1 trials · 15 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 91.2% ± 0.0% | 10% | 250 |
| **D2** Description → pick correct snippets (5-in-10) | 90.7% ± 0.0% | 50% | 250 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| Canada | 100.0% | 0.0% | 5 | 20 |
| animal | 100.0% | 0.0% | 5 | 25 |
| say country name | 45.0% | 5.0% | 5 | 20 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| animal | 100.0% | 0.0% | 5 | 5 |
| Canada | 72.0% | 4.9% | 5 | 5 |
| say country name | 60.0% | 6.3% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 93.3% ± 2.7% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:18:49Z · desc=`v2` · group=`a0`*
