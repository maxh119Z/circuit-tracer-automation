# Validation Report

**Prompt:** <bos>The opposite of "small" is "
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:28:08Z  |  **Runs:** 5
**Auto Attribution Coverage:** 58.4%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 77.0% ± 9.8% | 100.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 77.0% ± 10.2% | 93.0% ± 4.7% | 0.0% ± 0.0% | 50% |
| *Random M1* | 10.3% ± 4.2% | — | — | 10% |
| *Random M2* | 58.0% ± 5.3% | — | — | 50% |

*Easy: 90 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 91.9% ± 0.0% | 10% | 210 |
| **D2** Description → pick correct snippets (5-in-10) | 88.8% ± 0.0% | 50% | 210 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| small | 100.0% | 0.0% | 5 | 25 |
| say opposite | 80.0% | 8.2% | 5 | 15 |
| size | 76.0% | 4.0% | 5 | 25 |
| say large | 52.0% | 4.9% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| say opposite | 100.0% | 0.0% | 5 | 5 |
| small | 88.0% | 4.9% | 5 | 5 |
| size | 64.0% | 4.0% | 5 | 5 |
| say large | 56.0% | 4.0% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 93.0% ± 4.7% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:28:08Z · desc=`v2` · group=`a0`*
