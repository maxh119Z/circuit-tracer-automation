# Validation Report

**Prompt:** <bos>The country containing the capital of the state containing Denver is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:19:55Z  |  **Runs:** 5
**Auto Attribution Coverage:** 54.4%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 84.0% ± 8.5% | 86.0% ± 4.8% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 78.0% ± 9.0% | 93.0% ± 4.7% | 0.0% ± 0.0% | 50% |
| *Random M1* | 7.5% ± 1.3% | — | — | 10% |
| *Random M2* | 54.0% ± 6.2% | — | — | 50% |

*Easy: 95 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 93.3% ± 0.0% | 10% | 225 |
| **D2** Description → pick correct snippets (5-in-10) | 92.0% ± 0.0% | 50% | 225 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| Country | 100.0% | 0.0% | 5 | 20 |
| Containment relation | 96.0% | 4.0% | 5 | 25 |
| say US state | 76.0% | 4.0% | 5 | 25 |
| say location | 64.0% | 4.0% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| Containment relation | 100.0% | 0.0% | 5 | 5 |
| Country | 80.0% | 20.0% | 5 | 5 |
| say US state | 76.0% | 4.0% | 5 | 5 |
| say location | 56.0% | 4.0% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 86.0% ± 4.8% | 93.0% ± 4.7% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:19:55Z · desc=`v2` · group=`a0`*
