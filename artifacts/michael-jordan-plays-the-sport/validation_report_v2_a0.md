# Validation Report

**Prompt:** <bos>Michael Jordan plays the sport of
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:22:02Z  |  **Runs:** 5
**Auto Attribution Coverage:** 71.2%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 96.0% ± 4.0% | 96.0% ± 4.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 72.0% ± 12.1% | 78.4% ± 6.8% | 0.0% ± 0.0% | 50% |
| *Random M1* | 3.2% ± 2.0% | — | — | 10% |
| *Random M2* | 41.6% ± 6.1% | — | — | 50% |

*Easy: 105 M1 trials · 25 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 96.5% ± 0.0% | 10% | 230 |
| **D2** Description → pick correct snippets (5-in-10) | 89.5% ± 0.0% | 50% | 230 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| Basketball | 100.0% | 0.0% | 5 | 25 |
| person names | 100.0% | 0.0% | 5 | 25 |
| say a sport | 100.0% | 0.0% | 5 | 25 |
| sports topic | 100.0% | 0.0% | 5 | 20 |
| play sports | 80.0% | 12.2% | 5 | 10 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| person names | 100.0% | 0.0% | 5 | 5 |
| Basketball | 96.0% | 4.0% | 5 | 5 |
| say a sport | 72.0% | 8.0% | 5 | 5 |
| sports topic | 56.0% | 7.5% | 5 | 5 |
| play sports | 36.0% | 7.5% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 96.0% ± 4.0% | 78.4% ± 6.8% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:22:02Z · desc=`v2` · group=`a0`*
