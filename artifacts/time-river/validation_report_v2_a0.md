# Validation Report

**Prompt:** <bos>Time is like a
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:14:37Z  |  **Runs:** 5
**Auto Attribution Coverage:** 62.0%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 89.0% ± 9.7% | 99.0% ± 1.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 72.0% ± 12.1% | 78.0% ± 7.8% | 0.0% ± 0.0% | 50% |
| *Random M1* | 1.0% ± 1.0% | — | — | 10% |
| *Random M2* | 48.0% ± 1.6% | — | — | 50% |

*Easy: 75 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 86.2% ± 0.0% | 10% | 225 |
| **D2** Description → pick correct snippets (5-in-10) | 88.6% ± 0.0% | 50% | 225 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| say 'time' | 100.0% | 0.0% | 5 | 10 |
| say a noun phrase | 100.0% | 0.0% | 5 | 25 |
| say a simile | 96.0% | 4.0% | 5 | 25 |
| time concept | 60.0% | 6.7% | 5 | 15 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| time concept | 88.0% | 4.9% | 5 | 5 |
| say a simile | 84.0% | 9.8% | 5 | 5 |
| say 'time' | 80.0% | 6.3% | 5 | 5 |
| say a noun phrase | 36.0% | 4.0% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 99.0% ± 1.0% | 78.0% ± 7.8% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:14:37Z · desc=`v2` · group=`a0`*
