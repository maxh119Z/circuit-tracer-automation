# Validation Report

**Prompt:** <bos>The state to the West of the capital of the state containing Las Vegas
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:16:32Z  |  **Runs:** 5
**Auto Attribution Coverage:** 42.0%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 92.5% ± 7.5% | 100.0% ± 0.0% | 53.5% ± 7.0% | 10% |
| **M2** Text Match (5-in-10) | 87.0% ± 5.7% | 98.0% ± 2.0% | 91.0% ± 3.8% | 50% |
| *Random M1* | 2.0% ± 1.1% | — | — | 10% |
| *Random M2* | 57.0% ± 3.4% | — | — | 50% |

*Easy: 85 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 98.0% ± 0.0% | 10% | 250 |
| **D2** Description → pick correct snippets (5-in-10) | 92.8% ± 0.0% | 50% | 250 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| West | 100.0% | 0.0% | 5 | 25 |
| city name | 100.0% | 0.0% | 5 | 25 |
| say place after 'of' | 100.0% | 0.0% | 5 | 25 |
| U.S. state | 70.0% | 12.2% | 5 | 10 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| U.S. state | 96.0% | 4.0% | 5 | 5 |
| West | 96.0% | 4.0% | 5 | 5 |
| city name | 84.0% | 9.8% | 5 | 5 |
| say place after 'of' | 72.0% | 10.2% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 98.0% ± 2.0% |
| Hard   | 53.5% ± 7.0% | 91.0% ± 3.8% |

---

*Generated 2026-04-04T19:16:32Z · desc=`v2` · group=`a0`*
