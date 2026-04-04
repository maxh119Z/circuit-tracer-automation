# Validation Report

**Prompt:** <bos>The country containing the capital of the state containing Las Vegas is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:21:07Z  |  **Runs:** 5
**Auto Attribution Coverage:** 58.2%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 95.8% ± 2.8% | 100.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 87.5% ± 3.4% | 86.5% ± 3.4% | 0.0% ± 0.0% | 50% |
| *Random M1* | 7.1% ± 2.9% | — | — | 10% |
| *Random M2* | 53.0% ± 2.9% | — | — | 50% |

*Easy: 110 M1 trials · 40 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 98.4% ± 0.0% | 10% | 245 |
| **D2** Description → pick correct snippets (5-in-10) | 91.6% ± 0.0% | 50% | 245 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| California | 100.0% | 0.0% | 5 | 10 |
| capital city | 100.0% | 0.0% | 5 | 10 |
| contain | 100.0% | 0.0% | 5 | 25 |
| country | 100.0% | 0.0% | 5 | 15 |
| say Nevada | 100.0% | 0.0% | 5 | 10 |
| say US state | 100.0% | 0.0% | 5 | 10 |
| say state | 86.7% | 8.2% | 5 | 15 |
| say place after 'of' | 80.0% | 8.2% | 5 | 15 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| country | 100.0% | 0.0% | 5 | 5 |
| capital city | 96.0% | 4.0% | 5 | 5 |
| contain | 96.0% | 4.0% | 5 | 5 |
| California | 88.0% | 8.0% | 5 | 5 |
| say Nevada | 88.0% | 12.0% | 5 | 5 |
| say US state | 80.0% | 6.3% | 5 | 5 |
| say place after 'of' | 80.0% | 6.3% | 5 | 5 |
| say state | 72.0% | 4.9% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 86.5% ± 3.4% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:21:07Z · desc=`v2` · group=`a0`*
