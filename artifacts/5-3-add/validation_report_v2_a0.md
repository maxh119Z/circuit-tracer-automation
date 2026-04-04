# Validation Report

**Prompt:** <bos>5 + 3 =
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:29:06Z  |  **Runs:** 5
**Auto Attribution Coverage:** 46.4%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 85.5% ± 8.4% | 92.8% ± 4.4% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 85.0% ± 6.2% | 84.0% ± 7.8% | 0.0% ± 0.0% | 50% |
| *Random M1* | 0.0% ± 0.0% | — | — | 10% |
| *Random M2* | 50.0% ± 4.2% | — | — | 50% |

*Easy: 65 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 86.3% ± 0.0% | 10% | 175 |
| **D2** Description → pick correct snippets (5-in-10) | 90.5% ± 0.0% | 50% | 175 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| plus sign | 100.0% | 0.0% | 5 | 10 |
| start-of-sequence | 100.0% | 0.0% | 5 | 20 |
| say a digit | 72.0% | 10.2% | 5 | 25 |
| numeric result | 70.0% | 12.2% | 5 | 10 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| plus sign | 96.0% | 4.0% | 5 | 5 |
| numeric result | 92.0% | 4.9% | 5 | 5 |
| start-of-sequence | 84.0% | 7.5% | 5 | 5 |
| say a digit | 68.0% | 10.2% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 92.8% ± 4.4% | 84.0% ± 7.8% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:29:06Z · desc=`v2` · group=`a0`*
