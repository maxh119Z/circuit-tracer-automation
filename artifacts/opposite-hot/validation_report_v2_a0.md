# Validation Report

**Prompt:** <bos>The opposite of "hot" is "
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:26:56Z  |  **Runs:** 5
**Auto Attribution Coverage:** 80.9%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 96.0% ± 4.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 70.4% ± 12.2% | 0.0% ± 0.0% | 0.0% ± 0.0% | 50% |
| *Random M1* | 7.2% ± 2.9% | — | — | 10% |
| *Random M2* | 46.4% ± 3.7% | — | — | 50% |

*Easy: 100 M1 trials · 25 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 91.8% ± 0.0% | 10% | 195 |
| **D2** Description → pick correct snippets (5-in-10) | 87.5% ± 0.0% | 50% | 195 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| formatting boundary | 100.0% | 0.0% | 5 | 25 |
| hot token | 100.0% | 0.0% | 5 | 25 |
| say a temperature word | 100.0% | 0.0% | 5 | 10 |
| suppress cold/cool | 100.0% | 0.0% | 5 | 15 |
| temperature words | 80.0% | 0.0% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| hot token | 100.0% | 0.0% | 5 | 5 |
| formatting boundary | 96.0% | 4.0% | 5 | 5 |
| temperature words | 64.0% | 4.0% | 5 | 5 |
| suppress cold/cool | 56.0% | 7.5% | 5 | 5 |
| say a temperature word | 36.0% | 7.5% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 0.0% ± 0.0% | 0.0% ± 0.0% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:26:56Z · desc=`v2` · group=`a0`*
