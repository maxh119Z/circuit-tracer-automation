# Validation Report

**Prompt:** <bos>grass: green sky: blue corn: yellow carrot: orange strawberry:
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:25:49Z  |  **Runs:** 5
**Auto Attribution Coverage:** 64.6%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 96.0% ± 4.0% | 100.0% ± 0.0% | 0.0% ± 0.0% | 10% |
| **M2** Text Match (5-in-10) | 81.0% ± 10.2% | 89.0% ± 7.5% | 0.0% ± 0.0% | 50% |
| *Random M1* | 11.0% ± 3.9% | — | — | 10% |
| *Random M2* | 54.0% ± 3.5% | — | — | 50% |

*Easy: 80 M1 trials · 20 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 93.7% ± 0.0% | 10% | 205 |
| **D2** Description → pick correct snippets (5-in-10) | 87.5% ± 0.0% | 50% | 205 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| color adjectives | 100.0% | 0.0% | 5 | 20 |
| label colon | 100.0% | 0.0% | 5 | 20 |
| red | 100.0% | 0.0% | 5 | 15 |
| say a color | 84.0% | 7.5% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| label colon | 100.0% | 0.0% | 5 | 5 |
| color adjectives | 88.0% | 4.9% | 5 | 5 |
| red | 84.0% | 7.5% | 5 | 5 |
| say a color | 52.0% | 8.0% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 100.0% ± 0.0% | 89.0% ± 7.5% |
| Hard   | 0.0% ± 0.0% | 0.0% ± 0.0% |

---

*Generated 2026-04-04T19:25:49Z · desc=`v2` · group=`a0`*
