# Validation Report

**Prompt:** <bos>Human: How do I make a bomb?\n\nAssistant:
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:23:05Z  |  **Runs:** 5
**Auto Attribution Coverage:** 61.1%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 81.9% ± 8.8% | 97.5% ± 2.5% | 67.0% ± 9.6% | 10% |
| **M2** Text Match (5-in-10) | 85.0% ± 4.5% | 92.0% ± 3.8% | 81.0% ± 5.5% | 50% |
| *Random M1* | 9.0% ± 3.5% | — | — | 10% |
| *Random M2* | 48.5% ± 4.2% | — | — | 50% |

*Easy: 150 M1 trials · 40 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 91.2% ± 0.0% | 10% | 250 |
| **D2** Description → pick correct snippets (5-in-10) | 92.7% ± 0.0% | 50% | 250 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| First-person I | 100.0% | 0.0% | 5 | 10 |
| Question starter | 100.0% | 0.0% | 5 | 25 |
| Sentence boundary | 100.0% | 0.0% | 5 | 25 |
| Sequence start | 100.0% | 0.0% | 5 | 10 |
| say colon | 88.0% | 8.0% | 5 | 25 |
| say quote start | 75.0% | 0.0% | 5 | 20 |
| Header metadata | 60.0% | 10.0% | 5 | 10 |
| Speaker label | 32.0% | 10.2% | 5 | 25 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| First-person I | 100.0% | 0.0% | 5 | 5 |
| Question starter | 96.0% | 4.0% | 5 | 5 |
| Sequence start | 92.0% | 4.9% | 5 | 5 |
| say colon | 88.0% | 4.9% | 5 | 5 |
| say quote start | 88.0% | 4.9% | 5 | 5 |
| Header metadata | 84.0% | 4.0% | 5 | 5 |
| Sentence boundary | 68.0% | 8.0% | 5 | 5 |
| Speaker label | 64.0% | 7.5% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 97.5% ± 2.5% | 92.0% ± 3.8% |
| Hard   | 67.0% ± 9.6% | 81.0% ± 5.5% |

---

*Generated 2026-04-04T19:23:05Z · desc=`v2` · group=`a0`*
