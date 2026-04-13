# Validation Report

**Prompt:** <bos>Fact: the capital of the state containing Dallas is
**Description variant:** `v2`  |  **Grouping variant:** `a2`
**Date:** 2026-04-12T22:13:33Z  |  **Runs:** 5
**Auto Attribution Coverage:** 68.5%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features

| | Easy | Medium | Chance |
|:--|-----:|-------:|-------:|
| **M1** Feature ID (1-in-10) | 59.2% ± 19.4% | 92.0% ± 8.0% | 10% |
| **M2** Text Match (5-in-10) | 75.2% ± 11.1% | 83.2% ± 7.3% | 50% |
| *Random M1* | 4.4% ± 2.7% | — | 10% |
| *Random M2* | 50.4% ± 3.5% | — | 50% |

*Easy: 80 M1 trials · 25 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 89.0% ± 0.0% | 10% | 210 |
| **D2** Description → pick correct snippets (5-in-10) | 84.1% ± 0.0% | 50% | 210 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| Texas | 100.0% | 0.0% | 5 | 25 |
| say place name | 96.0% | 4.0% | 5 | 25 |
| Major city names | 70.0% | 12.2% | 5 | 10 |
| Capital (economic) | 30.0% | 12.2% | 5 | 10 |
| Capital (general) | 0.0% | 0.0% | 5 | 10 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| Capital (general) | 96.0% | 4.0% | 5 | 5 |
| Capital (economic) | 92.0% | 4.9% | 5 | 5 |
| Texas | 92.0% | 4.9% | 5 | 5 |
| Major city names | 48.0% | 8.0% | 5 | 5 |
| say place name | 48.0% | 4.9% | 5 | 5 |

### Medium Difficulty (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 92.0% ± 8.0% | 83.2% ± 7.3% |

---

*Generated 2026-04-12T22:13:33Z · desc=`v2` · group=`a2`*
