# Validation Report

**Prompt:** <bos>At first, Sally hated school. But over time she changed her mind. Now she is
**Description variant:** `v2`  |  **Grouping variant:** `a0`
**Date:** 2026-04-04T19:24:28Z  |  **Runs:** 5
**Auto Attribution Coverage:** 65.7%

---

## Method Scores

Scores are **mean ± stderr** across runs.  Higher is better. Random baseline is shuffled group assignments.

### M1 and M2 — Group Validation

Negative sources:  **easy** = other named groups  |  **medium** = Ungrouped features  |  **hard** = features outside the top-50 seed window

| | Easy | Medium | Hard | Chance |
|:--|-----:|-------:|-----:|-------:|
| **M1** Feature ID (1-in-10) | 85.2% ± 7.0% | 91.9% ± 5.6% | 67.4% ± 7.0% | 10% |
| **M2** Text Match (5-in-10) | 84.4% ± 4.8% | 87.2% ± 4.6% | 86.4% ± 4.0% | 50% |
| *Random M1* | 9.4% ± 3.5% | — | — | 10% |
| *Random M2* | 50.0% ± 2.9% | — | — | 50% |

*Easy: 165 M1 trials · 50 M2 tasks*

### D1 and D2 — Description Validation

| | Score | Chance | Tasks |
|:--|------:|-------:|------:|
| **D1** Evidence → pick correct description (1-in-10) | 93.2% ± 0.0% | 10% | 250 |
| **D2** Description → pick correct snippets (5-in-10) | 92.3% ± 0.0% | 50% | 250 |

---

## Per-Group Detail

Auto grouping, easy difficulty. Sorted by accuracy, highest first.

### M1 — Feature Identification

| Group | Accuracy | ±Stderr | Runs | Trials |
|:------|----------:|--------:|-----:|-------:|
| female pronoun she | 100.0% | 0.0% | 5 | 10 |
| hate | 100.0% | 0.0% | 5 | 25 |
| past-tense verbs | 100.0% | 0.0% | 5 | 10 |
| positive emotion | 100.0% | 0.0% | 5 | 10 |
| say 'At first' | 100.0% | 0.0% | 5 | 25 |
| school | 100.0% | 0.0% | 5 | 15 |
| say a noun phrase | 90.0% | 10.0% | 5 | 10 |
| predicate after 'is' | 60.0% | 6.3% | 5 | 25 |
| say over time | 55.0% | 12.2% | 5 | 20 |
| past be tokens | 46.7% | 17.0% | 5 | 15 |

### M2 — Text Snippet Match

| Group | Accuracy | ±Stderr | Runs | Tasks |
|:------|----------:|--------:|-----:|------:|
| female pronoun she | 100.0% | 0.0% | 5 | 5 |
| hate | 100.0% | 0.0% | 5 | 5 |
| past-tense verbs | 100.0% | 0.0% | 5 | 5 |
| school | 96.0% | 4.0% | 5 | 5 |
| say 'At first' | 88.0% | 8.0% | 5 | 5 |
| predicate after 'is' | 80.0% | 0.0% | 5 | 5 |
| say over time | 80.0% | 10.9% | 5 | 5 |
| positive emotion | 76.0% | 4.0% | 5 | 5 |
| past be tokens | 68.0% | 4.9% | 5 | 5 |
| say a noun phrase | 56.0% | 4.0% | 5 | 5 |

### Medium and Hard Difficulties (Auto — macro only)

| Difficulty | M1 | M2 |
|:-----------|---:|---:|
| Medium | 91.9% ± 5.6% | 87.2% ± 4.6% |
| Hard   | 67.4% ± 7.0% | 86.4% ± 4.0% |

---

*Generated 2026-04-04T19:24:28Z · desc=`v2` · group=`a0`*
