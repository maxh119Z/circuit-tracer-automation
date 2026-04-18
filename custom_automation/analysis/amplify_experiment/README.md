# Amplify Experiment

## Question

When the correct answer concept is already internally represented in the attribution
graph but the model still outputs the wrong token — can we amplify that feature to
force the correct output?

## Motivation

From `interesting_graphs.md`, several cases show the correct concept present
internally but suppressed or overridden at the output:

- **mquake-63**: Islam + Muhammad supernodes active, model outputs "The"
- **mquake-140**: Prophet Muhammad supernode active, model outputs "Abdul"
- **mquake-2043**: English-language supernodes active, model outputs "Russian"
- **mquake-2066**: European country supernodes active, model outputs "Africa"

These are cases where the model "knows" the answer but fails to emit it.
Amplifying the state node (the feature representing the correct concept) should
increase the probability of the correct output token.

## Method

1. **Identify candidates** (from all mquake cases): `model_correct=False` AND
   the correct answer token appears in the supernode group names
   (e.g. "say Europe" or "Europe" in `all_supernodes` when answer=Europe).

2. **Locate the state node**: find the transcoder feature(s) whose clerp or
   supernode label matches the correct answer token.

3. **Amplify**: multiply the activation of those feature(s) by a scalar α > 1
   (e.g. α = 2, 5, 10) during the forward pass.

4. **Measure**: does the model now output the correct answer? At what α does it
   flip? Does amplifying the right concept hurt other outputs (side effects)?

## Expected findings

- If amplification works at low α → the feature was genuinely suppressed, not absent.
  The circuit had the right concept but something downstream overrode it.
- If it requires very high α or never works → the feature was incidental / not
  causally connected to the output pathway.
- Compare to amplifying a random unrelated feature of similar influence as a control.

## Relationship to existing work

- Uses the same multiplicative steering approach as `run_interventions.py`
  (multiply feature activations by a scalar).
- Candidates come from `hop_analysis.csv`: rows where `model_correct=False`
  and `hop_found=True` (correct concept present despite wrong output).
- Complements the shortcut experiment: shortcuts test cases where the model is
  correct but skips steps; this tests cases where the model fails despite having
  the right internal concept.

## Candidates

Filter `hop_analysis.csv` for:
- `model_correct = False`
- correct answer token appears (case-insensitive) in `all_supernodes`

Known examples from current data:

| Slug | Predicted | Correct | Supernode present |
|------|-----------|---------|-------------------|
| mquake-63 | The | Muhammad | Islam / Muhammad |
| mquake-140 | Abdul | Muhammad | Prophet Muhammad |
| mquake-2043 | Russian | English | English (language) |
| mquake-2066 | Africa | Europe | European country / say Europe |

## Files

```
amplify_experiment/
├── README.md                    this file
└── (scripts to be added)
```
