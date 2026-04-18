# Shortcut Experiment

## Hypothesis

For multi-hop questions (3–4 hops) where the model answered **correctly** but the
attribution graph only shows the **final intermediate concept** (e.g. "United Kingdom")
and not the earlier ones (e.g. "The Beatles", "Brian Epstein"), the model may be
answering via a **direct association** — skipping intermediate reasoning steps entirely.

Example (from the literature):
> "The spouse of the **performer of Imagine** is" → Yoko Ono
> is filtered out if the model also predicts Yoko Ono for:
> - "The spouse of **Imagine** is" (drops "performer of")
> - "The spouse of the **performer** is" (drops "of Imagine")

## Our setup

For each shortcut candidate we generate rewritten prompts that **substitute a missed
intermediate entity directly** into the question, removing the indirection.

Original 4-hop:
> "What continent is the country in where the **director of the performer of Back in the U.S.S.R.** is a citizen?"
> chain: Back in the U.S.S.R. → The Beatles → Brian Epstein → United Kingdom → Europe

Shortcut variants:
> "What continent is the country in where **The Beatles** is from?"
> "What continent is the country in where **Brian Epstein** is from?"

If the model answers correctly on these simpler prompts, it confirms the shortcut.
If it answers correctly on the original but NOT on the simplified versions, that
suggests genuine multi-hop reasoning despite the hop not appearing in the graph.

## Files

```
shortcut_experiment/
├── README.md                         this file
├── generate_shortcut_prompts.py      reads hop_analysis.csv, calls GPT, writes CSVs
└── prompts/
    ├── prompts_shortcut.csv          → attribute these with circuit-tracer
    └── ground_truth_shortcut.csv     → use with analyze_hops.py
```

## Workflow

```bash
# Step 1 — generate shortcut prompts
cd custom_automation
python analysis/shortcut_experiment/generate_shortcut_prompts.py

# Step 2 — attribute (follow the printed instructions)
circuit-tracer attribute-batch \
  --csv analysis/shortcut_experiment/prompts/prompts_shortcut.csv \
  --graph_file_dir ./test_graphs

# Step 3 — group and push to viewer
GROUPING_VARIANTS=a2 ./batch_reproduce.sh \
  analysis/shortcut_experiment/prompts/prompts_shortcut.csv

# Step 4 — analyze
python analysis/analyze_hops.py \
  --ground_truth analysis/shortcut_experiment/prompts/ground_truth_shortcut.csv \
  --variants a2
```
