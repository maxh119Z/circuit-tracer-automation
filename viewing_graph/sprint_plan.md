# Sprint plan 9/5-9/8 Max + Ameen

Ship the generate-native + save-subgraph pipeline (see `plan.md`) that turns our SLT graphs into shareable Neuronpedia links. Capitals first, then all datasets.

## Person 0 - sprint plan (DONE)
- **Design:** `plan.md` · **Findings:** `insights.md`
- **Tokens:** `HF_TOKEN` (pull graphs), `NEURONPEDIA_API_KEY` (generate + save)

## Person 1 — generate + view engine (capitals)

### Person 2 checks person 1's work afterwards.

### a. Huggingface pull default capitals/
- `snapshot_download("circuit-tracer-automation/pipeline_automation", allow_patterns="gemma-2/capital/test_graphs/*")` with `HF_TOKEN`.
- Checkpoint: Make sure all capital `.json` on disk; can read `metadata.prompt` + `qParams`.

### b. Wire Neuronpedia generate
- `NPGraphMetadata.generate(model_id="gemma-2-2b", prompt, graph_id)` — pass `prompt` **with the leading `<bos>`**, default generation settings.
- Checkpoint: (your "are graphs generated?"): a graph generates; open it — nodes render natively (hover descriptions, pos/neg logits, exemplars all resolve). **Max: Basically make sure the default links for each graph works.**

### c. Wire save-view + share URL
- From the JSON: collect `pinnedIds`, `supernodes`, and clerps (`[[node_id, node["clerp"]], …]`, reshaping the existing clerps). `POST subgraph/save` → `subgraphId`. Build `/{model}/graph?slug={graph_slug}&subgraph={subgraphId}`.
- Checkpoint: (your "check save for 1 graph"): open the short URL — our supernodes **and** clerps show now. If so, move on.

### d. Batch capitals + make it robust
- Loop all capitals. Write `links_capital.csv` **incrementally** (`slug, prompt, graph_slug, subgraph_id, url`). 
- Checkpoint: full run completes; all 100 graphs have cerlp + supernodes shown when open link. Link is saved in a CSV with its slug. 


**P1 Done =** `links_capital.csv` with working links for every capital prompt in HF.

---

## Person 2 — make it work for all our graphs  *(starts after P1 Done)*

### Person 1 checks person 2's work afterwards.

Reuses P1's per-graph function unchanged; adds coverage + a source switch.

### a. Inventory
- Enumerate graphs across the HF folders: **capitals, wikipedia, anthropic,paper_prompts**. The hard one is the paper's prompts since it spans multiple datasets.
- **paper_prompts:** filter to the graphs the paper actually uses — cross-reference the figures /
  appendix (this is the fiddly one; budget time).
- Checkpoint: Have a list of all files with test_graphs + links to it we want to wire to.

### b. Simple source switch (dir argument)
- `--source {capital|wikipedia|anthropic|paper}` → different `gemma-2/<set>/test_graphs` subpath
  (or the paper manifest). Pipeline logic unchanged.
- Checkpoint: `--source wikipedia` pulls + runs a couple graphs.

### c. Run + verify each dataset
- Produce `links_<set>.csv` per dataset.
- Checkpoint: spot-check a few links per dataset (groupings + clerps render).

**P2 Done =** `links_<set>.csv` for capitals / wikipedia / anthropic / paper.

---