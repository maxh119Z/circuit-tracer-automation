# viewing_graph — Plan (draft for comment)

Turn our SLT `test_graphs/` into **shareable Neuronpedia links** that render the graph *natively*
(source set recognized, hover descriptions, pos/neg logits, text exemplars all work) **with our
own supernode groupings and custom feature descriptions overlaid.**

Status: approach validated end-to-end on `albuquerque-capital`. This is the build plan.

---

## Approach: generate-native + save-subgraph

We do **not** upload our graph JSON (see `insights.md` for why). Instead:

1. **Generate** the graph natively on Neuronpedia from the prompt → Neuronpedia builds it with its
   own hosted transcoders, so every node resolves (descriptions, logits, exemplars).
2. **Save our view** (pinnedIds + supernodes + **clerps**) as a *subgraph* on that graph → stored
   in Neuronpedia's DB, so there is **no URL-length limit** and our custom descriptions persist.
3. **Share URL** = `/{model}/graph?slug={graph_slug}&subgraph={subgraphId}` (short, stable).

**Why this works:** a Neuronpedia-native generation reproduces the *same* `node_id`s as our graphs
(same gemma tokenizer + same GemmaScope transcoders → deterministic feature indices), so our
`node_id`-keyed supernodes and clerps map on exactly. Verified: overlaying our albuquerque
supernodes grouped all 12 correctly; the 2-clerp `node_id`-keyed test showed our descriptions.

---

## APIs (official `neuronpedia` python client + one REST call)

Auth: `NEURONPEDIA_API_KEY` (from neuronpedia.org/account).

- **Generate** — `NPGraphMetadata.generate(model_id="gemma-2-2b", prompt, graph_id, **params)` → `.url`, slug.
  - `POST /api/graph/generate`. Required: `prompt`, `modelId`, `slug`(=graph_id).
  - Limits: **prompt ≤ 64 tokens**, **gemma-2-2b only**; optional `nodeThreshold`, `edgeThreshold`,
    `maxNLogits ≤15`, `maxFeatureNodes ≤10000`, `sourceSet` (default matched our node_ids).
- **Save subgraph** — `POST /api/graph/subgraph/save`
  - body: `{ modelId, slug, displayName, pinnedIds, supernodes, clerps, overwriteId? }` → `{ subgraphId }`.
  - `pinnedIds`, `supernodes`, `clerps` are all **required** and stored in the DB.
  - **Idempotent**: pass `overwriteId` to update the same subgraph instead of duplicating.
- **Exists check** — `NPGraphMetadata.get("gemma-2-2b", graph_id)` (skip re-generate).
- **Load** — graph URL with `&subgraph=<subgraphId>` overrides URL params with the saved view.

---

## Per-graph pipeline

For each `test_graphs/<slug>.json`:

1. Read `metadata.prompt` (**keep the leading `<bos>`**), `qParams.pinnedIds` (comma→list),
   `qParams.supernodes` (JSON→list).
2. Build **clerps**: for every `node_id` that appears in a supernode and has a non-empty `clerp`
   in the graph → `[node_id, clerp]`. (Nodes without our clerp, e.g. `Emb:` nodes, left native.)
3. `graph_id = <prefix>-<slug>` (globally unique + deterministic).
4. `get(model, graph_id)` → reuse if present; else `generate(prompt, graph_id)` (Neuronpedia's
   default generation settings — we don't pass thresholds).
5. `subgraph/save(...)` with our `pinnedIds` + `supernodes` + `clerps` → `subgraphId`. **The saved
   subgraph holds the whole view**, so nothing else needs to travel in the URL.
6. `url = /{model}/graph?slug={graph_slug}&subgraph={subgraphId}` — short link that loads the saved
   pins + supernodes + clerps. No long parameterized URL to build.
7. Append row → CSV. **Write incrementally** so a crash mid-batch keeps progress.

---

## Script + CLI

`viewing_graph/build_views.py --source <sel> [--name N] [--slug-prefix P] [--limit N]`

- **`--source`** (a `resolve_source()` selector — the "which graphs" function):
  - registered name (`capital` / `gemma` / `paper`) → predefined local dir or HF subpath,
  - a local directory of `<slug>.json`,
  - `hf:gemma-2/<set>/test_graphs` → `snapshot_download` from `circuit-tracer-automation/pipeline_automation`.
- **Output** → `links_<name>.csv` (`slug, prompt, graph_slug, subgraph_id, url`). The CSV *is* the
  state: re-runs read it back and skip slugs already done.
- Lightweight: only reads prompt + qParams; never touches the big nodes/links.

---

## Pitfalls (see `insights.md` for detail)

1. **Node-id match** needs identical tokenization + same transcoders — confirmed for capitals with
   default gemma-2-2b generation; spot-check per dataset.
2. **Generation pruning may drop supernode members** → they silently don't group. We use
   Neuronpedia's default generation settings; any dropped members just won't appear (acceptable).
3. **Prompt ≤ 64 tokens** — capitals fine; wikipedia/mquake may exceed → length-check/skip.
4. **gemma-2-2b only.**
5. **Slug globally unique** → prefix; deterministic id → idempotent re-runs (get / overwriteId).
6. **`<bos>`** passed exactly as `metadata.prompt` (avoid double-BOS shifting positions).
7. **Rate limits / server compute** — each generate runs attribution (seconds); retry+backoff,
   incremental CSV.
8. **`Emb:`/`Output:` supernodes** depend on the generated graph's top prediction matching ours
   (matched in the test); concept supernodes are robust.

---

## Open tweak points (for you)

- Slug-prefix scheme (e.g. your Neuronpedia handle / project tag).
- `displayName` for the saved subgraph (e.g. the slug).
- Final CSV columns.
