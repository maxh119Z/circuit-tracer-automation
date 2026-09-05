# Insights — reconstructing Neuronpedia links from our test_graphs

Findings from working out how to turn `test_graphs/<slug>.json` into shareable Neuronpedia graph
links. Ordered roughly as we discovered them.

## The core realization
Our `test_graphs` already contain everything needed to reconstruct a Neuronpedia view. `qParams`
(`pinnedIds`, `supernodes`, `clerps`, …) **is** the graph viewer's URL query state. Our own
`batch_fetch_neuronpedia.py` does the exact reverse (reads `supernodes` out of a Neuronpedia URL),
so the round-trip is proven — going JSON → link is just re-encoding those fields.

## Key facts

1. **`feature_type: "cross layer transcoder"` is a cosmetic mislabel.** `graph_models.py:55` sets it
   unconditionally on every transcoder node regardless of suite. Our graphs use **single-layer**
   GemmaScope transcoders (`mwhanna/gemma-scope-transcoders`). Nothing keys off the string —
   matching is by `node_id`. Harmless; leave it.

2. **`qParams` string vs array.** The local circuit-tracer viewer stores `pinnedIds` as a
   comma-string and `supernodes` as a JSON-string (its `init-cg.js` parses them at load). Neuronpedia
   expects arrays. (Only relevant to the upload path we abandoned.)

3. **`metadata.scan` == Neuronpedia `model_id`.** The upload client literally does
   `model_id = metadata["scan"]`. For us that must be **`gemma-2-2b`**, not the transcoder path.
   Allowed charset: alphanumeric + `_ - .` only (a `/` fails validation).

4. **Source set + descriptions come from `metadata.feature_details.neuronpedia_source_set`.** For
   gemma-2-2b transcoders that value is **`gemmascope-transcoder-16k`** → maps to
   `neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k`. We confirmed the set exists **and** hosts
   per-feature dashboards (`0-gemmascope-transcoder-16k/1961` returns activations + pos/neg logits).

5. **`node_id` encoding.** `node_id = "{layer}_{feature_idx}_{ctx}"`. The separate `feature` field is
   `cantor_pairing(layer, feature_idx)` — a unique node *encoder*, **not** the feature index. This
   mismatch is why an uploaded graph doesn't cleanly resolve per-node hovers: Neuronpedia can't map
   the node to its hosted feature.

6. **Native generation reproduces our `node_id`s.** Because features are deterministic given
   (model + transcoders + prompt), a Neuronpedia-generated graph has the same `node_id`s as ours →
   our `node_id`-keyed supernodes and clerps overlay perfectly. Verified: all 12 albuquerque
   supernodes grouped; a 2-clerp `node_id`-keyed test showed our descriptions.

7. **URL param formats.** `pinnedIds` = comma list; `supernodes` = `[[label, id, id, …], …]` JSON;
   `clerps` = `[[node_id, description], …]` JSON; plus `pruningThreshold`, `densityThreshold`. The
   **clerps key is `node_id`** (confirmed). URL-encode with `quote(..., safe='')`.

8. **URL length is the wall.** A ~24-feature graph with full clerps ≈ **7.8 KB** — right at the
   common ~8 KB URL cap. Bigger graphs overflow, so raw-URL-with-clerps isn't reliable at scale.

9. **`subgraph/save` removes that wall.** `POST /api/graph/subgraph/save` persists
   `{pinnedIds, supernodes, clerps, displayName}` in Neuronpedia's DB and returns a `subgraphId`,
   loaded via `&subgraph=<id>` (which overrides URL params). No length limit; clerps persist;
   idempotent via `overwriteId`. This is the same "Save" behavior as grouping nodes in their UI.

10. **Generate API constraints.** `prompt ≤ 64 tokens`, **gemma-2-2b only**; optional
    `nodeThreshold`/`edgeThreshold`/`maxFeatureNodes`. Runs attribution server-side (seconds/graph).

## Why not upload the graph (the path we rejected)
Uploading our own graph JSON is mechanically possible — after array-ifying `qParams`, setting
`scan = gemma-2-2b`, and adding `feature_details` it passes the schema — but it lands as a "Custom
Upload" that Neuronpedia doesn't natively understand: even with `neuronpedia_source_set` set, the
per-node hovers (pos/neg logits, text exemplars) don't resolve, because our nodes carry a
cantor-paired `feature` field instead of the raw feature index Neuronpedia's dashboards key on, so
it can't map a node to its hosted feature. Fixing that means either reverse-engineering and
rewriting every node's feature encoding to match Neuronpedia's expectation (fragile) or self-hosting
our own feature JSONs behind `feature_json_base_url` (extra infra: S3 + CORS + HTTPS) — all to host a
36 MB graph per prompt for data Neuronpedia already has. Generating natively sidesteps every bit of
this: Neuronpedia builds the graph in its own format so descriptions/logits/exemplars work for free,
and we only overlay our groupings. Cheaper, simpler, higher fidelity.
