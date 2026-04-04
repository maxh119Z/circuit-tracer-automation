# Custom Changes to circuit_tracer/

All modifications to the original `circuit_tracer/` library are documented here.
Change types: **ADDITIVE** (new code, nothing removed), **REPLACING** (existing line(s) swapped), **CONFIG** (wiring/reference change).

---

## 1. `circuit_tracer/__main__.py`

**Type: ADDITIVE**

Added batch attribution support. Zero changes to existing functions.

| Marker in code | Lines | What it does |
|---|---|---|
| `# CUSTOM EDITS START: custom-automation` | ~150–216 | New CLI subcommand parser: `circuit-tracer attribute-batch --csv prompts.csv --graph_file_dir ./test_graphs` |
| `# ADDED: dispatch for attribute-batch` | ~222–226 | Dispatch block in `main()` that calls `run_attribution_batch()` |
| `# --- ADDED: batch attribution handler` | ~335–438 | New `run_attribution_batch()` function — parses CSV, loads model once per unique `transcoder_set`, skips slugs whose graph JSON already exists |

---

## 2. `circuit_tracer/frontend/assets/attribution_graph/init-cg.js`

**Type: ADDITIVE + REPLACING**

Three custom blocks, all marked in the file with `// CUSTOM EDITS`.

| Marker in code | Lines | Type | What it does |
|---|---|---|---|
| `// CUSTOM EDITS START: custom-automation` | ~45–48 | REPLACING | Fixes truthy-array bug: `if (urlSupernodes)` → `if (urlSupernodes.length)` so an empty array does not overwrite existing state |
| `// CUSTOM EDITS START: custom-automation` | ~50–57 | ADDITIVE | Parses `visState.supernodes` from JSON string (pipeline or Save button) and `visState.pinnedIds` from comma-separated string into arrays; includes debug `console.log` |

---

## 3. `circuit_tracer/frontend/assets/index.html`

No functional change to HTML — comment detailing changes in init-cg.js

---

## Summary

| File | Type | Upstream compatible? |
|---|---|---|
| `__main__.py` | ADDITIVE | Yes — no original code modified |
| `init-cg.js` | ADDITIVE + REPLACING | Mostly — one line replaced (urlSupernodes check), rest additive |
| `index.html` | CONFIG (comment only) | Yes |
