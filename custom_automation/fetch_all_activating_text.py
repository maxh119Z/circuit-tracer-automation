import requests
import json
import gzip
import io
import time
import concurrent.futures
import os
from collections import Counter

# --- CONFIGURATION ---
HF_REPO = "mwhanna/gemma-scope-transcoders" 
OUTPUT_FILE = "pruned_activations.json" 
GRAPH_FILE = "../test_graphs/test-run.json"

# PRUNING THRESHOLD (Lower is stricter/more important)
PRUNING_THRESHOLD = 0.40

MAX_WORKERS = 10

def load_pruned_nodes_from_graph():
    if not os.path.exists(GRAPH_FILE):
        print(f"❌ Error: Graph file not found at {GRAPH_FILE}")
        return []
    
    print(f"📂 Loading graph from: {GRAPH_FILE}")
    with open(GRAPH_FILE, 'r') as f:
        data = json.load(f)
    
    all_nodes = data.get('nodes', [])
    all_links = data.get('links', [])
    
    # --- ORPHAN FILTER SETUP ---
    # Create a set of all nodes that are actually connected by edges.
    # The website hides nodes that have zero surviving connections.
    connected_ids = set()
    for link in all_links:
        connected_ids.add(link['source'])
        connected_ids.add(link['target'])
    
    pruned_list = []
    
    print(f"✂️  Pruning Logic:")
    print(f"   - Global Threshold: Score <= {PRUNING_THRESHOLD}")
    print(f"   - Type Filter: 'cross layer transcoder' only")
    print(f"   - Orphan Filter: Enabled")

    for n in all_nodes:
        f_type = n.get('feature_type', 'unknown')
        score = float(n.get('influence') or n.get('score') or 0.0)
        node_id = n.get('node_id', '')

        # 1. TYPE FILTER (Prevents overcounting Error Nodes in ALL layers)
        if f_type != 'cross layer transcoder':
            continue

        # 2. SCORE FILTER (Strict Raw Score)
        if score > PRUNING_THRESHOLD:
            continue

        # 3. ORPHAN FILTER (Ensures node is visible in the graph)
        if node_id not in connected_ids:
            continue

        try:
            parts = node_id.split('_')
            layer_idx = int(parts[0])
            feature_idx = int(parts[1])

            # Exclude final logit layers if necessary
            if layer_idx < 26:
                pruned_list.append({
                    "layer": layer_idx,
                    "feature": feature_idx,
                    "score": score,
                    "type": f_type
                })
        except (KeyError, IndexError, ValueError):
            continue

    return pruned_list

def load_global_index():
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/features/index.json.gz"
    print("📥 Downloading Global Index...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return None

def fetch_single_feature(node, index_data):
    layer = str(node['layer'])
    feature_idx = node['feature']
    
    if layer not in index_data: return None
    layer_info = index_data[layer]
    
    if feature_idx >= len(layer_info['offsets']) - 1: return None
    start = layer_info['offsets'][feature_idx]
    end = layer_info['offsets'][feature_idx + 1]
    
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/features/{layer_info['filename']}"
    headers = {"Range": f"bytes={start}-{end-1}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.content
        gzip_start = content.find(b'\x1f\x8b')
        if gzip_start != -1:
            data = json.loads(gzip.decompress(content[gzip_start:]))
        else:
            data = json.loads(content)
        return parse_usable_data(node, data)
    except Exception:
        return None

def parse_usable_data(node, raw_json):
    parsed = {
        "id": f"{node['layer']}_{node['feature']}", 
        "layer": node['layer'],
        "influence_score": node['score'],
        "top_activations": []
    }
    examples = raw_json.get('examples_quantiles', [{}])[0].get('examples', []) or raw_json.get('activations', [])
    for ex in examples[:5]:
        tokens = ex.get('tokens', [])
        scores = ex.get('tokens_acts_list', []) or ex.get('values', [])
        if scores and len(scores) == len(tokens):
            max_val = max(scores)
            parsed['top_activations'].append({
                "trigger": str(tokens[scores.index(max_val)]),
                "score": round(max_val, 2),
                "context": "".join([str(t) for t in tokens]).replace('\n', ' ')
            })
    return parsed

if __name__ == "__main__":
    nodes_to_fetch = load_pruned_nodes_from_graph()
    
    if not nodes_to_fetch:
        print("🛑 No nodes found meeting the criteria.")
    else:
        layer_counts = Counter(n['layer'] for n in nodes_to_fetch)
        print("\n📊 PRUNED CIRCUIT SUMMARY:")
        print("-" * 40)
        for layer in sorted(layer_counts.keys()):
            print(f"Layer {layer:02}: {layer_counts[layer]} nodes")
        print("-" * 40)

        global_index = load_global_index()
        if global_index:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_single_feature, n, global_index): n for n in nodes_to_fetch}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    res = future.result()
                    if res: results.append(res)
                    if (i + 1) % 10 == 0: 
                        print(f"✅ Downloaded {i+1}/{len(nodes_to_fetch)} features...")

            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n🎉 Done! {len(results)} features saved to {OUTPUT_FILE}.")
