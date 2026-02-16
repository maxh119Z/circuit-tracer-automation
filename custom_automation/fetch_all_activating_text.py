# This file extracts the trigger/activation tax from each layer with pruning threshold

#TO DO:
# More analysis needs to be done to ensure these pruned features match ones on Neuropedia website!
# I am pretty sure this pruning method is incorrect. Will figure out tomorrow. 2/15

import requests
import json
import gzip
import io
import time
import concurrent.futures
import os
from collections import Counter

# --- CONFIGURATION ---
HF_REPO = "mwhanna/gemma-scope-transcoders" # Activation text is stored here, not on NeuroPEDIA!
OUTPUT_FILE = "pruned_activations.json" 
GRAPH_FILE = "./test_graphs/test-run.json"
PRUNING_THRESHOLD = 0.67 # A bit weird, I'm not sure how to match 0.5 on website
MAX_WORKERS = 10

def load_pruned_nodes_from_graph():
    if not os.path.exists(GRAPH_FILE):
        print(f"Searching for graph JSON in ./test_graphs...")
        for root, dirs, files in os.walk("./test_graphs"):
            for file in files:
                if file.endswith(".json") and "metadata" not in file:
                    return load_from_path(os.path.join(root, file))
        return []
    return load_from_path(GRAPH_FILE)

def load_from_path(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    all_nodes = data.get('nodes', [])
    pruned_list = []
    
    for n in all_nodes:
        score = n.get('influence') or 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score >= PRUNING_THRESHOLD: #If this feature is relevant
            try:
                # Extract layer and local index from '14_2268_7' as discovered. 2268 is the local ID!
                parts = n['node_id'].split('_')
                pruned_list.append({
                    "layer": int(parts[0]),
                    "feature": int(parts[1]),
                    "score": score
                })
            except (KeyError, IndexError, ValueError):
                continue
    return pruned_list

def fetch_single_feature(node, index_data):
    layer = str(node['layer'])
    feature_idx = node['feature']
    if layer not in index_data: return None
    layer_info = index_data[layer]
    start, end = layer_info['offsets'][feature_idx], layer_info['offsets'][feature_idx + 1]
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/features/{layer_info['filename']}"
    headers = {"Range": f"bytes={start}-{end-1}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.content
        gzip_start = content.find(b'\x1f\x8b') # Handle the 4-byte prefix found in mwhanna's data
        if gzip_start != -1:
            data = json.loads(gzip.decompress(content[gzip_start:]))
        else:
            data = json.loads(content)
        return parse_usable_data(node, data)
    except Exception:
        return None

def parse_usable_data(node, raw_json):
    parsed = {"id": f"{node['layer']}_{node['feature']}", "layer": node['layer'], "top_activations": []}
    examples = raw_json.get('examples_quantiles', [{}])[0].get('examples', []) or raw_json.get('activations', [])
    for ex in examples[:5]:
        tokens, scores = ex.get('tokens', []), ex.get('tokens_acts_list', [])
        if scores and len(scores) == len(tokens):
            max_idx = scores.index(max(scores))
            parsed['top_activations'].append({
                "trigger": str(tokens[max_idx]),
                "context": "".join([str(t) for t in tokens]).replace('\n', ' ')
            })
    return parsed

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    nodes_to_fetch = load_pruned_nodes_from_graph()
    
    if not nodes_to_fetch:
        print("🛑 No nodes found meeting the threshold.")
    else:
        # --- LAYER SUMMARY ---
        layer_counts = Counter(n['layer'] for n in nodes_to_fetch)
        print("\n📊 PRUNED CIRCUIT SUMMARY (Threshold >= 0.5):")
        print("-" * 40)
        for layer in sorted(layer_counts.keys()):
            print(f"Layer {layer:02}: {layer_counts[layer]} nodes")
        print("-" * 40)
        print(f"Total Nodes to Fetch: {len(nodes_to_fetch)}\n")

        # Start scraping
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/features/index.json.gz"
        idx_resp = requests.get(url)
        with gzip.GzipFile(fileobj=io.BytesIO(idx_resp.content)) as f:
            global_index = json.load(f)

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_single_feature, n, global_index): n for n in nodes_to_fetch}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: results.append(res)
                if (i + 1) % 10 == 0: print(f"✅ Downloaded {i+1}/{len(nodes_to_fetch)} features...")

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n🎉 Done! Full activation data saved to {OUTPUT_FILE}.")