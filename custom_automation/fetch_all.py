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
# UPDATE THIS PATH to your actual graph file location
GRAPH_FILE = "../test_graphs/test-run.json"
# Set this to 0.50 to mach the website (Top 50% of max influence)
PRUNING_THRESHOLD = 0.40
MAX_WORKERS = 10

def load_pruned_nodes_from_graph():
    """Finds the graph file and loads nodes based on relative threshold."""
    if not os.path.exists(GRAPH_FILE):
        print(f"🔍 Searching for graph JSON in ./test_graphs...")
        for root, dirs, files in os.walk("./test_graphs"):
            for file in files:
                if file.endswith(".json") and "metadata" not in file:
                    found_path = os.path.join(root, file)
                    return load_from_path(found_path)
        print("❌ Error: No graph files found.")
        return []
    return load_from_path(GRAPH_FILE)

def load_from_path(path):
    print(f"📂 Loading graph from: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    all_nodes = data.get('nodes', [])
    pruned_list = []
    
    # EXACT JS LOGIC REPLICATION with Manual Layer 27 Exclusion
    print(f"✂️  Pruning Logic: Keep if Score <= {PRUNING_THRESHOLD} (Excluding Layer 27)")

    for n in all_nodes:
        # 1. Get properties safely
        f_type = n.get('feature_type', 'unknown')
        score = float(n.get('influence') or n.get('score') or 0.0)
        
        # 2. Apply the exact 3 conditions from the JS
        keep_condition = (
            f_type == 'embedding' or 
            f_type == 'logit' or 
            score <= PRUNING_THRESHOLD
        )

        if keep_condition:
            try:
                if '_' in n['node_id']:
                    parts = n['node_id'].split('_')
                    if len(parts) >= 2 and parts[0].isdigit():
                        layer_idx = int(parts[0])
                        feature_idx = int(parts[1])

                        # 🛑 EXPLICIT EXCLUSION FOR LAYER 27
                        if layer_idx == 27: 
                            continue

                        pruned_list.append({
                            "layer": layer_idx,
                            "feature": feature_idx,
                            "score": score,
                            "type": f_type
                        })
            except (KeyError, IndexError, ValueError):
                continue

    print(f"📊 Matches Website: Kept {len(pruned_list)} nodes.")
    return pruned_list

def load_global_index():
    """Downloads the master index from Hugging Face."""
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
    """Fetches a specific feature's activation text."""
    layer = str(node['layer'])
    feature_idx = node['feature']
    
    if layer not in index_data: return None
    layer_info = index_data[layer]
    
    # Get byte offsets
    if feature_idx >= len(layer_info['offsets']) - 1: return None
    start = layer_info['offsets'][feature_idx]
    end = layer_info['offsets'][feature_idx + 1]
    
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/features/{layer_info['filename']}"
    headers = {"Range": f"bytes={start}-{end-1}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.content
        
        # Handle GZIP prefix if present
        gzip_start = content.find(b'\x1f\x8b')
        if gzip_start != -1:
            data = json.loads(gzip.decompress(content[gzip_start:]))
        else:
            data = json.loads(content)
            
        return parse_usable_data(node, data)
    except Exception:
        return None

def parse_usable_data(node, raw_json):
    """Extracts clean text and scores for the LLM."""
    parsed = {
        "id": f"{node['layer']}_{node['feature']}", 
        "layer": node['layer'],
        "influence_score": node['score'],
        "top_activations": []
    }
    
    # Handle repo-specific structure
    examples = raw_json.get('examples_quantiles', [{}])[0].get('examples', []) or raw_json.get('activations', [])
    
    for ex in examples[:5]: # Keep top 5 examples
        tokens = ex.get('tokens', [])
        scores = ex.get('tokens_acts_list', []) or ex.get('values', [])
        
        if scores and len(scores) == len(tokens):
            max_val = max(scores)
            max_idx = scores.index(max_val)
            
            parsed['top_activations'].append({
                "trigger": str(tokens[max_idx]),
                "score": round(max_val, 2),
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
        print("\n📊 PRUNED CIRCUIT SUMMARY:")
        print("-" * 40)
        for layer in sorted(layer_counts.keys()):
            print(f"Layer {layer:02}: {layer_counts[layer]} nodes")
        print("-" * 40)
        print(f"Total Nodes to Fetch: {len(nodes_to_fetch)}\n")

        # Start scraping
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
            print(f"\n🎉 Done! Full activation data saved to {OUTPUT_FILE}.")