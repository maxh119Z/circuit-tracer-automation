#Clear browser cache to update after running if necessary


import json
import os

# --- CONFIGURATION ---
descriptions_file = "feature_descriptions.json"
graph_file = "../test_graphs/test-run.json"  # Ensure this is correct

def direct_id_merge():
    # 1. Load Descriptions
    if not os.path.exists(descriptions_file):
        print(f"❌ Missing {descriptions_file}")
        return

    with open(descriptions_file, 'r') as f:
        desc_list = json.load(f)
    
    # Map "0_7118_2" -> "Description"
    desc_map = {}
    for item in desc_list:
        # We use the ID exactly as it appears in the file
        key = str(item.get('id', 'MISSING'))
        desc_map[key] = item.get('generated_description', '')
    
    print(f"Loaded {len(desc_map)} descriptions.")

    # 2. Load Graph
    if not os.path.exists(graph_file):
        print(f"❌ Missing graph file at {graph_file}")
        return

    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    nodes = graph_data.get('nodes', [])
    input_tokens = graph_data.get('input_tokens', [])
    
    # Check for tokenizer (for label context)
    has_tokenizer = 'tokenizer' in globals()
    
    print(f"Scanning {len(nodes)} graph nodes...")

    match_count = 0
    for node in nodes:
        try:
            # --- THE FIX: USE THE NODE_ID DIRECTLY ---
            # Your debug proved this is "0_7118_2"
            node_id = str(node.get('node_id', ''))
            
            # Identify Context (for the "on 'word'" label)
            ctx_idx = node.get('ctx_idx')
            token_label = ""
            
            if ctx_idx is not None and isinstance(input_tokens, list):
                if 0 <= ctx_idx < len(input_tokens):
                    raw_token = input_tokens[ctx_idx]
                    if isinstance(raw_token, int) and has_tokenizer:
                        token_str = tokenizer.decode([raw_token]).strip()
                    else:
                        token_str = str(raw_token).strip()
                    token_label = f" (on '{token_str}')"

            # --- MERGE ---
            if node_id in desc_map:
                desc = desc_map[node_id]
                full_label = f"{desc}{token_label}"
                
                # Update ALL label fields to be safe
                node['clerp'] = full_label      # Standard field
                node['localClerp'] = full_label # User-reported field
                node['ppClerp'] = full_label    # Popup field
                
                match_count += 1

        except Exception:
            continue

    # 4. Save
    if match_count > 0:
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"✅ Success! Updated {match_count} nodes.")
        print("👉 Go refresh localhost:8041 now!")
    else:
        print("⚠️ Still 0 matches.") 
        print("DOUBLE CHECK: Are you sure 'feature_descriptions.json' has the same IDs as the graph?")

if __name__ == "__main__":
    direct_id_merge()