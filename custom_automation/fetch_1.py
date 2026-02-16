# This file extracts the trigger/activation tax for a singular feature

import requests
import json
import gzip
import io

# --- CONFIGURATION ---
HF_REPO = "mwhanna/gemma-scope-transcoders" 
TARGET_LAYER = 14       
TARGET_FEATURE = 2268   # Local Index

def get_feature_data():
    base_url = f"https://huggingface.co/{HF_REPO}/resolve/main/features"
    
    # 1. Fetch Index
    print(f"Fetching index...")
    resp = requests.get(f"{base_url}/index.json.gz")
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
        index_data = json.load(f)
        
    # 2. Get Offsets
    layer_info = index_data.get(str(TARGET_LAYER))
    filename = layer_info['filename']
    offsets = layer_info['offsets']
    start, end = offsets[TARGET_FEATURE], offsets[TARGET_FEATURE + 1]
    
    # 3. Fetch Data
    print(f"Fetching bytes {start}-{end} from {filename}...\n")
    headers = {"Range": f"bytes={start}-{end-1}"}
    resp = requests.get(f"{base_url}/{filename}", headers=headers)
    
    content = resp.content
    
    # 4. Decode (handling the 4-byte prefix)
    gzip_start = content.find(b'\x1f\x8b')
    if gzip_start != -1:
        try:
            decompressed = gzip.decompress(content[gzip_start:])
            return json.loads(decompressed)
        except Exception as e:
            print(f"❌ Decompression error: {e}")
            return None
    return json.loads(content)

# --- MAIN EXECUTION ---
data = get_feature_data()

if data:
    print(f"\nSUCCESS! Analysis of Top Activations for Layer {TARGET_LAYER}, Feature {TARGET_FEATURE}:")
    print("-" * 60)
    
    # Get top examples
    examples = []
    if 'examples_quantiles' in data:
        examples = data['examples_quantiles'][0]['examples']
    elif 'activations' in data:
        examples = data['activations']

    for i, ex in enumerate(examples[:10]):
        tokens = ex.get('tokens', [])
        scores = ex.get('tokens_acts_list') or ex.get('values') or []
        
        # Determine the "Main Trigger" (Max Score)
        if scores and len(scores) == len(tokens):
            max_score = max(scores)
            max_index = scores.index(max_score)
            main_token = str(tokens[max_index])
            full_text = "".join([str(t) for t in tokens]).replace('\n', ' ')
            
            print(f"Example {i+1}:")
            print(f"TRIGGER:  '{main_token}' (Score: {max_score:.2f})")
            print(f"CONTEXT:  \"{full_text[:119]}...\"")
            print("-" * 60)
            
        else:
            print(f"Example {i+1}: Data mismatch.")
            print(f"   Tokens: {len(tokens)}, Scores: {len(scores)}")