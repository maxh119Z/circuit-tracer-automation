import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. SETUP: Load the Explainer Model ---
# This might take a minute to download/load the weights
model_id = "Transluce/llama_8b_explainer"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto" 
)

# --- 2. DEFINE PROMPTS & HELPER FUNCTIONS ---

SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts. Your overall task is to describe features of text excerpts that cause the neuron to strongly activate.

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the neuron does not activate on any tokens in the excerpt.

Note: Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation. Note: make your final descriptions as concise as possible, using as few words as possible to describe text features that activate the neuron."""

def format_excerpt(context, trigger):
    """
    Wraps the trigger word in {{ }} within the context string.
    """
    clean_trigger = trigger.strip()
    if clean_trigger and clean_trigger in context:
        return context.replace(clean_trigger, f"{{{{{clean_trigger}}}}}")
    else:
        # Fallback if naive replacement fails
        return f"{context} [Activates on: {{{{{clean_trigger}}}}}]"

def generate_description(feature_data):
    """
    Takes a single feature object, formats its activations, and prompts the LLM.
    """
    # 1. Prepare User Prompt with Excerpts
    user_content = f"Neuron {feature_data.get('id', 'Unknown')}:\n\n"
    
    # Use top 10 activations to save context window space
    activations = feature_data.get('top_activations', [])[:10]
    
    for i, act in enumerate(activations):
        # Handle cases where 'trigger' or 'context' might be missing
        trigger = act.get('trigger', '')
        context = act.get('context', '')
        formatted_text = format_excerpt(context, trigger)
        user_content += f"Excerpt {i+1}: {formatted_text}\n"

    # 2. Format for Llama-3
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # 3. Generate
    outputs = model.generate(
        input_ids, 
        max_new_tokens=60, # Keep it concise
        do_sample=False,   # Deterministic for reproducibility
        temperature=0.0
    )

    # 4. Decode and Extract
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # The model often outputs "[DESCRIPTION]: result", so we clean that up
    if "[DESCRIPTION]:" in response:
        return response.split("[DESCRIPTION]:")[-1].strip()
    return response.strip()

# --- 3. MAIN EXECUTION ---

input_filename = "pruned_activations.json"
output_filename = "feature_descriptions.json"

# Check if file exists
if not os.path.exists(input_filename):
    print(f"ERROR: Could not find {input_filename} in the current directory.")
else:
    print(f"Loading data from {input_filename}...")
    with open(input_filename, 'r') as f:
        my_features_list = json.load(f)
    
    total = len(my_features_list)
    print(f"Found {total} features to process.\n")

    for index, feature in enumerate(my_features_list):
        print(f"[{index+1}/{total}] Processing Feature {feature.get('id')}...", end=" ")
        
        try:
            description = generate_description(feature)
            feature['generated_description'] = description
            print(f"DONE. -> {description[:50]}...")
        except Exception as e:
            print(f"FAILED: {e}")
            feature['generated_description'] = "Error generating description"

    # --- 4. SAVE RESULTS ---
    print(f"\nSaving results to {output_filename}...")
    with open(output_filename, 'w') as f:
        json.dump(my_features_list, f, indent=2)
    
    print("Success! You can now load 'feature_descriptions.json' into your viewer.")
