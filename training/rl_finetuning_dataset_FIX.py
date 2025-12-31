import json
import re
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ollama_utils import call_ollama_chat  # Ensure this is accessible

# --- CONFIGURATION ---
INPUT_FILE = "data/rl_finetuning_dataset_v2_trash.json"
OUTPUT_FILE = "data/rl_finetuning_dataset_FIXED.json"
TEACHER_MODEL = "gpt-oss"  # or "llama3:8b" if gpt-oss is unavailable

def generate_hybrid_output(item):
    """
    Uses the Teacher Model to generate a superior, SEO-friendly output.
    """
    # Extract components
    raw_input = item['input']
    
    # Extract Title specifically
    input_title = raw_input.split('\n')[0].replace("Title:", "").strip()
    
    # Extract Visual Truth and Query from instruction
    try:
        query = item['instruction'].split("query: '")[1].split("'")[0]
        visuals = item['instruction'].split("Visual Truth: ")[1].split("\n")[0]
    except:
        # Fallback if parsing fails
        query = "Product"
        visuals = "Visual details"

    # --- THE MAGIC PROMPT ---
    sys_msg = "You are an Elite SEO Copywriter. Your goal is to optimize product listings for Search Visibility AND Visual Accuracy."
    
    user_msg = f"""
### TASK
Rewrite the Product Title and Features to be High-Ranking (SEO) AND Visually Accurate.

### INPUT DATA
1. **Original Title:** "{input_title}"
2. **Target Query:** "{query}"
3. **Visual Truth:** "{visuals}"

### STRICT RULES FOR REWRITING
1. **KEEP THE BRAND:** You MUST retain the Brand Name (usually the first 1-2 words of the Original Title).
2. **KEYWORD PRESERVATION:** You MUST retain the exact phrase "{query}" in the title.
3. **VISUAL INJECTION:** Insert adjectives from "Visual Truth" (colors, materials, shapes, texture, patterns) to describe the product.
4. **NO DELETIONS:** Do NOT remove functional keywords like "Gift", "Valentines", "Woman", "Adjustable".
5. **PRESERVE SPECS & DIMENSIONS:** You MUST retain any measurements (oz, ml, cm, inches, lbs) found in the Original Title.
   - *Bad:* "Small Bottle"
   - *Good:* "1.7 fl oz (50ml) Bottle"
   - If the Visual Truth provides precise dimensions, add them. If not, keep the Original ones.
6. **NO PACKAGING HALLUCINATIONS:** - **WRONG:** "Clear Glass Bottle with White Cap" (This describes the container).
   - **RIGHT:** "Kinerase Pro Therapy Cream in a Clear Glass Bottle" (This describes the product).
   - Never replace the Product Name with the Container Name.
### FORMAT
Produce the output in this EXACT format:
[Optimized Title]
[Optimized Features (Pipe Separated)]

Do not write "Here is the title" or any other conversational filler. Just the two lines, DONT OUTPUT ANYTHINNG ELSE.
"""

    # Call the Teacher Model
    response = call_ollama_chat(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3
    )
    
    if response:
        return response.strip()
    return item['output'] # Fallback to old output if failed

# --- MAIN LOOP ---
print(f"🚀 Starting Data Regeneration using {TEACHER_MODEL}...")

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

fixed_data = []

# Process first 5 for testing, remove [:5] to run full dataset
for i, item in enumerate(tqdm(data)):
    new_output = generate_hybrid_output(item)
    
    # Sanity check: Ensure it didn't generate garbage
    if len(new_output) < 10: 
        new_output = item['output']
        
    # Update the item
    item['output'] = new_output
    
    # OPTIONAL: Chain-of-Thought Injection (Prepend Query to Input)
    # This teaches the model to look at the input for the query
    current_input_body = item['input']
    if "Target Query:" not in current_input_body:
        # We inject the query at the top of the input text
        query_str = item['instruction'].split("query: '")[1].split("'")[0]
        item['input'] = f"Target Query: {query_str}\n{current_input_body}"

    fixed_data.append(item)

# Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(fixed_data, f, indent=4)

print(f"✅ Regeneration Complete! Saved to {OUTPUT_FILE}")
print("👉 Now RETRAIN your model using this new file.")