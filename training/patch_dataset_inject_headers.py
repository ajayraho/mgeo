import json
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ollama_utils import call_ollama_chat 

# --- CONFIGURATION ---
INPUT_FILE = "data/rl_finetuning_dataset_verbose.json"  # Your current best dataset
OUTPUT_FILE = "data/rl_finetuning_dataset_structured.json" # The new Goal
TEACHER_MODEL = "gpt-oss" 

def inject_headers():
    if not os.path.exists(INPUT_FILE):
        print("❌ Input file not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"💉 Injecting Headers into {len(data)} items...")
    
    new_data = []
    
    for item in tqdm(data, desc="Structuring"):
        current_out = item['output']
        
        # PROMPT: Force Bold Headers
        sys_msg = """You are a Text Formatter.
TASK: Add **Bold Headers** to the product features.
INPUT: A Title and a Features paragraph.
OUTPUT: The SAME Title and Features, but with bold headers (e.g., **Design** - ...) inserted at the start of distinct phrases.
RULES:
1. Keep the content exactly the same. Just add headers.
2. Format: Title: ... \nFeatures: **Header** - Content. **Header** - Content.
3. Ensure 'Title:' and 'Features:' are on separate lines."""

        user_msg = f"TEXT TO FORMAT:\n{current_out}"

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        
        try:
            # Low temp for strictly formatting
            rewritten = call_ollama_chat(messages, model=TEACHER_MODEL, temperature=0.1)
            
            # Validation: Check if it actually added headers
            if rewritten and "**" in rewritten:
                item['output'] = rewritten
            else:
                pass # Keep original if fails
                
            new_data.append(item)
        except:
            new_data.append(item)
            
        if len(new_data) % 50 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(new_data, f, indent=4)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_data, f, indent=4)
        
    print(f"✅ Structure Injected. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    inject_headers()