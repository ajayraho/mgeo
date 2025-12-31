import json
import os
from tqdm import tqdm

import sys
import os
import json
import requests
import pandas as pd
import re
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ollama_utils import call_ollama_chat  # <--- Using your utils

# --- CONFIGURATION ---
INPUT_FILE = "data/rl_finetuning_dataset.json"
OUTPUT_FILE = "data/rl_finetuning_dataset_verbose.json"
TEACHER_MODEL = "gpt-oss"  # <--- The Smart Teacher

def rewrite_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"🔄 Rewriting {len(data)} examples using {TEACHER_MODEL}...")
    print("   Goal: Convert 'Robot Lists' -> 'Persuasive Paragraphs'")
    
    new_data = []
    
    # We iterate with index to catch errors if needed
    for i, item in enumerate(tqdm(data, desc="Polishing Data")):
        current_output = item['output']
        current_input = item['input'] # Use input context if available to help accuracy
        
        # 1. Define the System Role (The Style Guide)
        sys_msg = """You are an Expert E-Commerce Copywriter.
TASK: Rewrite the provided Product Description to be **Persuasive** and **Readable**.
RULES:
1. Convert pipe-separated features (|) into **Natural Bullet Points** or **Flowing Paragraphs**.
2. **Keep ALL visual details** (colors, materials, textures). Do not lose facts.
3. **Do NOT hallucinate.** Do not add features that are not in the input. STRICTLY DONT HALLUCINATE.
4. Use a premium, professional tone, long descriptive.
5. Return ONLY the rewritten text. No conversational filler.

Example:
- Great Output:
      Title: Non-Slip Chenille Bath Rug with Thick Sole Cushion - Super Soft Water Absorbent Anti-Skidding Fast Dry Plush Bathroom Mats for Shower Tub Entryway, Aqua, 20" W x 32" L
      Features: READY MADE - Oversized rectangle bath rug (20" x 32") in aqua color with textured stripe pattern. Suitable for bathroom, shower, tub, kitchen, bedroom, and door entrance. WELL ABSORBENT - Quickly soaks up water while stepping out of the bath or shower. NON-SLIP - Skid-resistant bottom made from hot melt adhesive material to prevent slipping. EXTRA SOFT - Pamper your feet with super soft microfiber chenille fabric. EASY CARE - Hand wash or machine wash in cold water, tumble dry low and shake up the rug to maintain its shape....
- Sub-optimal Output:
      Title: Non Slip Chenille Bathroom Slippers Thick Sole Cushion Non Slip
      Features: Aqua Blue Durable Stitched Material, 20" W x 32" L Aqua blue towel with distinctive texture and pattern | Durable stitched material for regular use | Rich vibrant hue adds visual appeal to any bathroom | Super soft water absorbent anti-skidding fast dry thick plush kitchen mats | 20" W x 32" L one piece design | Ideal for bathroom shower tub entryway

STRICT OUTPUT RULES:
1. **Output Structure:** You must return EXACTLY two lines.
   - Line 1: "Title: [Optimized Title]"
   - Line 2: "Features: [Single Long Descriptive Paragraph]"
2. **No Newlines:** Do NOT use line breaks (\n) inside the Features paragraph. The Features section must be one continuous line of text.
3. **Inline Formatting:** Instead of vertical lists, use inline full stop (.).
   - BAD: \n- Feature 1\n- Feature 2
   - GOOD: Feature 1 - Description. Feature 2 - Description.
"""
        # 2. Define the User Content (The Raw Data)
        user_msg = f"""
RAW DESCRIPTION:
{current_output}

REWRITE THIS:"""

        # 3. Call the Smart Model
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        
        rewritten = call_ollama_chat(
            messages=messages,
            model=TEACHER_MODEL,
            temperature=0.5
        )
        
        if rewritten and len(rewritten) > 20: # Basic validation
            item['output'] = rewritten
            new_data.append(item)
        else:
            # Fallback: If rewrite fails, keep original (safety net)
            # print(f"⚠️ Rewrite failed for item {i}. Keeping original.")
            new_data.append(item)

        # Periodic Save (Safety)
        if i % 50 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(new_data, f, indent=4)

    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_data, f, indent=4)
    
    print(f"✅ Success! Persuasive dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    rewrite_dataset()