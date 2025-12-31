import json
import os
from tqdm import tqdm
from ollama_utils import call_ollama_chat

# --- CONFIGURATION ---
GPU_ID=3
INPUT_FILE = f"data/test_dense_captions.json"
OUTPUT_FILE = f"data/test_dense_captions_refined_{GPU_ID}.json"
MODEL_NAME = "gpt-oss"  # or 'llama3', whatever your high-quality local model is

# --- STRICT SYSTEM PROMPT ---
sys_msg = """
You are a Technical Editor for an E-Commerce Database.
Your task is to MERGE multiple visual descriptions of the SAME product into a single, cohesive Master Description.

RULES:
1. **ELIMINATE NOISE:** Remove all references to "The image features...", "A close-up view of...", "background", "model", "woman", "man", "person holding".
2. **FOCUS ON PRODUCT:** Focus on product.
3. **NO HALLUCINATION:** Do not invent features not listed in the inputs.
4. **MERGE INTELLIGENTLY:**
   - If Input A says "Red dress" and Input B says "Silk fabric", output "Red silk dress".
   - If inputs repeat the same info, say it ONCE.
5. **OUTPUT FORMAT:** A single detailed paragraph. No bullet points. No intro text. Don't collapse everything, don't miss any visual detail about the image. Try including as many details as you can ONLY FROM THE INPUTS. DO NOT WRITE ANYTHING THAT IS NOT IN THE INPUT. Avoid condensing too much.

Example Input:
- "Image 1: A woman wearing a blue denim jacket with silver buttons."
- "Image 2: Close up of denim texture showing yellow stitching."

Example Output:
"A blue denim jacket featuring silver buttons and contrasting yellow stitching with a visible denim grain texture."
"""

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def refine_captions():
    # 1. Load Data
    dense_data = load_json(INPUT_FILE)
    refined_data = load_json(OUTPUT_FILE)
    global GPU_ID
    GPU_ID = GPU_ID-1
    dense_data = dict(list(dense_data.items())[len(dense_data)//3*GPU_ID : len(dense_data)//3*(GPU_ID+1) if GPU_ID != 2 else len(dense_data)])

    if not dense_data:
        print("❌ No dense captions found to refine.")
        return

    print(f"📂 Loaded {len(dense_data)} products from {INPUT_FILE}")
    print(f"📂 Loaded {len(refined_data)} already refined products.")

    # 2. Identify Work Items
    work_queue = []
    
    for item_id, images_dict in dense_data.items():
        current_image_count = len(images_dict)
        
        # Check if we need to process this item
        needs_processing = False
        
        if item_id not in refined_data:
            needs_processing = True
        else:
            # Smart Resume: Check if we have more images now than when we last refined
            last_count = refined_data[item_id].get("image_count_processed", 0)
            if current_image_count != last_count:
                print(f"   🔄 Update detected for {item_id}: {last_count} -> {current_image_count} images")
                needs_processing = True

        if needs_processing:
            work_queue.append((item_id, images_dict))

    if not work_queue:
        print("✅ All captions are up to date. No refining needed.")
        return

    print(f"🚀 Refining {len(work_queue)} products...")

    # 3. Processing Loop
    for item_id, images_dict in tqdm(work_queue, desc="Merging Captions"):
        
        # A. Collect all raw captions
        # We prefer PRODUCT_SOLO, but we take everything to be safe.
        raw_texts = []
        for img_key, details in images_dict.items():
            
            # Skip errors or empty captions
            if not isinstance(details, dict) or 'caption' not in details:
                continue
                
            c_type = details.get('type', 'UNKNOWN')
            caption = details.get('caption', '').strip()
            
            # Skip useless "No caption" entries
            # if len(caption) < 5: 
                # continue

            # Tag it for the LLM so it knows the source context (optional, but helps)
            # Actually, per your instruction, we just want the LLM to merge. 
            # We filter OUT purely "INFOGRAPHIC" text if it's just measurement numbers, 
            # but usually, we pass it all and let the System Prompt filter "Noise".
            raw_texts.append(f"- [{c_type}] {caption}")

        if not raw_texts:
            continue

        # B. Construct User Prompt
        joined_captions = "\n".join(raw_texts)
        user_msg = f"""
Here are the visual descriptions derived from multiple images of Item ID: {item_id}.
Merge them into one strictly physical product description.

INPUTS:
{joined_captions}

MERGED DESCRIPTION:
"""

        # C. Call LLM
        try:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ]
            merged_caption = call_ollama_chat(
                messages=messages,
                temperature=0.1 # Very low temp for strict merging
            )
            
            # Clean any potential quotes added by LLM
            merged_caption = merged_caption.strip().strip('"')

            # D. Update Data
            refined_data[item_id] = {
                "caption": merged_caption,
                "image_count_processed": len(images_dict) # Track this for resume logic
            }

            # Save incrementally (every item or every 10 items)
            # For safety, we save every item since LLM calls are slow.
            save_json(refined_data, OUTPUT_FILE)

        except Exception as e:
            print(f"   ⚠️ Error refining {item_id}: {e}")

    print(f"\n✅ Refinement Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    refine_captions()