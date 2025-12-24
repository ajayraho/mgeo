import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import sys
import re

# --- CONFIGURATION ---
IMAGE_DIR = "data/images"          # Where your images are stored
OUTPUT_FILE = "data/dense_captions.json"
MODEL_ID = "llava-hf/llava-1.5-13b-hf" 
BATCH_SAVE_INTERVAL = 10           # Save more frequently for safety

# --- HYBRID PROMPT: Classification + Your Strict Extraction Rules ---
SYSTEM_PROMPT = """
You are a Visual Attribute Extractor for an E-Commerce AI.
Your task is to CLASSIFY the image type and then ANALYZE the product in extreme detail.

STEP 1: CLASSIFY the image into exactly one of these categories:
- [PRODUCT_SOLO]: The product is the main focus against a clean/plain/white background.
- [LIFESTYLE]: The product is shown in a real-world setting (e.g., room, outdoors).
- [MODEL]: A human model is wearing or holding the product.
- [INFOGRAPHIC]: Image contains mostly text, size charts, diagrams, or comparisons.

STEP 2: VISUAL EXTRACTION
Focus ONLY on visual features of the product itself. 
Ignore the background, packaging boxes (unless the product is the box), surrounding props, and models (unless describing fit).

EXTRACT THESE DETAILS:
1. Material & Texture (e.g., Suede, Ribbed, Glossy, Matte, Knit)
2. Pattern & Print (e.g., Floral, Geometric, Striped, Solid)
3. Color Nuances (e.g., Teal, Navy, Distressed Gold, rather than just "Blue")
4. Shape & Cut (e.g., High-top, V-neck, Chesterfield, Round-toe)
5. Distinctive Parts (e.g., Brass buckles, Wooden legs, Embroidery)
6. DEFECTS/DETAILS: Note specific fasteners (zippers, buttons), stitching styles, or surface patterns.

CONSTRAINT:
- Output the [TYPE] tag on the first line.
- Follow with a single, dense paragraph description.
- Use dry, clinical language (e.g., "The object is..." NOT "This lovely item...").
- Do NOT interpret the product's use (e.g., do not say "good for parties"). Focus only on appearance.
"""

def setup_model():
    print(f"🚀 Loading {MODEL_ID} to GPU...")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("✅ Model loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def generate_caption(model, processor, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Prepare Inputs
        prompt = f"USER: <image>\n{SYSTEM_PROMPT}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        # Cleanup response (Remove Prompt)
        raw_response = output_text.split("ASSISTANT:")[-1].strip()
        
        # --- PARSING LOGIC ---
        # We look for the [TYPE] tag at the start
        classification = "UNKNOWN"
        description = raw_response
        
        # Regex to find [TAG] at the start
        match = re.match(r"\[(PRODUCT_SOLO|LIFESTYLE|MODEL|INFOGRAPHIC)\]", raw_response)
        if match:
            classification = match.group(1) # Extract text inside brackets
            # Remove the tag from the description to keep your dense paragraph clean
            description = raw_response.replace(match.group(0), "").strip()
        
        return {
            "type": classification, 
            "caption": description
        }
        
    except Exception as e:
        print(f"   ⚠️ Error processing {image_path}: {e}")
        return None

def extract_target_ids(query_repo_path):
    """Parses a query repo JSON file to get all involved item_ids."""
    if not os.path.exists(query_repo_path):
        print(f"❌ Error: Query repo '{query_repo_path}' not found.")
        sys.exit(1)
        
    print(f"📂 Parsing target items from: {query_repo_path}")
    with open(query_repo_path, 'r') as f:
        repo_data = json.load(f)
        
    target_ids = set()
    for entry in repo_data:
        results = entry.get('results', [])
        for item in results:
            if 'item_id' in item:
                target_ids.add(item['item_id'])
                
    print(f"   Found {len(target_ids)} unique items in repo.")
    return target_ids

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Generate classified visual captions for products.")
    parser.add_argument("query_repo", help="Path to query_repo.json OR 'all' to process every image.")
    parser.add_argument("--flat_structure", action="store_true", help="Use old flat directory structure (images directly in root).")
    args = parser.parse_args()

    product_has_dir = not args.flat_structure

    # 2. Check Directories
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Image directory '{IMAGE_DIR}' not found.")
        return
        
    # 3. Load Existing Captions (Append Mode)
    captions = {}
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 Found existing captions file ({OUTPUT_FILE}). Loading...")
        with open(OUTPUT_FILE, 'r') as f:
            captions = json.load(f)
        print(f"   Loaded {len(captions)} existing entries.")
    
    # 4 & 5. Map Directory & Build Queue
    queue = [] # Format: (item_id, full_relative_path, sub_key_or_none)

    # Determine Targets
    if args.query_repo.lower() == 'all':
        print("🌍 Mode: ALL. Processing entire directory.")
        if product_has_dir:
            # Scan subdirectories
            target_ids = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
        else:
            # Scan files
            target_ids = [os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    else:
        print(f"🎯 Mode: TARGETED. Processing items from {args.query_repo}")
        target_ids = extract_target_ids(args.query_repo)

    print("🔍 Scanning Image Directory...")
    
    # --- NEW LOGIC: Directory per Product ---
    if product_has_dir:
        for tid in target_ids:
            product_folder = os.path.join(IMAGE_DIR, tid)
            
            if not os.path.isdir(product_folder):
                continue
            
            # Ensure output structure is dict for this item
            if tid not in captions or not isinstance(captions[tid], dict):
                pass 

            # Scan for all images inside
            for img_file in os.listdir(product_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_key = os.path.splitext(img_file)[0] # e.g., "0", "1"
                    
                    # Check if already processed
                    already_done = (
                        tid in captions and 
                        isinstance(captions[tid], dict) and 
                        img_key in captions[tid]
                    )
                    
                    if not already_done:
                        full_rel_path = os.path.join(tid, img_file)
                        queue.append((tid, full_rel_path, img_key))

    # --- OLD LOGIC: Flat Structure (Backward Compatibility) ---
    else:
        available_files = {} 
        for f in os.listdir(IMAGE_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                item_id = os.path.splitext(f)[0]
                available_files[item_id] = f
        
        for tid in target_ids:
            if tid in available_files:
                if tid not in captions:
                    queue.append((tid, available_files[tid], None))

    if not queue:
        print("✅ No new images to process. All targets are already captioned.")
        return

    print(f"📋 Processing Queue: {len(queue)} new images.")
    
    # 6. Load Model (Only if we have work to do)
    model, processor = setup_model()
    
    # 7. Processing Loop
    processed_count = 0
    
    try:
        for item_id, rel_path, sub_key in tqdm(queue, desc="Classifying & Captioning"):
            image_path = os.path.join(IMAGE_DIR, rel_path)
            
            # Generate the structured result
            result = generate_caption(model, processor, image_path)
            
            if result:
                # --- NEW OUTPUT FORMAT: Nested Dict ---
                if product_has_dir:
                    if item_id not in captions or not isinstance(captions[item_id], dict):
                        captions[item_id] = {}
                    captions[item_id][sub_key] = result # {type: "...", caption: "..."}
                
                # --- OLD OUTPUT FORMAT: Flat Dict ---
                else:
                    captions[item_id] = result # {type: "...", caption: "..."}
                
                processed_count += 1
            
            # Periodic Save
            if processed_count % BATCH_SAVE_INTERVAL == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(captions, f, indent=4)
                    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")
    finally:
        # Final Save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(captions, f, indent=4)
        print(f"✅ Saved {len(captions)} total items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()