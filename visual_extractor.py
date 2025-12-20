import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import sys

# --- CONFIGURATION ---
IMAGE_DIR = "data/images"          # Where your images are stored
OUTPUT_FILE = "data/dense_captions.json"
MODEL_ID = "llava-hf/llava-1.5-13b-hf" 
BATCH_SAVE_INTERVAL = 10           # Save more frequently for safety

SYSTEM_PROMPT = """
You are a Visual Attribute Extractor for an E-Commerce AI.
Analyze this product image in extreme detail. 
Focus ONLY on visual features. Do not write a sales pitch.

EXTRACT THESE DETAILS:
1. Material & Texture (e.g., Suede, Ribbed, Glossy, Matte, Knit)
2. Pattern & Print (e.g., Floral, Geometric, Striped, Solid)
3. Color Nuances (e.g., Teal, Navy, Distressed Gold, rather than just "Blue")
4. Shape & Cut (e.g., High-top, V-neck, Chesterfield, Round-toe)
5. Distinctive Parts (e.g., Brass buckles, Wooden legs, Embroidery)

Output a dense, factual paragraph description.
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
        generate_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        # Cleanup response
        cleaned_text = output_text.split("ASSISTANT:")[-1].strip()
        return cleaned_text
        
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
    parser = argparse.ArgumentParser(description="Generate visual captions for products.")
    parser.add_argument("query_repo", help="Path to query_repo.json OR 'all' to process every image.")
    args = parser.parse_args()

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
        print(f"   Loaded {len(captions)} existing captions.")
    
    # 4. Map Directory Files (item_id -> filename)
    print("🔍 Scanning Image Directory...")
    available_files = {} # item_id -> filename
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Assuming filename is "B07XYZ.jpg" -> item_id = "B07XYZ"
            item_id = os.path.splitext(f)[0]
            available_files[item_id] = f

    # 5. Determine Processing Queue
    queue = []
    
    if args.query_repo.lower() == 'all':
        print("🌍 Mode: ALL. Processing entire directory.")
        # Process everything that isn't already captioned
        for item_id, filename in available_files.items():
            if item_id not in captions:
                queue.append((item_id, filename))
    else:
        print(f"🎯 Mode: TARGETED. Processing items from {args.query_repo}")
        target_ids = extract_target_ids(args.query_repo)
        
        # Filter: Must be in Target List AND Available in Folder AND Not already captioned
        for tid in target_ids:
            if tid in available_files:
                if tid not in captions:
                    queue.append((tid, available_files[tid]))
            else:
                # Silent skip or warn if image missing for a target item
                # tqdm.write(f"Warning: Image for target {tid} not found.")
                pass

    if not queue:
        print("✅ No new images to process. All targets are already captioned.")
        return

    print(f"📋 Processing Queue: {len(queue)} new images.")
    
    # 6. Load Model (Only if we have work to do)
    model, processor = setup_model()
    
    # 7. Processing Loop
    processed_count = 0
    
    try:
        for item_id, filename in tqdm(queue, desc="Generating Captions"):
            image_path = os.path.join(IMAGE_DIR, filename)
            description = generate_caption(model, processor, image_path)
            
            if description:
                captions[item_id] = description
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
        print(f"✅ Saved {len(captions)} total captions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()