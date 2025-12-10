import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = "data/images"          # Where you unzipped the images
OUTPUT_FILE = "data/dense_captions.json"
MODEL_ID = "llava-hf/llava-1.5-13b-hf" # The 13B Model (Best for Attributes)
BATCH_SAVE_INTERVAL = 50           # Save every 50 images to avoid data loss

# The Prompt is crucial. We force the model to be an "Attribute Scanner".
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
    # Load in float16 to fit in 32GB VRAM
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(0)
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("✅ Model Online.")
    return model, processor

def generate_caption(model, processor, image_path):
    try:
        # Load and ensure RGB
        image = Image.open(image_path).convert("RGB")
        
        # Prepare Inputs
        prompt = f"USER: <image>\n{SYSTEM_PROMPT}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

        # Generate (Greedy decoding is fine for factual extraction)
        output = model.generate(
            **inputs, 
            max_new_tokens=400,    # Enough for a dense paragraph
            do_sample=False,       # Deterministic = Better for facts
            temperature=0.0        # No creativity allowed
        )
        
        # Decode
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract just the Assistant's response
        # The prompt structure usually returns the whole thing, so we split
        response = decoded_output.split("ASSISTANT:")[-1].strip()
        return response
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    # 1. Setup
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: {IMAGE_DIR} not found. Unzip your images there first.")
        return
        
    model, processor = setup_model()
    
    # 2. Load Progress (Resume capability)
    captions = {}
    if os.path.exists(OUTPUT_FILE):
        print("📂 Found existing captions file. Resuming...")
        with open(OUTPUT_FILE, 'r') as f:
            captions = json.load(f)
            
    # 3. Scan Files
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    remaining_files = [f for f in all_files if f.split('.')[0] not in captions] # Match by ID
    
    print(f"Total Images: {len(all_files)}")
    print(f"Remaining:    {len(remaining_files)}")
    
    # 4. Processing Loop
    counter = 0
    for filename in tqdm(remaining_files, desc="Extracting Attributes"):
        item_id = filename.split('.')[0] # "B07XYZ.jpg" -> "B07XYZ"
        image_path = os.path.join(IMAGE_DIR, filename)
        
        description = generate_caption(model, processor, image_path)
        
        if description:
            captions[item_id] = description
            
        counter += 1
        
        # Periodic Save
        if counter % BATCH_SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(captions, f, indent=4)

    # Final Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(captions, f, indent=4)
        
    print(f"\n🎉 Extraction Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()