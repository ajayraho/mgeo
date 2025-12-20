import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import requests
from io import BytesIO

# --- CONFIGURATION ---
IMAGE_DIR = "data/images"  # Path where you store product images (B07....jpg)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

class VisualGroundingScorer:
    def __init__(self):
        print(f"👁️ Initializing CLIP Utility Judge ({CLIP_MODEL_ID})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    def _load_image(self, item_id, image_url=None):
        """
        Tries to load image from local disk, then URL.
        """
        # 1. Local Check
        local_path = os.path.join(IMAGE_DIR, f"{item_id}.jpg")
        if os.path.exists(local_path):
            return Image.open(local_path)
        
        # 2. URL Check
        if image_url:
            try:
                response = requests.get(image_url, timeout=5)
                return Image.open(BytesIO(response.content))
            except:
                pass
                
        return None

    def calculate_vgs(self, item_id, text, image_url=None):
        """
        Calculates Cosine Similarity between Text and Image using Sliding Window.
        Returns: Score 0.0 to 1.0 (Max across chunks)
        """
        image = self._load_image(item_id, image_url)
        
        if not image:
            # print(f"   ⚠️ VGS Warning: Image for {item_id} not found. Skipping Visual Check.")
            return 0.5 # Neutral score penalty for missing data

        # --- SLIDING WINDOW LOGIC ---
        # 1. Tokenize full text first to handle splitting correctly
        inputs = self.processor.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids'][0] # Flatten to 1D tensor

        # 2. Define Window Parameters
        window_size = 77  # CLIP limit
        stride = 50       # Overlap to catch phrases cut in half
        chunks = []

        # 3. Create Chunks
        if len(input_ids) <= window_size:
            # Short text: Take it all
            chunks.append(text)
        else:
            # Long text: Slide
            for i in range(0, len(input_ids), stride):
                chunk_ids = input_ids[i : i + window_size]
                if len(chunk_ids) < 10: continue # Skip tiny fragments at end
                
                # Decode back to string for the processor
                chunk_str = self.processor.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                chunks.append(chunk_str)

        # 4. Score Each Chunk
        chunk_scores = []
        
        for chunk_text in chunks:
            inputs = self.processor(
                text=[chunk_text], 
                images=image, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Raw cosine similarity
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            similarity = torch.matmul(text_embeds, image_embeds.t()).item()
            chunk_scores.append(similarity)

        # 5. Aggregation (MAX Pooling)
        # We take the BEST matching chunk as the representatitve score
        best_similarity = max(chunk_scores) if chunk_scores else 0.0
        
        # 6. Normalize
        # Clip similarity is usually 0.2-0.3 for consistent pairs. 
        # We scale it to be more readable (Human-like 0-1)
        adjusted_score = max(0, min(1, (best_similarity - 0.20) / 0.15))
        
        return round(adjusted_score, 4)