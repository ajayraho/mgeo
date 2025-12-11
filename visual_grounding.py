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
        Calculates Cosine Similarity between Text and Image.
        Returns: Score 0.0 to 1.0
        """
        image = self._load_image(item_id, image_url)
        
        if not image:
            print(f"   ⚠️ VGS Warning: Image for {item_id} not found. Skipping Visual Check.")
            return 0.5 # Neutral score penalty for missing data
            
        # Truncate text to fit CLIP context (77 tokens)
        # We focus on the feature bullets as they contain the visual claims
        inputs = self.processor(
            text=[text[:300]], # CLIP has short context, grab the first chunk
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Similarity Score (logits_per_image is dot product)
        # We normalize it roughly to 0-1 range (CLIP raw scores vary)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1) # Not strictly necessary for 1-1 pairs
        
        # Raw cosine similarity is better for Utility
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        similarity = torch.matmul(text_embeds, image_embeds.t()).item()
        
        # Clip similarity is usually 0.2-0.3 for consistent pairs. 
        # We scale it to be more readable (Human-like 0-1)
        # This is a heuristic scaling for "Utility"
        adjusted_score = max(0, min(1, (similarity - 0.20) / 0.15))
        
        return round(adjusted_score, 4)