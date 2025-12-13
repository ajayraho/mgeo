import json
import re
from ollama_utils import call_ollama 

class OptimizerAgent:
    def __init__(self, model_name="geo-optimizer"):
        self.model_name = model_name

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def _format_principles(self, principles):
        formatted = ""
        for i, p in enumerate(principles):
            formatted += f"PRINCIPLE #{i+1} [{p.get('rule_name', 'Strategy')}]:\n"
            formatted += f"   POLICY: {p['action_policy']}\n\n"
        return formatted

    # REMOVED: diagnosis argument
    def optimize_product(self, user_query, product_data, visual_desc, mgeo_principles):
        """
        Rewrites content applying MGEO principles autonomously.
        """
        rules_text = self._format_principles(mgeo_principles)
        
        # We give it the 'style' of the original data to mimic
        original_features = str(product_data.get('features', ''))[:200]
        
        prompt = f"""
### SYSTEM ROLE
You are an Elite GEO (Generative Engine Optimization) Specialist working as E-Commerce Copywriter.
Your task is to upgrade a product's text to ensure it gets **CITED** by AI Search Engines, maximizing its ranking in the Generative Engine Response.

### INPUT DATA
1. **Target Query:** "{user_query}"
2. **Visual Ground Truth (What the product actually looks like):** "{visual_desc}"
3. **Current Content:**
   - Title: {product_data.get('title', '')}
   - Features: {str(product_data.get('features', ''))}

### THE PLAYBOOK (MGEO PRINCIPLES)
You must rigorously apply these rules to bridge the gap between the Visual Truth and the Text:
{rules_text}

### CRITICAL CONSTRAINTS
1. **FORMATTING:** You MUST maintain the original format of the features (Pipe-separated `|` or Bullet points). Do NOT turn it into a paragraph.
2. **NO FLUFF:** Do not use marketing decorators like "Perfect for you," "Best choice," or "Buy now." 
3. **DENSITY:** Every phrase must add a specific visual fact (Material, Texture, Pattern, Shape).
4. **CONSTRAINT:** Do not invent features. Only describe what is in the "Visual Ground Truth".

### OUTPUT FORMAT (JSON ONLY)
{{
    "optimized_title": "The new, specific, attribute-rich title...",
    "optimized_features": "Feature 1 | Feature 2 | Feature 3...",
    "modifications_made": "Briefly explain which Principle you applied and why."
}}
"""
        print(f"   ✍️ Optimizer applying principles to '{product_data.get('item_id')}'...")
        try:
            response = call_ollama(prompt, model=self.model_name)
            return self._clean_json(response)
        except Exception as e:
            print(f"   ⚠️ Optimization Failed: {e}")
            return None