import json
import re
from ollama_utils import call_ollama 

class OptimizerAgent:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def _format_principles(self, principles):
        """Formats the MGEO rules into a strict instruction set."""
        formatted = ""
        for i, p in enumerate(principles):
            formatted += f"RULE #{i+1} [{p.get('rule_name', 'Strategy')}]:\n"
            formatted += f"   ACTION: {p['action_policy']}\n\n"
        return formatted

    def optimize_product(self, user_query, product_data, visual_desc, mgeo_principles):
        """
        Rewrites the Title and Features to align with Visual Truth + Query Intent.
        """
        
        rules_text = self._format_principles(mgeo_principles)
        
        prompt = f"""
### SYSTEM ROLE
You are an Elite E-Commerce Copywriter and SEO Specialist.
Your goal is to Rewrite a product's **Title** and **Feature Bullets** to maximize its ranking in a Generative Search Engine.

### INPUT DATA
1. **Target Query:** "{user_query}"
2. **Visual Ground Truth (What the product actually looks like):** "{visual_desc}"
3. **Current (Weak) Text:**
   - Title: {product_data.get('title', '')}
   - Features: {str(product_data.get('features', ''))[:1000]}

### THE PLAYBOOK (MGEO PRINCIPLES)
You MUST apply the following optimization rules to the text:
{rules_text}

### INSTRUCTIONS
1. **Title Optimization:** Rewrite the title to include specific visual attributes (Material, Texture, Pattern) found in the "Visual Ground Truth". Keep it natural but dense with attribute keywords.
2. **Feature Optimization:** Rewrite the bullet points. 
   - Inject missing visual details (Completeness).
   - Replace generic words with specific tactile ones (Specificity).
   - Align the tone with the visual mood (Atmosphere).
3. **Constraint:** Do NOT hallucinate. Only use details present in the "Visual Ground Truth" or the original text.

### OUTPUT FORMAT (JSON ONLY)
{{
    "optimized_title": "The new, specific, attribute-rich title...",
    "optimized_features": "Bullet 1 | Bullet 2 | Bullet 3...",
    "modifications_made": "Brief explanation of what you changed and why."
}}
"""
        print(f"   ✍️ Optimizer Agent rewriting '{product_data.get('item_id')}'...")
        try:
            response = call_ollama(prompt)
            return self._clean_json(response)
        except Exception as e:
            print(f"   ⚠️ Optimization Failed: {e}")
            return None