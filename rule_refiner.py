import json
import os
from ollama_utils import call_ollama 
import re

# --- CONFIGURATION ---
INPUT_FILE = "data/mgeo_principles.json"
OUTPUT_FILE = "data/mgeo_principles_refined.json"

class PolicyRefiner:
    def __init__(self):
        pass

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def refine(self, current_principles):
        """
        Takes a list of redundant principles and merges them into an Orthogonal Set.
        """
        
        # 1. Create a readable digest for the Editor
        digest = ""
        for i, p in enumerate(current_principles):
            digest += f"Principle #{i+1}: {p['strategy_name']}\n"
            digest += f"   - Policy: {p['action_policy']}\n\n"

        prompt = f"""
### SYSTEM ROLE
You are a Senior Editor for an AI System.
You have been handed a draft of "Optimization Rules" that is **little repetitive**.
Your job is to **Deduplicate** and **Sharpen** these rules.

### INPUT DATA (The Redundant Draft)
{digest}

### THE PROBLEM
Notice how many rules just say "Describe every attribute" or "Be specific." 
This is redundant. We need **Distinct, Orthogonal Strategies**.

### TASK
1. **Merge Duplicates:** If 3 rules say "List all attributes", combine them into ONE master rule named "Visual Completeness Protocol".
2. **Isolate Specifics:** If there is a rule specifically about **Texture** (Velvet/Suede) or **Pattern** (Floral/Striped), keep it separate, but make sure its policy is *strictly* about Texture/Pattern, not general completeness.
3. **Rewrite Policies:** Make the `action_policy` executable and distinct.
    - Bad: "Describe the image well."
    - Good: "Scan for specific tactile adjectives (e.g. 'Ribbed', 'Burnished'). If missing, inject them."

### OUTPUT FORMAT (JSON ONLY)
Return the cleaned list under "refined_principles".
{{
    "refined_principles": [
        {{
            "principle_id": "GEO_COMPLETENESS",
            "rule_name": "The Visual Completeness Axiom",
            "trigger": "Missing Object/Feature",
            "action_policy": "Enumeration Strategy: Identify every distinct visual object (buckles, straps, soles) visible in the image but missing from text. Inject them as a comma-separated feature list."
        }},
        {{
            "principle_id": "GEO_TEXTURE",
            "rule_name": "Tactile Specificity Protocol",
            "trigger": "Vague Material Terms",
            "action_policy": "Refinement Strategy: Replace generic material nouns (e.g. 'Cloth', 'Leather') with specific tactile descriptors observed in the image (e.g. 'Distressed Suede', 'Cable-knit Wool')."
        }}
    ]
}}
"""
        print(f"🧠 Refining {len(current_principles)} principles into an Orthogonal Set...")
        try:
            response = call_ollama(prompt)
            return self._clean_json(response)
        except Exception as e:
            print(f"Error refining: {e}")
            return None

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print("❌ Input file not found.")
    else:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
            raw_principles = data.get('mgeo_principles', [])
        
        if not raw_principles:
            print("❌ No principles found to refine.")
        else:
            refiner = PolicyRefiner()
            result = refiner.refine(raw_principles)
            
            if result and 'refined_principles' in result:
                final_list = result['refined_principles']
                
                # Wrap in the standard format
                output = {"mgeo_principles": final_list}
                
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(output, f, indent=4)
                
                print(f"✅ Refinement Complete.")
                print(f"   Collapsed {len(raw_principles)} redundant rules -> {len(final_list)} orthogonal strategies.")
                print(f"   Saved to {OUTPUT_FILE}")