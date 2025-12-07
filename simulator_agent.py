import json
import re
from ollama_utils import call_ollama 

class SimulatorAgent:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def _clean_json(self, response_text):
        try:
            return json.loads(response_text)
        except:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def rank_products(self, user_query, rag_context, num_candidates):
        """
        Simulates a Real-World Generative Engine with Scientifically Documented Biases.
        """
        # removed brands/domain from authority bias
        prompt = f"""
### SYSTEM ROLE
You are a Realistic Generative Search Engine (like ChatGPT Search).
Your goal is to rank products for the User Query: "{user_query}"

### BIAS CONFIGURATION (Research Parameters)
Apply these realistic biases to your ranking logic:
1. **AUTHORITY BIAS:** Prefer recognized brands (e.g., 'AmazonBasics', 'Sony') over unknowns.
2. **VERBOSITY BIAS:** Prefer detailed, lengthy descriptions over short ones.
3. **PRIMACY BIAS:** Trust the order of results provided, but re-rank if relevance is clearly different.

### CONSTRAINTS
1. **BLINDNESS:** Rely ONLY on text. Do not hallucinate visual features.
2. **EXHAUSTIVE RANKING:** You MUST rank ALL {num_candidates} provided sources from 1 to {num_candidates}. Do not skip any.
3. **REASONING:** Briefly explain why you assigned this rank.

### SEARCH RESULTS
{rag_context}

### OUTPUT FORMAT (JSON ONLY)
Return a valid JSON object containing a SINGLE list "ranked_results" sorted by rank.
{{
    "ranked_results": [
        {{"rank": 1, "item_id": "...", "reason": "Strong keyword match + Trusted Brand"}},
        {{"rank": 2, "item_id": "...", "reason": "..."}},
        ...
        {{"rank": {num_candidates}, "item_id": "...", "reason": "Weak text description"}}
    ]
}}
"""
        # Call Ollama
        response_text = call_ollama(prompt)
        
        try:
            return self._clean_json(response_text)
        except Exception as e:
            print(f"❌ Simulator Failed: {e}")
            return None