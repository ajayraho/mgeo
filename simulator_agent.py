import json
import re
from ollama_utils import call_ollama 

class SimulatorAgent:
    def __init__(self):
        pass

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
You are an Advanced Generative Search Engine (like Perplexity AI).
You have two goals:
1. **Synthesize** a helpful, natural language answer for the user based *only* on the search results.
2. **Rank** the results based on relevance.

### INSTRUCTIONS
**PART 1: THE GENERATIVE RESPONSE**
- Write a direct answer to the User Query: "{user_query}".
- You MUST cite your claims using the format `[Source ID]`.
- Example: "For a vintage look, the [B07XYZ] is the best option due to its..."
- Mention and compare the top products naturally.

**PART 2: THE STRUCTURED RANKING**
- Output the strict ranking JSON as before.

### BIAS CONFIGURATION
1. **AUTHORITY BIAS:** Prefer recognized brands.
2. **SOCIAL PROOF BIAS:** Trust high ratings.
3. **VERBOSITY BIAS:** Prefer detailed text.

### SEARCH RESULTS
{rag_context}

### OUTPUT FORMAT
You must output the response in this exact format:

---RESPONSE_START---
(Write your natural language answer here with [citations])
---RESPONSE_END---

---JSON_START---
{{
    "ranked_results": [ ... ]
}}
---JSON_END---
"""
        # Call Ollama
        print("---")
        print("Prompt:\n")
        print(prompt)
        response_text = call_ollama(prompt)
        print("Response from LLM:")
        print(response_text[:200])
        try:
            return self._clean_json(response_text)
        except Exception as e:
            print(f"❌ Simulator Failed: {e}")
            return None