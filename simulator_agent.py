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

    def generate_response(self, user_query, rag_context):
        """
        STEP 1: Pure Generation.
        Goal: Maximize citations and length to create a visibility gradient.
        """
        prompt = f"""
### SYSTEM ROLE
You are an Advanced Generative Search Engine (like Perplexity AI).

### TASK
Write a comprehensive **Buyer's Guide** for the User Query: "{user_query}".

### DATA SOURCES
{rag_context}

### REQUIREMENTS
1. **Compare and Contrast:** Discuss the pros and cons of the provided options.
2. **Citations:** You MUST cite your sources using `[Source ID]` whenever you mention a product's feature.
3. **Comprehensive:** Try to cover as many relevant products as possible to give the user a full market overview. 

You are biased - 
### BIAS CONFIGURATION
1. **AUTHORITY BIAS:** Prefer recognized brands.
2. **SOCIAL PROOF BIAS:** Trust high ratings.
3. **VERBOSITY BIAS:** Prefer detailed text.

### OUTPUT
Return ONLY the natural language response. Do not output JSON yet.
"""
        # print(f"   Generating text for '{user_query}'...")
        return call_ollama(prompt)

    def analyze_visibility(self, user_query, rag_context, generated_text):
        """
        STEP 2: Post-Hoc Analysis.
        Goal: Explain WHY certain items got high/low visibility.
        """
        prompt = f"""
### SYSTEM ROLE
You are a Search Algorithm Auditor.

### CONTEXT
1. **User Query:** "{user_query}"
2. **Available Sources:** {rag_context}

3. **Generated Response (The Output):**
```
{generated_text}
```

### TASK
Analyze why the Search Engine chose to cite specific products and ignore others in the "Generated Response".

### OUTPUT FORMAT (JSON ONLY)
Return a list of objects for EVERY source in the context.
{{
    "analysis": [
        {{
            "item_id": "...",
            "perceived_relevance": (1-10 Score),
            "reason_for_coverage": "Explain why this was cited (or ignored). Was it the brand? Rating? Lack of details?"
        }},
        ...
    ]
}}
"""
        # print(f"   Auditing visibility decisions...")
        est_tokens = len(prompt) / 3.0
        if est_tokens > 2048:
             print(f"\n⚠️ SIMULATOR PROMPT IS HUGE ({int(est_tokens)} tokens). Ensure num_ctx > {int(est_tokens)}!\n")
        response = call_ollama(prompt)
        return self._clean_json(response)