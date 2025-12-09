import json
import hashlib
import os
import re
from ollama_utils import call_ollama 

# --- CONFIGURATION ---
PAIRS_FILE = "data/causal_pairs.json"
VISUAL_CAPTIONS = "data/dense_captions.json"
REPO_DATA = "data/query.json"
OUTPUT_RULES = "data/optimization_rules.json"

class ExplainerAgent:
    def __init__(self):
      pass

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def explain_pair(self, query, w_data, l_data, w_vis, l_vis):
        """
        Analyzes the Winner vs Loser to find the Causal Gap.
        """
        prompt = f"""
### SYSTEM ROLE
You are a Causal Search Analyst.
Investigate why a "Merit Winner" (Rank {w_data['rank']}) beat a "Loser" (Rank {l_data['rank']}).

### GOAL
Identify a **VISUAL ATTRIBUTE** that is:
1. Visually present in **BOTH** products (or at least the Loser).
2. Explicitly mentioned in the **WINNER'S TEXT**.
3. **MISSING** from the **LOSER'S TEXT**.

### EVIDENCE
**1. USER QUERY:** "{query}"

**2. WINNER (Rank {w_data['rank']})**
- **Text:** {str(w_data['features'])[:500]}
- **Visual Truth:** {w_vis}

**3. LOSER (Rank {l_data['rank']})**
- **Text:** {str(l_data['features'])[:500]}
- **Visual Truth:** {l_vis}

### OUTPUT JSON
Return a valid JSON object.
{{
    "found_gap": true,
    "relevant_attribute": "e.g. 'Floral Pattern'",
    "evidence": "Both images show Floral Pattern. Winner text mentions 'Floral', Loser text only says 'Multicolor'.",
    "rule": "IF Image contains 'Floral Pattern', INJECT 'Floral Pattern' into Title."
}}
If no clear gap exists, set "found_gap": false.
"""
        print(f"   prompting LLM for {query[:20]}...")
        try:
            response = call_ollama(prompt)
            return self._clean_json(response)
        except Exception as e:
            print(f"   ⚠️ LLM Error: {e}")
            return None

def load_existing_progress():
    """
    Loads existing rules to allow resuming execution.
    Returns: list of rules, set of processed pair IDs
    """
    if not os.path.exists(OUTPUT_RULES):
        return [], set()

    try:
        with open(OUTPUT_RULES, 'r') as f:
            rules = json.load(f)
        
        # Track which pairs have already generated a rule
        processed_ids = set()
        for r in rules:
            if 'source_pair' in r:
                processed_ids.add(r['source_pair'])
        
        return rules, processed_ids
    except:
        return [], set()

def run_explainer():
    if not os.path.exists(PAIRS_FILE):
        print("❌ Causal pairs not found.")
        return

    # Load Data
    with open(PAIRS_FILE, 'r') as f: grouped_pairs = json.load(f)
    with open(VISUAL_CAPTIONS, 'r') as f: captions = json.load(f)
    with open(REPO_DATA, 'r') as f: repo = json.load(f)
    
    # Fast Lookup
    item_map = {}
    for q in repo:
        for res in q['results']:
            item_map[res['item_id']] = res

    # --- RESUME LOGIC ---
    rules, processed_pairs_ids = load_existing_progress()
    print(f"🚀 Starting Explainer. Found {len(rules)} existing rules. Resuming...")

    agent = ExplainerAgent()
    seen_hashes = {hashlib.md5(r['rule'].encode()).hexdigest() for r in rules if 'rule' in r}

    total_groups = len(grouped_pairs)
    
    # Loop through ALL groups (Removed [:2] limit)
    for i, group in enumerate(grouped_pairs):
        query = group['query']
        pairs = group['pairs']
        
        print(f"\n📂 Processing Query Group {i+1}/{total_groups}: '{query}'")
        
        for pair in pairs:
            w_id = pair['winner_id']
            l_id = pair['loser_id']
            
            # Unique Signature for this comparison
            pair_signature = f"{w_id}_vs_{l_id}"
            
            # Skip if we already found a rule for this specific pair
            if pair_signature in processed_pairs_ids:
                # print(f"   ⏩ Skipping {pair_signature} (Already Processed)")
                continue

            # Prepare Data
            w_data = item_map.get(w_id)
            l_data = item_map.get(l_id)
            
            if not (w_data and l_data): continue
            
            # Enrich with Rank info from the pair data
            w_data['rank'] = pair['winner_rank']
            l_data['rank'] = pair['loser_rank']
            
            w_vis = captions.get(w_id, "No description")
            l_vis = captions.get(l_id, "No description")
            
            # Run Analysis
            insight = agent.explain_pair(query, w_data, l_data, w_vis, l_vis)
            
            if insight and insight.get('found_gap'):
                rule_text = insight['rule']
                h = hashlib.md5(rule_text.encode()).hexdigest()
                
                if h not in seen_hashes:
                    # Add provenance metadata
                    insight['source_query'] = query
                    insight['source_pair'] = pair_signature
                    
                    rules.append(insight)
                    seen_hashes.add(h)
                    processed_pairs_ids.add(pair_signature)
                    
                    print(f"   💡 New Rule: {rule_text}")
                    
                    # --- INCREMENTAL SAVE ---
                    # Save immediately after finding a rule
                    with open(OUTPUT_RULES, 'w') as f:
                        json.dump(rules, f, indent=4)
                    print(f"      💾 Saved progress ({len(rules)} rules total)")
            else:
                 print("      (No gap found)")

    print(f"\n✅ Exploration Complete. Total Unique Rules: {len(rules)}")
    print(f"   Final save to {OUTPUT_RULES}")

if __name__ == "__main__":
    run_explainer()