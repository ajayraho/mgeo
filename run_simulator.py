import json
import os
from simulator_agent import SimulatorAgent 

# --- CONFIGURATION ---
REPO_FILE = "data/query.json"
OUTPUT_LOG = "data/simulation_logs.json"
def smart_truncate(text, max_chars=2000):
    """Truncates text to max_chars, ensuring we don't cut words in half."""
    if not text: return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    
    # Cut to limit
    cut_text = text[:max_chars]
    # Backtrack to the last space to avoid cutting a word "flowe..."
    last_space = cut_text.rfind(' ')
    if last_space != -1:
        cut_text = cut_text[:last_space]
    
    return cut_text + " [TRUNCATED]"

def format_rag_context(results_list):
    """
    Formats the 'Blind' text context for the Simulator.
    """
    context_str = ""
    for item in results_list:
        # Extract metadata
        origin_str = "Unknown"
        if isinstance(item.get('origin'), dict):
            origin_str = item['origin'].get('domain_name', 'Unknown')

        # --- SYNTHETIC SOCIAL PROOF INJECTION ---
        rating = item.get('rating', 0)
        reviews = item.get('reviews', 0)
        
        social_proof_str = f"Rating: {rating}/5.0 ({reviews} verified reviews)"
            
        context_str += f"""
[Source ID: {item['item_id']}]
Category: {item['category']}
Title: {item['title']}
{social_proof_str}
Features: {(smart_truncate(item['features'])).replace(' | ', ',')}... 
--------------------------------------------------
"""
    # removed Origin/Domain: {origin_str}
    return context_str

def run_simulation_loop():
    if not os.path.exists(REPO_FILE):
        print(f"❌ Error: {REPO_FILE} not found.")
        return

    with open(REPO_FILE, 'r') as f:
        query_repo = json.load(f)
        
    print(f"🚀 Loaded {len(query_repo)} queries.")
    agent = SimulatorAgent() 
    
    simulation_logs = []
    
    for i, case in enumerate(query_repo):
        query = case['query']
        candidates = case['results']
        num_candidates = len(candidates)
        
        print(f"\n--- Simulation {i+1}/{len(query_repo)}: '{query}' ---")
        
        # 1. Format Context
        rag_context = format_rag_context(candidates)
        
        # 2. Run Simulator
        # You need to implement your actual call_ollama inside the class
        rank_output = agent.rank_products(query, rag_context, num_candidates)
        
        if rank_output and 'ranked_results' in rank_output:
            # 3. Save Lightweight Log
            # We ONLY keep the ranking decision. 
            # The full candidate data stays in query_repository.json
            simulation_logs.append({
                "query_id": i, # Simple ID to link back to repo
                "query": query,
                "rankings": rank_output['ranked_results'] 
            })
            
            # Print top 1 for sanity check
            top_1 = rank_output['ranked_results'][0]
            print(f"   🥇 Rank 1: {top_1['item_id']} ({top_1['reason'][:50]}...)")
            
        else:
            print("   ⚠️ Simulation failed (No JSON returned).")

    # 4. Save to Disk
    with open(OUTPUT_LOG, 'w') as f:
        json.dump(simulation_logs, f, indent=4)
        
    print(f"\n✅ Simulation Complete. Logs saved to {OUTPUT_LOG}")

if __name__ == "__main__":
    run_simulation_loop()