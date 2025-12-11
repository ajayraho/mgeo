import json
import os
import math
import re
from simulator_agent import SimulatorAgent 

# --- CONFIGURATION ---
REPO_FILE = "data/query.json"
OUTPUT_LOG = "data/simulation_logs.json"

def format_rag_context(results_list):
    """Formats the text context for the Simulator."""
    context_str = ""
    for item in results_list:
        origin_str = "Unknown"
        if isinstance(item.get('origin'), dict):
            origin_str = item['origin'].get('domain_name', 'Unknown')
        
        rating = item.get('sim_rating', item.get('rating', 0))
        reviews = item.get('sim_reviews', item.get('reviews', 0))
        social_proof = f"Rating: {rating}/5.0 ({reviews} verified reviews)"
            
        context_str += f"""
[Source ID: {item['item_id']}]
Category: {item['category']}
Title: {item['title']}
Brand/Domain: {origin_str}
{social_proof}
Features: {str(item['features'])[:1500]}
--------------------------------------------------
"""
    return context_str

def calculate_visibility_score(generated_text, item_id):
    """Calculates Impression Score (WordPos)."""
    if not generated_text: return 0.0
    
    sentences = re.split(r'(?<=[.!?]) +', generated_text)
    total_score = 0.0
    
    for i, sent in enumerate(sentences):
        if item_id in sent:
            # Decay factor
            pos_weight = math.exp(-1 * i / max(len(sentences), 1))
            # Count factor
            citation_count = len(re.findall(r'\[', sent)) or 1
            
            total_score += (1.0 * pos_weight) / citation_count
            
    return round(total_score, 4)

def run_simulation_loop():
    if not os.path.exists(REPO_FILE):
        print("❌ Repo file not found.")
        return

    with open(REPO_FILE, 'r') as f:
        query_repo = json.load(f)
        
    agent = SimulatorAgent(model_name="llama3") 
    simulation_logs = []
    
    print(f"🚀 Starting Two-Step Simulation on {len(query_repo)} queries...")
    
    for i, case in enumerate(query_repo):
        query = case['query']
        candidates = case['results']
        
        print(f"\n--- Simulation {i+1}: '{query}' ---")
        rag_context = format_rag_context(candidates)
        
        # --- STEP 1: GENERATE TEXT ---
        gen_text = agent.generate_response(query, rag_context)
        if not gen_text:
            print("   ⚠️ Generation failed.")
            continue

        # --- INTERMEDIATE: CALCULATE SCORES ---
        scored_candidates = []
        for cand in candidates:
            vid = cand['item_id']
            v_score = calculate_visibility_score(gen_text, vid)
            scored_candidates.append({
                "item_id": vid,
                "visibility_score": v_score
            })
            
        # --- STEP 2: AUDIT/EXPLAIN ---
        audit_data = agent.analyze_visibility(query, rag_context, gen_text)
        
        # Merge Audit Data with Scores
        final_rankings = []
        if audit_data and 'analysis' in audit_data:
            analysis_map = {x['item_id']: x for x in audit_data['analysis']}
            
            for scorer in scored_candidates:
                audit_info = analysis_map.get(scorer['item_id'], {})
                merged = {
                    "item_id": scorer['item_id'],
                    "visibility_score": scorer['visibility_score'],
                    "perceived_relevance": audit_info.get('perceived_relevance', 0),
                    "reason": audit_info.get('reason_for_coverage', 'No explanation provided.')
                }
                final_rankings.append(merged)
        else:
            print("   ⚠️ Audit failed. Saving scores only.")
            final_rankings = scored_candidates

        # Save Log
        simulation_logs.append({
            "query": query,
            "generated_response": gen_text, 
            "rankings": final_rankings
        })
        
        # Sort by Visibility to show winner
        final_rankings.sort(key=lambda x: x.get('visibility_score', 0), reverse=True)
        top_1 = final_rankings[0]
        print(f"   🥇 Top Visible: {top_1['item_id']} (Score: {top_1['visibility_score']})")

    with open(OUTPUT_LOG, 'w') as f:
        json.dump(simulation_logs, f, indent=4)
        
    print(f"\n✅ Simulation Complete. Logs saved to {OUTPUT_LOG}")

if __name__ == "__main__":
    run_simulation_loop()