import json
import os
import argparse
import math
import re
from simulator_agent import SimulatorAgent

# --- CONFIGURATION ---
REPO_FILE = "data/query.json"
OPTIMIZED_FILE = "data/optimized_product.json"
OUTPUT_VERIFICATION = "data/verification_result.json"

def format_rag_context(results_list):
    """
    Standard formatting for the Simulator.
    """
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
    """
    Implements the Impression Score (WordPos).
    """
    if not generated_text: return 0.0
    
    sentences = re.split(r'(?<=[.!?]) +', generated_text)
    total_score = 0.0
    
    for i, sent in enumerate(sentences):
        if item_id in sent:
            # Decay factor (Earlier sentences matter more)
            pos_weight = math.exp(-1 * i / max(len(sentences), 1))
            
            # Count factor (Shared credit)
            citation_count = len(re.findall(r'\[', sent)) or 1
            
            total_score += (1.0 * pos_weight) / citation_count
            
    return round(total_score, 4)

def run_verification():
    # 1. Load Data
    if not os.path.exists(OPTIMIZED_FILE):
        print("❌ Optimized file not found.")
        return

    with open(OPTIMIZED_FILE, 'r') as f:
        new_product = json.load(f)
    
    # Extract Metadata
    target_query = new_product['optimization_log']['applied_query']
    target_id = new_product['item_id']
    old_rank = new_product['optimization_log'].get('original_rank', 99)
    old_vis = new_product['optimization_log'].get('original_vis', 0.0)
    
    print(f"🚀 Verifying Optimization for '{target_id}'...")
    print(f"   Query: {target_query}")
    print(f"   Baseline: Rank {old_rank} | Vis {old_vis}")

    # 2. Load the Competition
    if not os.path.exists(REPO_FILE):
        print("❌ Repo file missing.")
        return

    with open(REPO_FILE, 'r') as f:
        repo = json.load(f)
        
    query_group = next((q for q in repo if q['query'] == target_query), None)
    if not query_group:
        print("❌ Original query group not found.")
        return
        
    # 3. THE HOT SWAP
    test_candidates = []
    for item in query_group['results']:
        if item['item_id'] == target_id:
            print("   🔄 Swapping in Optimized Content...")
            modified_item = item.copy()
            modified_item['title'] = new_product['title']
            modified_item['features'] = new_product['features']
            test_candidates.append(modified_item)
        else:
            test_candidates.append(item)

    # 4. Run Hybrid Simulation (Two-Step Mode)
    agent = SimulatorAgent()
    rag_context = format_rag_context(test_candidates)
    
    print("   🤖 Running Simulator (Generation Step)...")
    
    # STEP 1: Generate Text
    gen_text = agent.generate_response(target_query, rag_context)
    
    if not gen_text:
        print("❌ Simulation Failed: No text generated.")
        return

    # STEP 2: Calculate Visibility for ALL items to determine Rank
    scored_candidates = []
    for item in test_candidates:
        v_score = calculate_visibility_score(gen_text, item['item_id'])
        scored_candidates.append({
            "item_id": item['item_id'],
            "visibility_score": v_score
        })
    
    # Sort by Visibility (Highest Score = Rank 1)
    scored_candidates.sort(key=lambda x: x['visibility_score'], reverse=True)
    
    # Find our Target's new position
    new_rank = 99
    new_vis_score = 0.0
    
    for i, candidate in enumerate(scored_candidates):
        if candidate['item_id'] == target_id:
            new_rank = i + 1 # 1-based index
            new_vis_score = candidate['visibility_score']
            break
            
    # 5. Analyze Results
    print(f"\n✨ FINAL VERIFICATION RESULTS:")
    print(f"   [Retrieval] Old Rank: {old_rank} -> New Rank: {new_rank}")
    print(f"   [Generative] Old Vis: {old_vis} -> New Vis: {new_vis_score}")
    
    # Success Logic
    success_msg = "😐 NO CHANGE"
    if new_vis_score > old_vis:
        success_msg = "🏆 GEO SUCCESS! Visibility Increased."
    elif new_rank < old_rank:
        success_msg = "📈 PARTIAL SUCCESS! Rank Improved (Latent Optimization)."
    elif new_vis_score == 0 and old_vis == 0:
        success_msg = "👻 STILL INVISIBLE."
        
    print(f"   Verdict: {success_msg}")

    # Save Result
    result_log = {
        "item_id": target_id,
        "query": target_query,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_vis": old_vis,
        "new_vis": new_vis_score,
        "rank_improvement": old_rank - new_rank,
        "vis_improvement": round(new_vis_score - old_vis, 4),
        "generated_text": gen_text
    }
    with open(OUTPUT_VERIFICATION, 'w') as f:
        json.dump(result_log, f, indent=4)
    print(f"   💾 Saved verification to {OUTPUT_VERIFICATION}")

if __name__ == "__main__":
    run_verification()