import json
import os
import argparse
from simulator_agent import SimulatorAgent

# --- CONFIGURATION ---
REPO_FILE = "data/query.json"
OPTIMIZED_FILE = "data/optimized_product.json"
OUTPUT_VERIFICATION = "data/verification_result.json"

def parse_hybrid_output(response_text):
    # Extract JSON
    json_part = response_text.split("---JSON_START---")[-1].split("---JSON_END---")[0]
    rankings = json.loads(json_part)
    
    # Extract Text
    text_part = response_text.split("---RESPONSE_START---")[-1].split("---RESPONSE_END---")[0].strip()
    
    return text_part, rankings
  
def format_rag_context(results_list):
    """
    Reusing the exact same formatting logic to ensure a fair test.
    """
    context_str = ""
    for item in results_list:
        origin_str = "Unknown"
        if isinstance(item.get('origin'), dict):
            origin_str = item['origin'].get('domain_name', 'Unknown')
            
        # Use existing social proof or default
        rating = item.get('sim_rating', 'N/A')
        reviews = item.get('sim_reviews', 0)
        
        social_proof = f"Rating: {rating}/5.0 ({reviews} verified reviews)"
            
        context_str += f"""
[Source ID: {item['item_id']}]
Category: {item['category']}
Title: {item['title']}
Brand/Domain: {origin_str}
{social_proof}
Features: {str(item['features'])[:800]}... 
--------------------------------------------------
"""
    return context_str

def run_verification():
    # 1. Load the Optimized Patient
    if not os.path.exists(OPTIMIZED_FILE):
        print(f"❌ Error: {OPTIMIZED_FILE} not found. Run optimizer first.")
        return

    with open(OPTIMIZED_FILE, 'r') as f:
        new_product = json.load(f)
    
    # Extract Metadata
    target_query = new_product['optimization_log']['applied_query']
    target_id = new_product['item_id']
    old_rank = new_product['optimization_log']['original_rank']
    
    print(f"🚀 Verifying Optimization for '{target_id}'...")
    print(f"   Query: {target_query}")
    print(f"   Old Rank: {old_rank}")

    # 2. Load the Competition (The Original Context)
    with open(REPO_FILE, 'r') as f:
        repo = json.load(f)
        
    # Find the specific query group
    query_group = next((q for q in repo if q['query'] == target_query), None)
    if not query_group:
        print("❌ Original query group not found in repo.")
        return
        
    original_candidates = query_group['results']
    
    # 3. THE HOT SWAP (Replace Old with New)
    # We create a new list for the simulation so we don't corrupt the repo
    test_candidates = []
    
    for item in original_candidates:
        if item['item_id'] == target_id:
            print("   🔄 Swapping in Optimized Content...")
            # Inject the NEW text, but keep the OLD social proof/metadata
            # This ensures the ONLY variable changing is the Content (Title/Features)
            modified_item = item.copy()
            modified_item['title'] = new_product['title']
            modified_item['features'] = new_product['features']
            test_candidates.append(modified_item)
        else:
            test_candidates.append(item)

    # 4. Run the Simulation
    agent = SimulatorAgent(model_name="llama3")
    
    rag_context = format_rag_context(test_candidates)
    
    # Run ranking on the modified list
    print("   🤖 Running Simulator on New Context...")
    rank_output = agent.rank_products(target_query, rag_context, len(test_candidates))
    
    # 5. Analyze Results
    if rank_output and 'ranked_results' in rank_output:
        # Find new rank
        new_rank_obj = next((r for r in rank_output['ranked_results'] if r['item_id'] == target_id), None)
        
        if new_rank_obj:
            new_rank = new_rank_obj['rank']
            print(f"\n✨ RESULTS:")
            print(f"   Old Rank: {old_rank}")
            print(f"   New Rank: {new_rank}")
            
            improvement = old_rank - new_rank
            if improvement > 0:
                print(f"   📈 SUCCESS! Gained {improvement} spots.")
            elif improvement < 0:
                print(f"   📉 FAIL. Dropped {-improvement} spots.")
            else:
                print(f"   😐 NO CHANGE.")
                
            # Save Result
            result_log = {
                "item_id": target_id,
                "query": target_query,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "improvement": improvement,
                "reasoning": new_rank_obj.get('reason', 'N/A')
            }
            with open(OUTPUT_VERIFICATION, 'w') as f:
                json.dump(result_log, f, indent=4)
        else:
            print("❌ Error: Target item not found in ranking results.")
    else:
        print("❌ Simulation Failed.")

if __name__ == "__main__":
    run_verification()