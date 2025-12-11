import json
import argparse
import os
from optimizer_agent import OptimizerAgent

# --- CONFIGURATION ---
CANDIDATES_FILE = "data/target_candidates.json"
REPO_FILE = "data/query_repository.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json" # Using the refined set
OUTPUT_FILE = "data/optimized_product.json"

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_full_product_data(repo_data, query_str, item_id):
    """Finds the full product object in the repository."""
    for q_obj in repo_data:
        # We try to match query, but item_id is unique enough usually
        if q_obj['query'] == query_str:
            for res in q_obj['results']:
                if res['item_id'] == item_id:
                    return res
    return None

def main():
    parser = argparse.ArgumentParser(description="Run MGEO Optimizer on a specific Target.")
    parser.add_argument("--q_idx", type=int, default=0, help="Index of the Query to select (0 to N)")
    parser.add_argument("--c_idx", type=int, default=0, help="Index of the Candidate Source to select (0 to N)")
    args = parser.parse_args()

    # 1. Load All Data
    print("🚀 Loading Phase 1 Data...")
    candidates_map = load_json(CANDIDATES_FILE)
    repo_data = load_json(REPO_FILE)
    captions = load_json(VISUALS_FILE)
    principles_data = load_json(PRINCIPLES_FILE)
    
    if not (candidates_map and repo_data and captions and principles_data):
        return

    # Extract the list of principles
    mgeo_rules = principles_data.get('mgeo_principles', [])
    if not mgeo_rules:
        # Fallback if structure is different
        mgeo_rules = principles_data.get('refined_principles', [])

    # 2. Select the Target
    queries = list(candidates_map.keys())
    if args.q_idx >= len(queries):
        print(f"❌ Query Index {args.q_idx} out of range (Max: {len(queries)-1})")
        return
    
    target_query = queries[args.q_idx]
    candidate_list = candidates_map[target_query]
    
    if args.c_idx >= len(candidate_list):
        print(f"❌ Candidate Index {args.c_idx} out of range (Max: {len(candidate_list)-1})")
        return

    target_candidate = candidate_list[args.c_idx]
    target_id = target_candidate['item_id']
    
    print(f"\n🎯 TARGET SELECTED:")
    print(f"   Query: {target_query}")
    print(f"   Item ID: {target_id}")
    print(f"   Current Rank: {target_candidate['current_rank']}")
    print(f"   Diagnosis: {target_candidate['diagnosis']}")

    # 3. Fetch Context
    product_data = get_full_product_data(repo_data, target_query, target_id)
    if not product_data:
        print("❌ Could not find full product data in Repository.")
        return
        
    visual_desc = captions.get(target_id, "")
    if not visual_desc:
        print("⚠️ Warning: No visual description found. Optimization may be weak.")

    # 4. Run Optimization
    agent = OptimizerAgent()
    result = agent.optimize_product(target_query, product_data, visual_desc, mgeo_rules)

    if result:
        print("\n✨ OPTIMIZATION SUCCESSFUL!")
        print(f"   Old Title: {product_data['title'][:50]}...")
        print(f"   New Title: {result['optimized_title'][:50]}...")
        
        # 5. Create the "Intervention Object" (Clone + Update)
        optimized_product = product_data.copy()
        optimized_product['title'] = result['optimized_title']
        optimized_product['features'] = result['optimized_features']
        
        # Add metadata for the experiment record
        optimized_product['optimization_log'] = {
            "original_rank": target_candidate['current_rank'],
            "modifications": result['modifications_made'],
            "applied_query": target_query
        }
        
        # Save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(optimized_product, f, indent=4)
        print(f"   💾 Saved optimized product to {OUTPUT_FILE}")
        print("   Ready for A/B Testing.")

if __name__ == "__main__":
    main()