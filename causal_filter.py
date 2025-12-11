import json
import numpy as np

# --- CONFIGURATION ---
INPUT_LOGS = "data/simulation_logs.json"
REPO_FILE = "data/query.json"
BRAND_FILE = "data/brand_popularity.json"
OUTPUT_PAIRS = "data/causal_pairs.json"

# Tuned Weights
W_LENGTH = 0.8
W_BRAND = 1.2
W_RATING = 1.2
THRESHOLD = 0.90 

def calculate_propensity(text_len, brand_score, rating_val, max_len=2000):
    norm_len = min(text_len / max_len, 1.0)
    norm_rating = max(0, (rating_val - 3.0) / 2.0)
    logits = (W_LENGTH * norm_len) + (W_BRAND * brand_score) + (W_RATING * norm_rating)
    return 1 / (1 + np.exp(-logits))

def get_brand_score(item_data, brand_map):
    specs = item_data.get('specifications') or {}
    brand = specs.get('brand')
    if not brand:
        brand = " ".join(item_data['title'].split()[:2])
    brand_key = str(brand).lower().replace("amazon brand - ", "").strip()
    if brand_key in brand_map:
        return brand_map[brand_key]['popularity_score']
    return 0.0

def apply_pairwise_filter():
    try:
        with open(INPUT_LOGS, 'r') as f: logs = json.load(f)
        with open(REPO_FILE, 'r') as f: repo = json.load(f)
        with open(BRAND_FILE, 'r') as f: brand_map = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Missing File: {e}")
        return

    item_map = {}
    for q in repo:
        for res in q['results']:
            item_map[res['item_id']] = res

    grouped_causal_data = []
    stats = {"total_queries": 0, "pairs_generated": 0}

    print(f"⚖️ Applying Causal Filter (Derived Rank from Visibility)...")

    for entry in logs:
        stats["total_queries"] += 1
        query = entry['query']
        raw_rankings = entry['rankings'] 
        
        if not raw_rankings: continue

        # --- STEP 1: Derive Ranks from Visibility ---
        # Sort by Visibility Score (High to Low) to establish the "True Rank"
        # If scores are tied, it keeps original order
        sorted_items = sorted(raw_rankings, key=lambda x: x.get('visibility_score', 0), reverse=True)
        
        # Inject the derived rank into the local objects
        # Rank 1 = Index 0 + 1
        for rank_idx, item in enumerate(sorted_items):
            item['derived_rank'] = rank_idx + 1

        # --- STEP 2: Calculate Propensities ---
        item_props = {} 
        item_vis = {}
        
        for item in sorted_items:
            item_id = item['item_id']
            full_data = item_map.get(item_id)
            if not full_data: continue

            w_len = len(str(full_data.get('features', '')))
            w_brand = get_brand_score(full_data, brand_map)
            w_rating = full_data.get('sim_rating', 4.0)
            
            p_score = calculate_propensity(w_len, w_brand, w_rating)
            item_props[item_id] = p_score
            item_vis[item_id] = item.get('visibility_score', 0.0)

        # --- STEP 3: Generate Pairs ---
        query_pairs = []
        
        for i in range(len(sorted_items)):
            for j in range(i + 1, len(sorted_items)):
                winner = sorted_items[i]
                loser = sorted_items[j]
                
                w_id = winner['item_id']
                l_id = loser['item_id']
                
                if w_id not in item_props or l_id not in item_props: continue
                
                w_vis = item_vis[w_id]
                l_vis = item_vis[l_id]
                w_prop = item_props[w_id]
                
                # --- VISIBILITY LOGIC ---
                # 1. Winner must be visible (> 0.1)
                if w_vis < 0.1: continue
                
                # 2. Significant Gap required (Winner is clearly preferred)
                if (w_vis - l_vis) < 0.2: continue
                
                # 3. Causal Gatekeeper (Winner must be Low Bias / Merit Winner)
                if w_prop < THRESHOLD:
                    query_pairs.append({
                        "winner_id": w_id,
                        "loser_id": l_id,
                        
                        # --- FIX: Use the Derived Rank ---
                        "winner_rank": winner['derived_rank'],
                        "loser_rank": loser['derived_rank'],
                        
                        "winner_vis": w_vis,
                        "loser_vis": l_vis,
                        "winner_propensity": w_prop,
                        "weight": 1.0 / w_prop
                    })
                    stats["pairs_generated"] += 1
        
        if query_pairs:
            grouped_causal_data.append({
                "query": query,
                "pairs": query_pairs
            })

    with open(OUTPUT_PAIRS, 'w') as f:
        json.dump(grouped_causal_data, f, indent=4)

    print(f"\n--- REPORT ---")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Generated Pairs: {stats['pairs_generated']}")
    print(f"Saved to {OUTPUT_PAIRS}")

if __name__ == "__main__":
    apply_pairwise_filter()