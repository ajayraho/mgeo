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
THRESHOLD = 0.86

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

    # --- CHANGED: Using a list of Query Groups ---
    grouped_causal_data = []
    
    stats = {"total_queries": 0, "pairs_generated": 0}

    print(f"⚖️ Applying Pairwise Causal Filter (Hierarchical)...")

    for entry in logs:
        stats["total_queries"] += 1
        query = entry['query']
        rankings = entry['rankings'] 
        
        if not rankings or len(rankings) < 2: continue

        # 1. Calc Propensities
        item_props = {} 
        for rank_item in rankings:
            item_id = rank_item['item_id']
            full_data = item_map.get(item_id)
            if not full_data: continue

            w_len = len(str(full_data.get('features', '')))
            w_brand = get_brand_score(full_data, brand_map)
            # Use 'sim_rating' from Repo data, fallback to 4.0
            w_rating = full_data.get('sim_rating', 4.0)
            
            p_score = calculate_propensity(w_len, w_brand, w_rating)
            item_props[item_id] = p_score

        # 2. Generate Pairs
        query_pairs = []
        
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                winner = rankings[i]
                loser = rankings[j]
                
                w_id = winner['item_id']
                l_id = loser['item_id']
                
                if w_id not in item_props or l_id not in item_props: continue
                
                w_prop = item_props[w_id]
                
                # Filter: Winner must be Low Bias
                if w_prop < THRESHOLD:
                    query_pairs.append({
                        "winner_id": w_id,
                        "loser_id": l_id,
                        "winner_rank": winner['rank'],
                        "loser_rank": loser['rank'],
                        "winner_propensity": w_prop,
                        "weight": 1.0 / w_prop
                    })
                    stats["pairs_generated"] += 1
        
        # Only add the query group if valid pairs were found
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
    print(f"Saved grouped data to {OUTPUT_PAIRS}")

if __name__ == "__main__":
    apply_pairwise_filter()