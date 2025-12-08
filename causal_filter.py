import json
import numpy as np

# --- CONFIGURATION ---
INPUT_LOGS = "data/simulation_logs.json"
REPO_FILE = "data/query.json"
BRAND_FILE = "data/brand_popularity.json"
OUTPUT_FILTERED = "data/causal_winners.json"

# BIAS WEIGHTS (Our Hypothesis)
W_LENGTH = 1.5
W_BRAND = 2.0 
W_RATING = 2.5  # High weight: We assume engine LOVES 5-star items

def calculate_propensity(text_len, brand_score, rating_val, max_len=2000):
    """
    Calculates P(Win | Bias).
    """
    # 1. Length Factor (0-1)
    norm_len = min(text_len / max_len, 1.0)
    
    # 2. Brand Factor (0-1) - Passed in
    
    # 3. Rating Factor (0-1)
    # We normalize 3.0-5.0 to 0.0-1.0 range.
    # If rating < 3.0, score is 0. If 5.0, score is 1.
    norm_rating = max(0, (rating_val - 3.0) / 2.0)
    
    # Logits
    logits = (W_LENGTH * norm_len) + (W_BRAND * brand_score) + (W_RATING * norm_rating)
    
    return 1 / (1 + np.exp(-logits))

def apply_causal_filter():
    # Load Data
    with open(INPUT_LOGS, 'r') as f: logs = json.load(f)
    with open(REPO_FILE, 'r') as f: repo = json.load(f)
    with open(BRAND_FILE, 'r') as f: brand_map = json.load(f)
    
    # Helper Map
    item_map = {}
    for q in repo:
        for res in q['results']:
            item_map[res['item_id']] = res
            
    causal_dataset = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    
    print(f"Applying 3-Factor IPS Filter...")
    
    for entry in logs:
        stats["total"] += 1
        rankings = entry['rankings']
        if not rankings: continue
        
        winner = rankings[0]
        loser = rankings[-1]
        
        w_data = item_map.get(winner['item_id'])
        if not w_data: continue
            
        # --- FACTORS ---
        # 1. Length
        w_len = len(str(w_data.get('features', '')))
        
        # 2. Brand (Dynamic Lookup)
        # Extract brand name logic (simplified for brevity)
        brand_key = "unknown" # Implement extraction logic here same as brand_analyzer
        w_brand_score = brand_map.get(brand_key, {}).get('popularity_score', 0.0)
        
        # 3. Rating (From Simulation Log)
        w_rating = w_data.get('rating', 4.0)
        
        # --- PROPENSITY ---
        e_score = calculate_propensity(w_len, w_brand_score, w_rating)
        
        # --- DECISION ---
        if e_score < 0.80: # Slightly stricter threshold
            causal_dataset.append({
                "query": entry['query'],
                "winner_id": winner['item_id'],
                "loser_id": loser['item_id'],
                "causal_weight": 1.0 / e_score
            })
            stats["kept"] += 1
            print(f"✅ KEEP: {w_data['title'][:20]} (P: {e_score:.2f})")
        else:
            stats["dropped"] += 1
            print(f"❌ DROP: {w_data['title'][:20]} (P: {e_score:.2f} - Bias Win)")

    with open(OUTPUT_FILTERED, 'w') as f:
        json.dump(causal_dataset, f, indent=4)
        
    print(f"\n--- REPORT ---")
    print(f"Total: {stats['total']} | Kept: {stats['kept']} | Dropped: {stats['dropped']}")

if __name__ == "__main__":
    apply_causal_filter()