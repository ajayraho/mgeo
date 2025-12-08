import json
import numpy as np

# --- CONFIGURATION ---
INPUT_LOGS = "data/simulation_logs.json"
REPO_FILE = "data/query.json"
BRAND_FILE = "data/brand_popularity.json"
OUTPUT_FILTERED = "data/causal_winners.json"

# --- TUNED WEIGHTS (Less Aggressive) ---
W_LENGTH = 0.8   # Was 1.5
W_BRAND = 1.2    # Was 2.0
W_RATING = 1.2   # Was 2.5 (The killer)
THRESHOLD = 0.86
def calculate_propensity(text_len, brand_score, rating_val, max_len=2000):
    # 1. Length Factor
    norm_len = min(text_len / max_len, 1.0)
    
    # 2. Rating Factor (Normalized)
    # Map 3.0 -> 0.0, 5.0 -> 1.0
    norm_rating = max(0, (rating_val - 3.0) / 2.0)
    
    # Logits
    logits = (W_LENGTH * norm_len) + (W_BRAND * brand_score) + (W_RATING * norm_rating)
    
    return 1 / (1 + np.exp(-logits))

def get_brand_score(item_data, brand_map):
    """Extracts Brand Score properly."""
    specs = item_data.get('specifications') or {}
    brand = specs.get('brand')
    
    if not brand:
        # Fallback to Title
        brand = " ".join(item_data['title'].split()[:2])
        
    # Clean string
    brand_key = str(brand).lower().replace("amazon brand - ", "").strip()
    
    # Lookup
    if brand_key in brand_map:
        return brand_map[brand_key]['popularity_score']
    return 0.0

def apply_causal_filter():
    with open(INPUT_LOGS, 'r') as f: logs = json.load(f)
    with open(REPO_FILE, 'r') as f: repo = json.load(f)
    with open(BRAND_FILE, 'r') as f: brand_map = json.load(f)
    
    item_map = {}
    for q in repo:
        for res in q['results']:
            item_map[res['item_id']] = res
            
    causal_dataset = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    
    print(f"Applying Tuned IPS Filter...")
    
    for entry in logs:
        stats["total"] += 1
        rankings = entry['rankings']
        if not rankings: continue
        
        winner = rankings[0]
        loser = rankings[-1]
        
        w_data = item_map.get(winner['item_id'])
        if not w_data: continue
            
        # --- FACTORS ---
        w_len = len(str(w_data.get('features', '')))
        w_brand_score = get_brand_score(w_data, brand_map)
        w_rating = w_data.get('rating', 4.0) # Correct key from Repo
        
        # --- PROPENSITY ---
        e_score = calculate_propensity(w_len, w_brand_score, w_rating)
        
        # --- DECISION ---
        # Relaxed Threshold: 0.90
        if e_score < THRESHOLD: 
            causal_dataset.append({
                "query": entry['query'],
                "winner_id": winner['item_id'],
                "loser_id": loser['item_id'],
                "causal_weight": 1.0 / e_score
            })
            stats["kept"] += 1
            print(f"✅ KEEP: {w_data['title'][:40]}... (P: {e_score:.2f})")
        else:
            stats["dropped"] += 1
            print(f"❌ DROP: {w_data['title'][:40]}... (P: {e_score:.2f} - Bias)")

    with open(OUTPUT_FILTERED, 'w') as f:
        json.dump(causal_dataset, f, indent=4)
        
    print(f"\n--- REPORT ---")
    print(f"Total: {stats['total']} | Kept: {stats['kept']} | Dropped: {stats['dropped']}")

if __name__ == "__main__":
    apply_causal_filter()