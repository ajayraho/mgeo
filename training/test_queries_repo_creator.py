import json
import pandas as pd
import os
import sys
import random
from tqdm import tqdm

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search_engine import LocalSearchEngine
from synthetic_reviews import SocialProofGenerator

# --- CONFIGURATION ---
DATA_FILE = "data/amazon_dataset.csv"           # The Source of Truth
UNSEEN_QUERIES_FILE = "data/unseen_queries.json"     # The 20 Hard Questions
OUTPUT_REPO = "data/test_repo.json"             # The New Environment
OUTPUT_CANDIDATES = "data/test_candidates.json" # The Losers to Fix

SAMPLES_PER_QUERY = 3  # Increase N for statistical significance (Total ~60 tests)

def parse_json_col(val):
    """Helper to safely parse JSON strings from CSV."""
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except:
        return val

def main():
    print("🚀 PHASE 1: GENERATING TEST SUITE (RICH ENVIRONMENTS)...")
    df = pd.read_csv(DATA_FILE)
    
    if df.empty:
        print("❌ Dataset empty.")
        return
    
    # 1. Validation
    if not os.path.exists(UNSEEN_QUERIES_FILE):
        print(f"❌ Error: {UNSEEN_QUERIES_FILE} not found.")
        return
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found. Search Engine needs data.")
        return
        
    # 2. Initialization
    print("   Initializing Search Engine & Social Proof Generator...")
    search_engine = LocalSearchEngine(df)
    proof_gen = SocialProofGenerator()
    
    with open(UNSEEN_QUERIES_FILE, 'r') as f:
        queries = json.load(f)
        
    test_repo = []
    test_candidates = {}
    
    print(f"   Simulating {len(queries)} Unseen Queries...")
    
    for query in tqdm(queries):
        # A. SEARCH (Find the "Before" state)
        # We get the DataFrame rows directly
        results_df = search_engine.search(query, top_k=20)
        
        if results_df.empty or len(results_df) < 5:
            tqdm.write(f"   ⚠️ Low results for '{query}'. Skipping.")
            continue
            
        results_list = []
        
        # B. ENRICH (Add Reviews & Metadata)
        for i, (_, row) in enumerate(results_df.iterrows()):
            origin_data = parse_json_col(row.get('origin'))
            specs_data = parse_json_col(row.get('other_attributes'))
            stars, reviews = proof_gen.generate()
            
            item_data = {
                "rank": i + 1,
                "item_id": row['item_id'],
                "relevance_score": float(row.get('relevance_score', 0.0)),
                "category": row.get('category', 'Unknown'),
                
                # Core Content
                "title": row['title'],
                "features": row['features'],
                
                # Context
                "origin": origin_data,
                "specifications": specs_data,
                
                # Social Proof
                "rating": stars,
                "reviews": reviews,
                
                # Visuals
                "image_path": row.get('path', ''),
                "main_image_url": row.get('main_image_url', '') # Fallback
            }
            results_list.append(item_data)
            
        # Save the Environment
        test_repo.append({
            "query": query,
            "results": results_list
        })
        
        # C. PICK LOSERS (Rank 11-20)
        # We assume products ranked 11-20 are the "Patients"
        available_losers = results_list[10:] 
        if not available_losers:
            available_losers = results_list[5:] # Fallback
        
        if not available_losers: continue

        # --- UPDATED SELECTION LOGIC ---
        # Pick 3 unique losers (or fewer if not enough exist)
        count = min(SAMPLES_PER_QUERY, len(available_losers))
        selected_targets = random.sample(available_losers, count)
        
        candidates_list = []
        for target in selected_targets:
            candidate_entry = {
                "item_id": target['item_id'],
                "title": target['title'],
                "features": target['features'],
                "current_rank": target['rank'],
                "current_vis": 0.0, # Losers have ~0 visibility
            }
            candidates_list.append(candidate_entry)
        
        test_candidates[query] = candidates_list
        
    # D. SAVE FILES
    with open(OUTPUT_REPO, 'w') as f:
        json.dump(test_repo, f, indent=4)
        
    with open(OUTPUT_CANDIDATES, 'w') as f:
        json.dump(test_candidates, f, indent=4)
        
    total_subjects = sum(len(x) for x in test_candidates.values())
    print(f"\n✅ Test Suite Created!")
    print(f"   - Environment: {OUTPUT_REPO}")
    print(f"   - Subjects: {OUTPUT_CANDIDATES}")
    print(f"   - Total Tests: {total_subjects} (approx 3 per query)")

if __name__ == "__main__":
    main()