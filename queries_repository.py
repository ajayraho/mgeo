import json
import pandas as pd
from search_engine import LocalSearchEngine
from synthetic_reviews import SocialProofGenerator
# --- CONFIGURATION ---
OUTPUT_FILE = "data/query.json"
TOP_K = 10
DATA_FILE = "data/amazon_dataset.csv"

QUERIES = [
    "Women's floral embroidery slip-on shoes",
    "Men's leather oxford dress shoes black",
    "Bohemian style multi-colored rug",
    "Industrial style metal pendant light",
    "Vintage brass drawer handles",
    "Running shoes with breathable mesh",
    "Tufted velvet chesterfield sofa",
    "Geometric print throw pillow blue",
    "Ceramic vase modern white",
    "Women's high heel sandals gladiator style"
]

def parse_json_col(val):
    """Helper to safely parse JSON strings back to Dicts for the final output."""
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except:
        return val

def build_repository():
    print("Loading Rich Dataset...")
    # This loads the dataframe with ['item_id', 'origin', 'category', 'title', 'features', 'path', 'other_attributes']
    df = pd.read_csv(DATA_FILE)
    
    if df.empty:
        print("❌ Dataset empty.")
        return

    # Initialize Engine (Uses Cache if available)
    engine = LocalSearchEngine(df, force_refresh=False)
    proof_gen = SocialProofGenerator()

    repository = []
    
    print(f"\nGeneratng Golden Set ({len(QUERIES)} queries)...")
    
    for q in QUERIES:
        print(f"   🔍 Query: {q}")
        
        # Run Search
        results_df = engine.search(q, top_k=TOP_K)

        # Convert results to Rich JSON
        results_list = []
        for _, row in results_df.iterrows():
            
            # We parse the stored JSON strings back into objects
            # so the final JSON is nested, not double-stringified.
            origin_data = parse_json_col(row.get('origin'))
            specs_data = parse_json_col(row.get('other_attributes'))

            stars, reviews = proof_gen.generate()
          
            item_data = {
                "rank": len(results_list) + 1, # Explicit Ranking
                "item_id": row['item_id'],
                "relevance_score": float(row['relevance_score']),
                "category": row['category'],
                
                # Core Content
                "title": row['title'],
                "features": row['features'],
                
                # The "Perfect" Context (What you asked for)
                "origin": origin_data,       # e.g. {"domain_name": "amazon.co.uk"}
                "specifications": specs_data, # e.g. {"material": "Leather", "color": "Black"}

                # Frozen Social Proof
                "rating": stars,
                "reviews": reviews,
              
                # For LLaVA
                "image_path": row['path']
            }
            results_list.append(item_data)
            
        # Add to Repo
        repository.append({
            "query": q,
            "results": results_list
        })

    # Save to Disk
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(repository, f, indent=4)
        
    print(f"\n✅ Repository saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_repository()