import json
import pandas as pd
import ast
import os
from search_engine import LocalSearchEngine

# --- CONFIGURATION ---
OUTPUT_FILE = "data/query.json"
TOP_K = 10
# We use the PT file directly because it contains the DF + Vectors
DATA_FILE = "data/mgeo_master_dataset.csv"
CACHE_FILE = "data/mgeo_master_dataset.pt"

QUERIES = [
    "Dolphin silver pendant",
    "Glitter crystal necklace bohemian style",
    "Matte black nail polish long lasting",
    "floral citrus perfume for women",
    "Bronze hoop earrings African boho style",
    "pink sweatsuit long sleeve for teen girls",
    "Red silk scarf for formal evening",
    "Vintage leather jacket with distressed look",
    "Best anti-fog goggles for blue light protection",
    "Comfortable black sandals for office work",
    "Grace Karin sleeveless pleated cocktail dress summer",
    "Wedding anniversary handmade wooden sign gift",
    "wide tooth comb for curly hair",
    "handmade floral greeting cards 5x7",
    "Light blue waterproof sofa cover for pets",
    "Christopher Knight Home iron fireplace screen black", 
    "handmade floral greeting cards",
    "linen pillow cover independence day theme",
    "waterproof chair seat covers teal for dining",
    "cotton kitchen towels diamond stripe",
    "black curtain rod set",
    "Soft winter Christmas stockings set of 3 for kids",
    "black leather watch strap 20mm",
    "waterproof eyeliner pen",
    "shampoo for dry scalp with natural oils",
]

def parse_col(val):
    """Safely parses stringified lists/dicts (e.g. specs or images)."""
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return ast.literal_eval(str(val))
    except:
        return val

def build_repository():
    
    df = pd.read_csv(DATA_FILE)
    
    if df.empty:
        print("❌ Dataset empty.")
        return

    # Initialize Engine (Uses Cache if available)
    engine = LocalSearchEngine(df, force_refresh=False, cache_file=CACHE_FILE)
    
    repository = []
    
    print(f"\nGeneratng Golden Set ({len(QUERIES)} queries)...")
    
    for q in QUERIES:
        print(f"   🔍 Query: {q}")
        
        # Run Search
        results_df = engine.search(q, top_k=TOP_K)

        # Convert results to Rich JSON
        results_list = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            
            # Parse 'specs' (formerly other_attributes/details)
            specs_data = parse_col(row.get('specs'))
            
            # Parse 'images' (formerly path)
            images_data = parse_col(row.get('images'))

            # Handle Rating (Real Data now, not synthetic)
            # Default to 0/0 if missing
            rating_val = row.get('rating') if pd.notna(row.get('rating')) else 0.0
            reviews_val = row.get('rating_number') if pd.notna(row.get('rating_number')) else 0

            item_data = {
                "rank": i + 1,
                "item_id": row.get('item_id'),
                "relevance_score": float(row.get('relevance_score', 0.0)),
                
                # Category might be 'dataset_source' or 'category' depending on your merge
                # We try 'category' first, fallback to 'dataset_source'
                "category": row.get('category') or row.get('dataset_source'),
                
                # Core Content
                "title": row.get('title'),
                "features": row.get('features'),
                
                # Context
                "specifications": specs_data,

                # Social Proof (Real)
                "rating": float(rating_val),
                "reviews": int(reviews_val),
              
                # Visuals
                "images": images_data
            }
            results_list.append(item_data)
            
        # Add to Repo
        repository.append({
            "query": q,
            "results": results_list
        })

    # Save to Disk
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(repository, f, indent=4)
        
    print(f"\n✅ Repository saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_repository()