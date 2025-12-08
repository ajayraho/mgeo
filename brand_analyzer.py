import json
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
OUTPUT_FILE = "data/brand_popularity.json"

def normalize_brand(brand_raw):
    """
    Cleans up brand names to merge variations.
    e.g., "Amazon Brand - Solimo" -> "solimo"
    """
    if not brand_raw: return "unknown"

    b = str(brand_raw).lower().strip()

    # Heuristic: Remove common prefixes/suffixes
    b = b.replace("amazon brand - ", "")
    b = b.replace("amazonbasics", "amazon basics") # Normalize spelling

    # Take the first meaningful word if it's a long string
    # (Optional: keeps 'solimo' from 'solimo designer cases')
    # b = b.split()[0]

    return b

def build_global_brand_map():
    print("🚀 Loading Full 7k Dataset for Brand Analysis...")
    df = pd.read_csv('data/amazon_dataset.csv', engine='python') # Loads the full dataframe

    if df.empty:
        print("❌ Dataset empty.")
        return

    print(f"   Scanning {len(df)} products...")

    # 1. Extract Brands
    # We look in 'other_attributes' (JSON) first, then fallback to Title
    raw_brands = []

    for _, row in df.iterrows():
        brand = None

        # Try JSON specs first
        try:
            if row['other_attributes']:
                specs = json.loads(row['other_attributes'])
                if isinstance(specs, dict):
                    brand = specs.get('brand')
        except:
            pass

        # Fallback to Title (heuristic: first 2 words)
        if not brand and row['title']:
            brand = " ".join(row['title'].split()[:2])

        if brand:
            raw_brands.append(normalize_brand(brand))

    # 2. Count Frequencies
    brand_counts = Counter(raw_brands)
    total_products = len(df)

    print(f"   Found {len(brand_counts)} unique brands.")
    print(f"   Top 5 Giants: {brand_counts.most_common(5)}")

    # 3. Calculate Popularity Scores (Log-Normalized)
    # Score 1.0 = The most popular brand in the dataset (The "King")
    # Score 0.0 = A brand that appears once

    max_freq = brand_counts.most_common(1)[0][1] # Count of the top brand

    brand_map = {}
    for brand, count in brand_counts.items():
        # Frequency Ratio: count / max_freq
        # We use Log scaling to dampen the curve (so 500 items isn't 500x more biased than 1)
        # Formula: log(count) / log(max_freq)

        if count == 1:
            score = 0.0
        else:
            score = np.log(count) / np.log(max_freq)

        brand_map[brand] = {
            "count": count,
            "popularity_score": round(score, 4)
        }

    # 4. Save to Disk
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(brand_map, f, indent=4)

    print(f"\n✅ Brand Analysis Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_global_brand_map()