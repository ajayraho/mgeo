import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import os
import argparse  # Added for command line arguments

# --- CONFIGURATION ---
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128

class LocalSearchEngine:
    def __init__(self, dataframe, cache_file, force_refresh=False):
        """
        Initializes the Search Engine with Persistent Caching.
        
        Args:
            dataframe: The pd.DataFrame containing product data.
            cache_file: Path to the .pt file where vectors will be stored.
            force_refresh: If True, ignores existing cache and re-computes.
        """
        self.cache_file = cache_file # Store the specific cache path for this instance
        print(f"🚀 Initializing Search Engine on {DEVICE}...")
        self.df = dataframe.copy().reset_index(drop=True)
        self.model = None 
        
        # 1. Try to Load from Disk
        if os.path.exists(self.cache_file) and not force_refresh:
            print(f"   📂 Found cached index '{self.cache_file}'. Loading...")
            try:
                self.load_index()
                print("   ✅ Loaded from disk successfully.")
                return 
            except Exception as e:
                print(f"   ⚠️ Cache load failed ({e}). Re-computing...")
        
        # 2. Compute if not cached
        print(f"   ⚙️ Computing new vector index (Model: {MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        self._create_search_payload()
        
        print(f"   Vectorizing {len(self.df)} products...")
        embeddings_cpu = self.model.encode(
            self.df['search_payload'].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=DEVICE
        )
        
        self.embeddings = embeddings_cpu
        self.save_index()
        print(f"   💾 Index saved to disk at: {self.cache_file}")
        print("✅ Search Engine Online.")
    def _clean_specs(self, x):
        """
        Converts a dictionary string "{'Color': 'Red'}" 
        into a clean semantic string "Color: Red ; ..."
        """
        import ast
        try:
            # 1. Handle Empty/NaN
            if pd.isna(x) or x == "": 
                return ""
            
            # 2. Parse String -> Dict (Safe eval)
            # If it's already a dict (from .pt load), use it directly.
            if isinstance(x, dict):
                specs = x
            else:
                specs = ast.literal_eval(str(x))
            
            if not isinstance(specs, dict): 
                return ""
            
            # 3. Format into "Key: Value" string
            # We skip internal keys like 'node' or 'dimensions' if you want, 
            # or just take everything that is a string/number.
            parts = []
            for k, v in specs.items():
                # Filter out complex nested lists/dicts to keep it simple
                if isinstance(v, (str, int, float)):
                    parts.append(f"{k}: {v}")
            
            return " ; ".join(parts)
            
        except Exception:
            return ""

    def _create_search_payload(self):
        def clean(x): return str(x).strip() if pd.notna(x) else ""
        
        # 1. Ensure columns exist
        for col in ['title', 'features', 'specs']:
            if col not in self.df.columns:
                self.df[col] = ""

        # 2. Pre-process the Specs (Dict -> String)
        print("   🔨 Formatting specifications for vectorization...")
        # This creates the clean "Color: Red ; Brand: X" string
        self.df['formatted_specs'] = self.df['specs'].apply(self._clean_specs)

        # 3. Construct the Master Vector String
        instruction = "Represent this product document for retrieval: "
        
        self.df['search_payload'] = (
            instruction + 
            "Title: " + self.df['title'].apply(clean) + " ; " +
            "Specs: " + self.df['formatted_specs'] + " ; " +  # <--- Cleaned Text
            "Description: " + self.df['features'].apply(clean)
        )

    def save_index(self):
        """Saves to disk. Moves to CPU to avoid pickling GPU tensors."""
        payload = {
            'embeddings': self.embeddings.cpu(),
            'dataframe': self.df
        }
        torch.save(payload, self.cache_file)

    def load_index(self):
        """Loads from disk. Fixes PyTorch 2.6+ security error."""
        # weights_only=False is required to load Pandas DataFrames
        payload = torch.load(self.cache_file, map_location=DEVICE, weights_only=False)
        self.embeddings = payload['embeddings'].to(DEVICE)
        self.df = payload['dataframe']

    def search(self, query, top_k=5):
        """Performs Pure Semantic Search (No Hard Filters)."""
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)

        instruction = "Represent this sentence for searching relevant passages: "
        
        with torch.no_grad():
            query_vec = self.model.encode(
                [instruction + query], 
                normalize_embeddings=True, 
                convert_to_tensor=True, 
                device=DEVICE
            )
            
            scores = util.dot_score(query_vec, self.embeddings)[0]
            top_results = torch.topk(scores, k=top_k)
            
            top_indices = top_results.indices.cpu().numpy()
            top_scores = top_results.values.cpu().numpy()
        
        results_df = self.df.iloc[top_indices].copy()
        results_df['relevance_score'] = top_scores
        return results_df

    def format_for_rag(self, results_df):
        context_str = ""
        for i, (idx, row) in enumerate(results_df.iterrows()):
            context_str += f"""
[Result #{i+1} | ID: {row.get('item_id', 'N/A')}]
Title: {row.get('title', 'N/A')}
Features: {str(row.get('features', 'N/A'))}
Specs: {str(row.get('formatted_specs', 'N/A'))}
Score: {row['relevance_score']:.4f}
--------------------------------------------------
"""
        return context_str

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Embed a CSV dataset for Semantic Search.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    
    args = parser.parse_args()
    input_csv = args.csv_path
    
    # 2. Determine Output Filename (.csv -> .pt)
    base, ext = os.path.splitext(input_csv)
    output_pt = base + ".pt"
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: File '{input_csv}' not found.")
        exit(1)

    print(f"📂 Reading CSV: {input_csv}")
    print(f"💾 Target Cache: {output_pt}")

    # 3. Load Data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        exit(1)

    # 4. Initialize Engine (Computes and Saves .pt file)
    engine = LocalSearchEngine(df, cache_file=output_pt)

    # 5. Quick Test
    print("\n🔎 Running Sanity Check (Query: 'quality product')...")
    results = engine.search("quality product", top_k=10)
    print(engine.format_for_rag(results))