import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import os
import pickle

# --- CONFIGURATION ---
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
CACHE_FILE = 'data/search_index.pt'  # The file where we store the vectors

class LocalSearchEngine:
    def __init__(self, dataframe, force_refresh=False):
        """
        Initializes the Search Engine with Persistent Caching.
        """
        print(f"🚀 Initializing Search Engine on {DEVICE}...")
        self.df = dataframe.copy().reset_index(drop=True)
        self.model = None 
        
        # 1. Try to Load from Disk
        if os.path.exists(CACHE_FILE) and not force_refresh:
            print(f"   📂 Found cached index '{CACHE_FILE}'. Loading...")
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
        print("   💾 Index saved to disk.")
        print("✅ Search Engine Online.")

    def _create_search_payload(self):
        def clean(x): return str(x).strip() if pd.notna(x) else ""
        instruction = "Represent this product document for retrieval: "
        self.df['search_payload'] = (
            instruction + 
            "Category: " + self.df['category'].apply(clean) + " ; " +
            "Title: " + self.df['title'].apply(clean) + " ; " +
            "Description: " + self.df['features'].apply(clean)
        )

    def save_index(self):
        """Saves to disk. Moves to CPU to avoid pickling GPU tensors."""
        payload = {
            'embeddings': self.embeddings.cpu(),
            'dataframe': self.df
        }
        torch.save(payload, CACHE_FILE)

    def load_index(self):
        """Loads from disk. Fixes PyTorch 2.6+ security error."""
        # weights_only=False is required to load Pandas DataFrames
        payload = torch.load(CACHE_FILE, map_location=DEVICE, weights_only=False)
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
[Result #{i+1} | ID: {row['item_id']}]
Category: {row['category']}
Title: {row['title']}
Features: {str(row['features'])}...
Score: {row['relevance_score']:.4f}
--------------------------------------------------
"""
        return context_str

# --- INTEGRATION TEST ---
if __name__ == "__main__":
    # 1. Load your DataFrame from the Loader
    df = pd.read_csv('data/amazon_dataset.csv')
    
    # First Run: Will Compute
    # print("\n--- RUN 1 (Computing) ---")
    # engine = LocalSearchEngine(df, force_refresh=True)
    
    # Second Run: Will Load from Cache
    print("\n--- Loading ---")
    engine_cached = LocalSearchEngine(df)
    
    # Test Search
    print(engine_cached.format_for_rag(engine_cached.search("women shoes", top_k=5)))