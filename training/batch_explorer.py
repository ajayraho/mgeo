# Add parent folder to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import copy
from tqdm import tqdm
from optimizer_agent import OptimizerAgent
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
# We reuse helper functions from your existing files
from verify_optimization import format_rag_context, calculate_visibility_score

# --- CONFIGURATION ---
CANDIDATES_FILE = "data/target_candidates.json"
REPO_FILE = "data/query.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"
OUTPUT_DATASET = "data/rl_finetuning_dataset.json"
PROGRESS_LOG = "data/batch_explorer_progress.txt" # Tracks completed tasks

files = {
    "CANDIDATES_FILE": CANDIDATES_FILE,
    "REPO_FILE": REPO_FILE,
    "VISUALS_FILE": VISUALS_FILE,
    "PRINCIPLES_FILE": PRINCIPLES_FILE,
}

print("Checking file paths...\n")
for name, path in files.items():
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        print(f"[OK] {name} exists → {abs_path}")
    else:
        print(f"[ERROR] {name} NOT FOUND → {abs_path}")

# Hyperparameters
SAMPLES_PER_PRODUCT = 5   # How many variations to try per product
LAMBDA_PENALTY = 0.5      # Same safety setting as your verified success

def score_variation(target_id, target_query, new_title, new_features, repo_data, sim_agent, vgs_judge):
    """
    Runs the 'Mini-Verification' loop for a single variation.
    """
    # 1. Setup Context (Hot Swap)
    query_group = next((q for q in repo_data if q['query'] == target_query), None)
    if not query_group: return -10, 0, 0

    test_candidates = []
    image_url = None
    
    for item in query_group['results']:
        if item['item_id'] == target_id:
            mod = item.copy()
            mod['title'] = new_title
            mod['features'] = new_features
            test_candidates.append(mod)
            image_url = item.get('main_image_url')
        else:
            test_candidates.append(item)

    # 2. Run Simulator (Get Visibility)
    rag_ctx = format_rag_context(test_candidates)
    
    # We use the Two-Step generation you implemented
    gen_text = sim_agent.generate_response(target_query, rag_ctx)
    if not gen_text: return -10, 0, 0
    
    # Calculate Vis for ALL items to find the rank (optional, but good for filtering)
    vis_score = calculate_visibility_score(gen_text, target_id)

    # 3. Run Utility Judge (Get VGS)
    full_text = f"{new_title} {new_features}"
    vgs_score = vgs_judge.calculate_vgs(target_id, full_text, image_url)

    # 4. Calculate Reward
    # Reward = Visibility - Penalty * (1 - VisualAccuracy)
    reward = vis_score - (LAMBDA_PENALTY * (1.0 - vgs_score))
    
    return reward, vis_score, vgs_score

def load_progress():
    """Loads existing dataset and set of processed task IDs."""
    dataset = []
    processed = set()
    
    # Load Dataset
    if os.path.exists(OUTPUT_DATASET):
        try:
            with open(OUTPUT_DATASET, 'r') as f:
                dataset = json.load(f)
            print(f"🔄 Resuming: Loaded {len(dataset)} existing training examples.")
        except json.JSONDecodeError:
            print("⚠️ Warning: Dataset file corrupted or empty. Starting fresh.")
    
    # Load Progress Log (Completed Tasks)
    if os.path.exists(PROGRESS_LOG):
        try:
            with open(PROGRESS_LOG, 'r') as f:
                processed = set(line.strip() for line in f if line.strip())
            print(f"🔄 Resuming: Skipped {len(processed)} previously processed tasks.")
        except:
            pass
            
    return dataset, processed

def save_checkpoint(dataset, processed_id):
    """Incrementally saves JSON and appends to progress log."""
    # 1. Save JSON (Overwrite)
    with open(OUTPUT_DATASET, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    # 2. Append to Log
    with open(PROGRESS_LOG, 'a') as f:
        f.write(f"{processed_id}\n")

def main():
    print("🚀 Starting Batch Explorer (Data Mining)...")
    
    # Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles = json.load(f)
    
    mgeo_rules = principles.get('mgeo_principles', []) or principles.get('refined_principles', [])

    # Initialize Agents
    opt_agent = OptimizerAgent()
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()

    # --- RESUME LOGIC ---
    collected_examples, processed_tasks = load_progress()
    
    # Flatten candidates list
    all_tasks = []
    for query, items in candidates_map.items():
        for item in items:
            # Create a unique signature for this task
            task_sig = f"{query}|{item['item_id']}"
            all_tasks.append((task_sig, query, item))

    print(f"   Found {len(all_tasks)} total candidates. {len(all_tasks) - len(processed_tasks)} remaining.")

    for task_sig, query, candidate in tqdm(all_tasks):
        # Skip if done
        if task_sig in processed_tasks:
            continue

        target_id = candidate['item_id']
        visual_desc = captions.get(target_id, "")
        
        # Get full product data
        product_data = None
        for q_obj in repo:
            if q_obj['query'] == query:
                for res in q_obj['results']:
                    if res['item_id'] == target_id:
                        product_data = res
                        break
        if not product_data: 
            # Mark missing data as processed so we don't retry forever
            save_checkpoint(collected_examples, task_sig)
            continue

        best_reward = -100
        best_trajectory = None

        # --- THE EXPLORATION LOOP ---
        # --- NEW HARVEST LOGIC ---
        # We don't track just one winner. We save EVERYTHING that works.
        
        for i in range(SAMPLES_PER_PRODUCT):
            # 1. Optimize
            result = opt_agent.optimize_product(query, product_data, visual_desc, mgeo_rules)
            if not result: continue
            
            # 2. Score
            reward, vis, vgs = score_variation(
                target_id, query, 
                result['optimized_title'], 
                result['optimized_features'], 
                repo, sim_agent, vgs_judge
            )
            
            # 3. The Harvest Filter
            # CRITERIA:
            # - Vis > 0.2: It successfully moved up the ranking (Rank < ~20).
            # - VGS > 0.65: It is truthful (Safe for Sliding Window logic).
            # - Reward > 0: General sanity check.
            if vis > 0.3 and vgs > 0.65 and reward > 0:
                
                trajectory = {
                    "instruction": f"Optimize the product text for query: '{query}'.\nVisual Truth: {visual_desc}\nApply MGEO Principles.",
                    "input": f"Title: {product_data['title']}\nFeatures: {product_data['features']}",
                    "output": f"{result['optimized_title']}\n{result['optimized_features']}",
                    "metrics": {"reward": reward, "vis": vis, "vgs": vgs}
                }
                
                collected_examples.append(trajectory)
                tqdm.write(f"   🌟 Saved Variant: Vis {vis:.2f} | VGS {vgs:.2f} | Reward {reward:.2f}")
        
        # --- CHECKPOINT ---
        # We save progress whether we found a winner or not, to mark this Item as "Attempted"
        save_checkpoint(collected_examples, task_sig)

    print(f"\n✅ Mining Complete. Collected {len(collected_examples)} Golden Examples.")
    print(f"   Saved to {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()