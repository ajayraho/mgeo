import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer_agent import OptimizerAgent
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import format_rag_context, calculate_visibility_score

# --- CONFIGURATION ---
CANDIDATES_FILE = "data/test_candidates.json"
REPO_FILE = "data/test_repo.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"
OUTPUT_ABLATION = "data/results_ablation.json"
MODEL_NAME = "geo-optimizer" # We use the expert to test the rules

def get_overall_score(vis, vgs):
    return (vis + vgs) / 2

def run_ablation():
    print(f"\n🔬 PHASE 3: ABLATION STUDY (RULE ISOLATION)...")
    
    # 1. Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: all_principles = json.load(f)
    rules = all_principles.get('mgeo_principles', [])

    # 2. Init Agents
    agent = OptimizerAgent(model_name=MODEL_NAME)
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()
    
    results = []
    
    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))

    for query, product in tqdm(tasks, desc="Isolating Rules"):
        target_id = product['item_id']
        visual_desc = captions.get(target_id, "")
        
        # Test Each Rule Individually
        for rule in rules:
            single_rule = [rule] # Isolate!
            
            # Optimize
            opt_res = agent.optimize_product(query, product, visual_desc, single_rule)
            
            if not opt_res:
                vis, vgs, ovr = 0, 0, 0
            else:
                # Context
                query_group = next((q for q in repo if q['query'] == query), None)
                if not query_group: 
                    vis, vgs, ovr = 0, 0, 0
                else:
                    test_candidates = []
                    image_url = None
                    for item in query_group['results']:
                        if item['item_id'] == target_id:
                            mod = item.copy()
                            mod['title'] = opt_res['optimized_title']
                            mod['features'] = opt_res['optimized_features']
                            test_candidates.append(mod)
                            image_url = item.get('main_image_url')
                        else:
                            test_candidates.append(item)
                    
                    # Score
                    rag_ctx = format_rag_context(test_candidates)
                    gen_text = sim_agent.generate_response(query, rag_ctx)
                    vis = calculate_visibility_score(gen_text, target_id)
                    
                    full_text = f"{opt_res['optimized_title']} {opt_res['optimized_features']}"
                    vgs = vgs_judge.calculate_vgs(target_id, full_text, image_url)
                    ovr = get_overall_score(vis, vgs)

            results.append({
                "Rule_Name": rule['rule_name'],
                "Vis": vis,
                "VGS": vgs,
                "Overall": ovr
            })

    # Save
    pd.DataFrame(results).to_json(OUTPUT_ABLATION, orient='records', indent=4)
    print(f"\n✅ Ablation results saved to {OUTPUT_ABLATION}")

if __name__ == "__main__":
    run_ablation()