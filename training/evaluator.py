import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from optimizer_agent import OptimizerAgent
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import format_rag_context, calculate_visibility_score


# --- CONFIGURATION ---
# We will pick 5 random candidates from your existing list to test
CANDIDATES_FILE = "data/target_candidates.json"
REPO_FILE = "data/query.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"

TEST_SIZE = 5 # Number of products to test
LAMBDA_PENALTY = 0.5

def run_test_pass(agent_model_name, label):
    """
    Runs a single pass of optimization using the specified model.
    """
    print(f"\n🥊 STARTING ROUND: {label} (Model: {agent_model_name})")
    
    # Init Agents
    # NOTE: We force the optimizer agent to use the specific model for this test
    opt_agent = OptimizerAgent(model_name=agent_model_name)
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()
    
    # Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles = json.load(f)
    mgeo_rules = principles.get('mgeo_principles', [])

    results = []
    
    # Flatten and pick first N tasks
    tasks = []
    for query, items in candidates_map.items():
        for item in items:
            tasks.append((query, item))
            if len(tasks) >= TEST_SIZE: break
        if len(tasks) >= TEST_SIZE: break

    for query, candidate in tasks:
        target_id = candidate['item_id']
        visual_desc = captions.get(target_id, "")
        
        # Get Product Data
        product_data = next((res for q in repo if q['query'] == query for res in q['results'] if res['item_id'] == target_id), None)
        if not product_data: continue

        print(f"   Testing {target_id}...")
        
        # 1. OPTIMIZE (Single Shot - No retries! We want to test Intelligence)
        start_ts = pd.Timestamp.now()
        opt_res = opt_agent.optimize_product(query, product_data, visual_desc, mgeo_rules)
        
        if not opt_res:
            results.append({"reward": -1.0, "vis": 0, "vgs": 0})
            continue

        # 2. VERIFY
        # Setup Swap
        query_group = next((q for q in repo if q['query'] == query), None)
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
        
        # Run Sim
        rag_ctx = format_rag_context(test_candidates)
        gen_text = sim_agent.generate_response(query, rag_ctx)
        
        if gen_text:
            vis = calculate_visibility_score(gen_text, target_id)
            full_text = f"{opt_res['optimized_title']} {opt_res['optimized_features']}"
            vgs = vgs_judge.calculate_vgs(target_id, full_text, image_url)
            reward = vis - (LAMBDA_PENALTY * (1.0 - vgs))
        else:
            vis, vgs, reward = 0, 0, -1.0

        results.append({
            "item_id": target_id,
            "reward": reward,
            "vis": vis,
            "vgs": vgs
        })
        print(f"      -> Reward: {reward:.4f} (Vis: {vis}, VGS: {vgs})")

    return results

def main():
    # 1. Run Baseline (Standard Llama 3)
    # Note: Ensure you have 'llama3' pulled in Ollama or use another generic model tag
    baseline_results = run_test_pass("llama3", "BASELINE (Generic)")

    # 2. Run Trained (Your New Model)
    trained_results = run_test_pass("geo-optimizer", "TRAINED (Specialist)")

    # 3. Compare
    print("\n\n📊 FINAL SCORECARD")
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Trained':<10} | {'Delta':<10}")
    print("-" * 60)
    
    # Calculate Averages
    def get_avg(res_list, key):
        vals = [r[key] for r in res_list if r['reward'] > -0.5] # Filter failures
        return sum(vals)/len(vals) if vals else 0.0

    metrics = ['reward', 'vis', 'vgs']
    for m in metrics:
        base_avg = get_avg(baseline_results, m)
        train_avg = get_avg(trained_results, m)
        delta = train_avg - base_avg
        print(f"{m.upper():<20} | {base_avg:.4f}     | {train_avg:.4f}     | {delta:+.4f}")

    # Win Rate
    base_wins = len([r for r in baseline_results if r['reward'] > 0])
    train_wins = len([r for r in trained_results if r['reward'] > 0])
    print(f"{'Success Rate':<20} | {base_wins}/{TEST_SIZE}       | {train_wins}/{TEST_SIZE}       | {train_wins-base_wins:+}")

if __name__ == "__main__":
    main()