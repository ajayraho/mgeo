import sys
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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
OUTPUT_RESULTS = "data/results_comparative.json"
LOG_FILE = "data/battle_logs.txt"

# --- VERBOSITY SETTINGS ---
VERBOSE = True  # Set to True to see LLM output and detailed scores in console/logs

# MODELS
BASELINE_MODEL = "llama3:8b"     # Stock Model
TRAINED_MODEL = "geo-optimizer"        # Your Model

def get_overall_score(vis, vgs):
    return (vis + vgs) / 2

def log_message(message):
    """Prints to console and appends to log file."""
    if VERBOSE:
        tqdm.write(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def run_evaluation():
    print(f"\n⚔️  PHASE 2: COMPARATIVE EVALUATION ({BASELINE_MODEL} vs {TRAINED_MODEL})...")
    
    # Initialize Log File
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- BATTLE LOG START: {datetime.now()} ---\n")
        f.write(f"Baseline: {BASELINE_MODEL}\n")
        f.write(f"Challenger: {TRAINED_MODEL}\n")
        f.write("="*60 + "\n")

    # 1. Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles = json.load(f)
    mgeo_rules = principles.get('mgeo_principles', [])

    # 2. Init Agents
    log_message("🤖 Initializing Agents...")
    agent_base = OptimizerAgent(model_name=BASELINE_MODEL)
    agent_train = OptimizerAgent(model_name=TRAINED_MODEL) 
    
    sim_agent = SimulatorAgent() # Uses gpt-oss by default (the Judge)
    vgs_judge = VisualGroundingScorer()

    results = []
    
    # Flatten Dictionary
    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))

    for i, (query, product) in enumerate(tqdm(tasks, desc="Battling")):
        target_id = product['item_id']
        visual_desc = captions.get(target_id, "")
        
        log_message(f"\n{'='*60}")
        log_message(f"⚔️ BATTLE {i+1}/{len(tasks)}: {target_id}")
        log_message(f"   Query: {query}")
        log_message(f"{'='*60}")
        
        # --- SCORING HELPER ---
        def evaluate_agent(agent, model_label):
            log_message(f"\n👉 Invoking Agent: {model_label} ...")
            
            # 1. Optimization
            # The Agent handles the prompt construction and JSON parsing
            opt_res = agent.optimize_product(query, product, visual_desc, mgeo_rules)
            
            if not opt_res: 
                log_message(f"   ⚠️ {model_label} FAILED to produce valid JSON.")
                return 0, 0, 0
            
            # Log Output
            if VERBOSE:
                log_message(f"   📝 OUTPUT ({model_label}):")
                log_message(f"      Title: {opt_res['optimized_title']}")
                log_message(f"      Feat : {opt_res['optimized_features'][:100]}...") # Truncate for sanity
            
            # 2. Context Setup for Simulator
            query_group = next((q for q in repo if q['query'] == query), None)
            if not query_group: 
                log_message("   ❌ Error: Query context not found in repo.")
                return 0, 0, 0
            
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
            
            # 3. Calculate Scores
            # A. Visibility
            rag_ctx = format_rag_context(test_candidates)
            gen_text = sim_agent.generate_response(query, rag_ctx)
            vis = calculate_visibility_score(gen_text, target_id)
            
            # B. Visual Grounding
            full_text = f"{opt_res['optimized_title']} {opt_res['optimized_features']}"
            vgs = vgs_judge.calculate_vgs(target_id, full_text, image_url)
            
            overall = get_overall_score(vis, vgs)
            
            log_message(f"   📊 SCORES ({model_label}): Vis={vis:.2f} | VGS={vgs:.2f} | Overall={overall:.2f}")
            return vis, vgs, overall

        # --- RUN EVALUATION ---
        # 1. Baseline
        b_vis, b_vgs, b_ovr = evaluate_agent(agent_base, BASELINE_MODEL)
        
        # 2. Trained
        t_vis, t_vgs, t_ovr = evaluate_agent(agent_train, TRAINED_MODEL)

        # Winner Logic
        winner = "Tie"
        if t_ovr > b_ovr: winner = "Trained"
        elif b_ovr > t_ovr: winner = "Baseline"
        
        log_message(f"\n🏆 WINNER: {winner} (Train {t_ovr:.2f} vs Base {b_ovr:.2f})")

        results.append({
            "query": query,
            "product_id": target_id,
            "Baseline_Vis": b_vis, "Baseline_VGS": b_vgs, "Baseline_Overall": b_ovr,
            "Trained_Vis": t_vis,   "Trained_VGS": t_vgs,   "Trained_Overall": t_ovr,
            "Winner": winner
        })
        
        # Incremental Save (So you don't lose data if it crashes)
        pd.DataFrame(results).to_json(OUTPUT_RESULTS, orient='records', indent=4)

    print(f"\n✅ Comparative results saved to {OUTPUT_RESULTS}")
    print(f"📜 Detailed logs saved to {LOG_FILE}")
    
    # Quick Stats
    df = pd.DataFrame(results)
    if not df.empty:
        print("\n📊 QUICK LOOK:")
        print(f"Avg Train Overall: {df['Trained_Overall'].mean():.4f}")
        print(f"Avg Base Overall : {df['Baseline_Overall'].mean():.4f}")
        print(f"Train Win Rate   : {(len(df[df['Winner']=='Trained'])/len(df))*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()