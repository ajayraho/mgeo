import sys
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter, defaultdict

# --- IMPORTS (Adjust paths if needed) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import format_rag_context, calculate_visibility_score
from evaluator import BaselineAgent, TrainedAgent  # Reuse your existing agents

# --- CONFIGURATION ---
REPO_CAT_FILE = "data/test_repo_cat.json"       # Source of Truth for Categories
CANDIDATES_FILE = "data/test_candidates.json"   # The items to optimize
VISUALS_FILE = "data/test_dense_captions_f.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"

OUTPUT_FULL = "data/category_results_full.csv"
OUTPUT_SUMMARY = "data/category_results_summary.csv"
LOG_FILE = "data/category_battle_logs.txt"

VERBOSE = True

def get_query_category_map(repo_path):
    """
    Classifies queries based on the dominant category of their results.
    Returns: dict { "query_string": "Category Name" }
    """
    print(f"📂 Classification: Scanning {repo_path}...")
    with open(repo_path, encoding="utf-8") as f:
        data = json.load(f)

    query_to_cat = {}
    
    for q_entry in data:
        query_text = q_entry.get("query")
        cats = [p.get("category") for p in q_entry.get("results", []) if p.get("category")]
        
        if not cats:
            dominant = "Uncategorized"
        else:
            # Your Logic: Dominant Category wins
            dominant = Counter(cats).most_common(1)[0][0]
            
        query_to_cat[query_text] = dominant
        
    print(f"✅ Classified {len(query_to_cat)} queries.")
    return query_to_cat

def run_category_evaluation():
    # 1. SETUP
    print(f"\n🚀 STARTING CATEGORY-WISE EVALUATION...")
    with open(LOG_FILE, "w") as f: f.write(f"--- START: {datetime.now()} ---\n")

    # Load Data
    query_cat_map = get_query_category_map(REPO_CAT_FILE)
    
    # Ensure Candidates Exist
    if not os.path.exists(CANDIDATES_FILE):
        print(f"❌ Error: {CANDIDATES_FILE} missing. Run previous steps to generate candidates.")
        return

    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_CAT_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles = json.load(f)
    mgeo_rules = principles.get('mgeo_principles', [])

    # Initialize Agents
    baseline_agent = BaselineAgent("llama3:8b")
    trained_agent = TrainedAgent("geo-optimizer")
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()

    results = []
    
    # Flatten Tasks
    tasks = []
    for q, items in candidates_map.items():
        # Tag the query with its category immediately
        cat = query_cat_map.get(q, "Unknown")
        for i in items:
            tasks.append((q, i, cat))

    # 2. BATTLE LOOP
    for i, (query, product, category) in enumerate(tqdm(tasks, desc="Evaluator")):
        target_id = product['item_id']
        visual_desc = captions.get(target_id, "")
        
        # --- EXECUTION (Reusing Logic) ---
        def evaluate_agent(agent):
            res = agent.optimize(query, product, visual_desc, mgeo_rules)
            if not res or not res.get('optimized_title'): return 0, 0, 0
            
            # Context for Simulator
            query_group = next((x for x in repo if x['query'] == query), None)
            if not query_group: return 0, 0, 0
            
            # Construct RAG Context
            test_candidates = []
            image_url = None
            for item in query_group['results']:
                if item['item_id'] == target_id:
                    mod = item.copy()
                    mod['title'] = res['optimized_title']
                    mod['features'] = res['optimized_features']
                    test_candidates.append(mod)
                    image_url = item.get('main_image_url')
                else:
                    test_candidates.append(item)
            
            # Sim & VGS
            rag_ctx = format_rag_context(test_candidates)
            gen_text = sim_agent.generate_response(query, rag_ctx)
            vis = calculate_visibility_score(gen_text, target_id)
            full_txt = f"{res['optimized_title']} {res['optimized_features']}"
            vgs = vgs_judge.calculate_vgs(target_id, full_txt, image_url)
            
            return vis, vgs, (vis + vgs) / 2

        # Run Both
        b_vis, b_vgs, b_ovr = evaluate_agent(baseline_agent)
        t_vis, t_vgs, t_ovr = evaluate_agent(trained_agent)

        # Determine Winner
        if t_ovr > b_ovr: winner = "Trained"
        elif b_ovr > t_ovr: winner = "Baseline"
        else: winner = "Tie"

        # Log Result Row
        results.append({
            "Category": category,  # <--- CRITICAL FIELD
            "Query": query,
            "Item_ID": target_id,
            "Baseline_Score": b_ovr,
            "Trained_Score": t_ovr,
            "Score_Delta": t_ovr - b_ovr,
            "Winner": winner,
            "Baseline_Vis": b_vis,
            "Trained_Vis": t_vis,
            "Baseline_VGS": b_vgs,
            "Trained_VGS": t_vgs
        })
        
        # Incremental Save (Safety)
        if i % 2 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FULL, index=False)

    # 3. SAVE FINAL FULL RESULTS
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FULL, index=False)
    print(f"\n✅ Raw Results saved to {OUTPUT_FULL}")

    # 4. GENERATE SUMMARY (The "Transaction Paper" Table)
    print("\n📊 Generating Executive Summary...")
    
    summary_rows = []
    
    # Group by Category
    grouped = df.groupby("Category")
    
    for cat, group in grouped:
        total = len(group)
        trained_wins = len(group[group["Winner"] == "Trained"])
        baseline_wins = len(group[group["Winner"] == "Baseline"])
        ties = len(group[group["Winner"] == "Tie"])
        
        # Calculate Win Rates
        win_rate = (trained_wins / total) * 100
        
        # Calculate Avg Improvements
        avg_base = group["Baseline_Score"].mean()
        avg_train = group["Trained_Score"].mean()
        improvement = ((avg_train - avg_base) / avg_base) * 100 if avg_base > 0 else 0
        
        summary_rows.append({
            "Category": cat,
            "Samples": total,
            "Trained_Win_Rate": f"{win_rate:.1f}%",
            "Baseline_Wins": baseline_wins,
            "Ties": ties,
            "Avg_Baseline_Score": f"{avg_base:.2f}",
            "Avg_Trained_Score": f"{avg_train:.2f}",
            "Rel_Improvement": f"+{improvement:.1f}%"
        })
    
    # Also add a "GLOBAL" row for comparison
    total = len(df)
    t_wins = len(df[df["Winner"] == "Trained"])
    avg_base = df["Baseline_Score"].mean()
    avg_train = df["Trained_Score"].mean()
    imp = ((avg_train - avg_base) / avg_base) * 100 if avg_base > 0 else 0
    
    summary_rows.append({
        "Category": "ALL_CATEGORIES (Global)",
        "Samples": total,
        "Trained_Win_Rate": f"{(t_wins/total)*100:.1f}%",
        "Baseline_Wins": len(df[df["Winner"] == "Baseline"]),
        "Ties": len(df[df["Winner"] == "Tie"]),
        "Avg_Baseline_Score": f"{avg_base:.2f}",
        "Avg_Trained_Score": f"{avg_train:.2f}",
        "Rel_Improvement": f"+{imp:.1f}%"
    })

    summary_df = pd.DataFrame(summary_rows)
    # Reorder columns for readability
    cols = ["Category", "Samples", "Trained_Win_Rate", "Rel_Improvement", 
            "Avg_Baseline_Score", "Avg_Trained_Score", "Baseline_Wins", "Ties"]
    summary_df = summary_df[cols]
    
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"✅ Summary saved to {OUTPUT_SUMMARY}")
    print("\n")
    print(summary_df.to_string())

if __name__ == "__main__":
    run_category_evaluation()