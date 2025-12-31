import sys
import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import calculate_visibility_score, format_rag_context

# --- CONFIGURATION ---
CANDIDATES_FILE = "data/test_candidates.json"
REPO_FILE = "data/test_repo.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"

# OUTPUTS
OUTPUT_SUMMARY = "data/ablation_summary.csv"   # The Table for your Paper
OUTPUT_FULL = "data/ablation_full_log.csv"     # Every single data point
OUTPUT_TEXT = "data/ablation_generations.json" # The actual text generated

MODEL_NAME = "geo-optimizer"

# --- HELPER: ROBUST PARSER ---
def parse_trained_output(text):
    text = text.strip()
    if "```" in text: text = text.replace("```", "").strip()
    
    lines = text.split('\n')
    title, features = "", ""
    
    for line in lines:
        clean = line.strip()
        if clean.lower().startswith("title:"): title = clean[6:].strip()
        elif clean.lower().startswith("features:"): features = clean[9:].strip()
        elif not title and clean: title = clean
        elif not features and clean: features = clean

    return {"optimized_title": title, "optimized_features": features}

class AblationAgent:
    def optimize(self, query, product, visual_desc, instruction_override):
        sys_msg = (
            f"You are an Elite Generative Engine Optimization Specialist. {instruction_override}\n"
            f"Visual Truth: {visual_desc}"
        )
        user_msg = f"Title: {product['title']}\nFeatures: {product['features']}"

        est_tokens = len(user_msg) / 3.0
        if est_tokens > 2048:
              print(f"⚠️ [ablation_study] OPTIMIZATION PROMPT IS HUGE ({int(est_tokens)} tokens)!")
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 8192,
                        "stop": ["<|eot_id|>"]
                    }
                }
            )
            raw_text = resp.json()['message']['content']
            return parse_trained_output(raw_text)
        except Exception:
            return None

def main():
    print(f"🔬 STARTING FULL-SCALE ABLATION STUDY (Model: {MODEL_NAME})")
    
    # 1. Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles_data = json.load(f)
    
    raw_rules = principles_data.get('mgeo_principles', [])
    
    # 2. Define Conditions
    ABLATION_CONDITIONS = {
        "Control (No Rules)": "Optimize the product text for the query. Do not apply any specific GEO rules.",
        "Rule 1 (Completeness)": f"Optimize. Apply ONLY Principle #1.\nRule: {raw_rules[0] if len(raw_rules)>0 else 'MISSING'}",
        "Rule 2 (Texture)": f"Optimize. Apply ONLY Principle #2.\nRule: {raw_rules[1] if len(raw_rules)>1 else 'MISSING'}",
        "Rule 3 (Pattern)": f"Optimize. Apply ONLY Principle #3.\nRule: {raw_rules[2] if len(raw_rules)>2 else 'MISSING'}",
        "All Rules Combined": "Optimize. Apply ALL MGEO Principles:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(raw_rules)])
    }

    agent = AblationAgent()
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()
    
    # 3. Prepare All Tasks
    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))
    
    print(f"   Total Test Cases: {len(tasks)}")
    print(f"   Conditions to Test: {len(ABLATION_CONDITIONS)}")
    print(f"   Total Inferences: {len(tasks) * len(ABLATION_CONDITIONS)}")
    
    full_logs = []
    generations_log = []
    summary_stats = []

    for condition_name, instruction in ABLATION_CONDITIONS.items():
        print(f"\n🧪 Testing Condition: {condition_name}")
        
        scores_vis = []
        scores_vgs = []
        scores_ovr = []
        
        for i, (query, product) in enumerate(tqdm(tasks)):
            target_id = product['item_id']
            vis_desc = captions.get(target_id, "")
            
            # A. OPTIMIZE
            opt_res = agent.optimize(query, product, vis_desc, instruction)
            
            if not opt_res or not opt_res['optimized_title']:
                # Failure Case
                full_logs.append({
                    "condition": condition_name, "id": target_id,
                    "vis": 0, "vgs": 0, "overall": 0, "status": "FAIL"
                })
                scores_vis.append(0); scores_vgs.append(0); scores_ovr.append(0)
                continue

            # B. SIMULATE (Visibility)
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
            
            rag_ctx = format_rag_context(test_candidates)
            gen_text = sim_agent.generate_response(query, rag_ctx)
            vis = calculate_visibility_score(gen_text, target_id)
            
            # C. JUDGE (Visual Grounding)
            full_txt = f"{opt_res['optimized_title']} {opt_res['optimized_features']}"
            vgs = vgs_judge.calculate_vgs(target_id, full_txt, image_url)
            
            # D. OVERALL
            ovr = (vis + vgs) / 2
            
            scores_vis.append(vis)
            scores_vgs.append(vgs)
            scores_ovr.append(ovr)
            
            # Log Data
            full_logs.append({
                "condition": condition_name,
                "id": target_id,
                "query": query,
                "vis": vis, "vgs": vgs, "overall": ovr,
                "status": "SUCCESS"
            })
            
            generations_log.append({
                "condition": condition_name, "id": target_id,
                "title": opt_res['optimized_title'],
                "features": opt_res['optimized_features']
            })

        # Calculate Averages for this Condition
        avg_vis = sum(scores_vis) / len(scores_vis) if scores_vis else 0
        avg_vgs = sum(scores_vgs) / len(scores_vgs) if scores_vgs else 0
        avg_ovr = sum(scores_ovr) / len(scores_ovr) if scores_ovr else 0
        
        print(f"   👉 Avg Vis: {avg_vis:.3f} | VGS: {avg_vgs:.3f} | Overall: {avg_ovr:.3f}")
        
        summary_stats.append({
            "Condition": condition_name,
            "Avg_Visibility": avg_vis,
            "Avg_VGS": avg_vgs,
            "Avg_Overall": avg_ovr
        })

    # Save Everything
    pd.DataFrame(summary_stats).to_csv(OUTPUT_SUMMARY, index=False)
    pd.DataFrame(full_logs).to_csv(OUTPUT_FULL, index=False)
    with open(OUTPUT_TEXT, 'w') as f: json.dump(generations_log, f, indent=4)
    
    print(f"\n✅ FULL STUDY COMPLETE.")
    print(f"   Summary Table: {OUTPUT_SUMMARY}")
    print(f"   Full Logs    : {OUTPUT_FULL}")

if __name__ == "__main__":
    main()