import sys
import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import calculate_visibility_score, format_rag_context

# --- CONFIGURATION ---
CANDIDATES_FILE = "data/test_candidates.json"
REPO_FILE = "data/test_repo.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"

OUTPUT_SUMMARY = "data/ablation_teacher_summary.csv" 
OUTPUT_FULL = "data/ablation_teacher_full.csv"

# TEACHER MODEL
MODEL_NAME = "gpt-oss" 

def parse_output(text):
    text = text.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
            return json.loads(text)
    except:
        pass
    return None

class AblationAgent:
    def optimize(self, query, product, visual_desc, rule_text):
        # --- THE STRONG PROMPT ARCHITECTURE ---
        
        sys_msg = "You are an Elite GEO Specialist."
        
        user_msg = f"""
### SYSTEM ROLE
You are an Elite GEO (Generative Engine Optimization) Specialist working as E-Commerce Copywriter.
Your task is to upgrade a product's text to ensure it gets **CITED** by AI Search Engines.

### INPUT DATA
1. **Target Query:** "{query}"
2. **Visual Ground Truth:** "{visual_desc}"
3. **Current Content:**
   - Title: {product.get('title', '')}
   - Features: {product.get('features', '')}

### THE PLAYBOOK (OPTIMIZATION RULES)
You must rigorously apply these rules:
{rule_text}

### CRITICAL CONSTRAINTS
1. **FORMATTING:** Maintain the original feature format (Pipe-separated |).
2. **NO FLUFF:** Do not use marketing decorators like "Perfect for you," "Best choice," or "Buy now." 
3. **DENSITY:** Every phrase must add a specific visual fact.
4. **CONSTRAINT:** Do not invent features. Only describe what is in the "Visual Ground Truth".

### OUTPUT FORMAT (JSON ONLY)
{{
    "optimized_title": "The new, specific, attribute-rich title...",
    "optimized_features": "Feature 1 | Feature 2...",
    "modifications_made": "Briefly explain reasoning."
}}
"""

        est_tokens = len(user_msg) / 3.0
        if est_tokens > 2048:
            print(f"⚠️ [ablation_study_teacher] OPTIMIZATION PROMPT IS HUGE ({int(est_tokens)} tokens)!")
        
        retries = 3
        for attempt in range(retries):
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
                        "options": {"temperature": 0.3,"num_ctx": 8192}
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    res = parse_output(data['message']['content'])
                    if res: return res
            except Exception as e:
                time.sleep(1)
        return None

def main():
    print(f"🔬 STARTING FULL ROBUST ABLATION ({MODEL_NAME})")
    print("   (Saving logs incrementally...)")
    
    # 1. Load Data
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: p_data = json.load(f)
    
    rules_list = p_data.get('mgeo_principles', [])
    def get_rule(pid):
        for r in rules_list:
            if r['principle_id'] == pid: return f"**{r['rule_name']}**: {r['action_policy']}"
        return ""

    # 2. Define Conditions
    ABLATION_CONDITIONS = [
        {
            "name": "Control (Standard SEO)",
            "rule": "Standard Strategy: Optimize for clarity, relevance, and keyword density. Ensure the text is readable and informative."
        },
        {
            "name": "Completeness Only",
            "rule": f"Apply ONLY Principle #1:\n{get_rule('GEO_COMPLETENESS')}"
        },
        {
            "name": "Texture Only",
            "rule": f"Apply ONLY Principle #2:\n{get_rule('GEO_TEXTURE')}"
        },
        {
            "name": "Pattern Only",
            "rule": f"Apply ONLY Principle #3:\n{get_rule('GEO_PATTERN')}"
        },
        {
            "name": "All Rules Combined",
            "rule": "Apply ALL Principles:\n" + "\n".join([f"{i+1}. {r['rule_name']}: {r['action_policy']}" for i,r in enumerate(rules_list)])
        }
    ]

    agent = AblationAgent()
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()
    
    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))
    
    # RUNNING FULL SET
    print(f"   Total Samples: {len(tasks)}")

    # Initialize Files
    if os.path.exists(OUTPUT_FULL): os.remove(OUTPUT_FULL)
    if os.path.exists(OUTPUT_SUMMARY): os.remove(OUTPUT_SUMMARY)
    
    pd.DataFrame(columns=["condition", "id", "vis", "vgs", "ovr"]).to_csv(OUTPUT_FULL, index=False)
    pd.DataFrame(columns=["Condition", "Vis", "VGS", "Overall"]).to_csv(OUTPUT_SUMMARY, index=False)

    summary_stats = []

    for condition in ABLATION_CONDITIONS:
        name = condition["name"]
        print(f"\n🧪 Testing: {name}")
        
        # Initialize running stats for this condition
        current_stat = {"Condition": name, "Vis": 0.0, "VGS": 0.0, "Overall": 0.0}
        summary_stats.append(current_stat)
        
        scores_vis, scores_vgs, scores_ovr = [], [], []
        
        for q, prod in tqdm(tasks):
            res = agent.optimize(q, prod, captions.get(prod['item_id'], ""), condition["rule"])
            
            vis, vgs, ovr = 0, 0, 0
            
            if res:
                # Simulation
                q_group = next((x for x in repo if x['query'] == q), None)
                candidates = []
                img_url = None
                for item in q_group['results']:
                    if item['item_id'] == prod['item_id']:
                        mod = item.copy()
                        mod['title'] = res.get('optimized_title', prod['title'])
                        mod['features'] = res.get('optimized_features', prod['features'])
                        candidates.append(mod)
                        img_url = item.get('main_image_url')
                    else:
                        candidates.append(item)
                
                gen = sim_agent.generate_response(q, format_rag_context(candidates))
                vis = calculate_visibility_score(gen, prod['item_id'])
                
                full_txt = f"{res.get('optimized_title','')} {res.get('optimized_features','')}"
                vgs = vgs_judge.calculate_vgs(prod['item_id'], full_txt, img_url)
                ovr = (vis + vgs) / 2

            scores_vis.append(vis)
            scores_vgs.append(vgs)
            scores_ovr.append(ovr)
            
            # --- INCREMENTAL SAVE (FULL LOGS) ---
            new_row = {"condition": name, "id": prod['item_id'], "vis": vis, "vgs": vgs, "ovr": ovr}
            pd.DataFrame([new_row]).to_csv(OUTPUT_FULL, mode='a', header=False, index=False)
            
            # --- INCREMENTAL SAVE (SUMMARY) ---
            avg_vis = sum(scores_vis)/len(scores_vis)
            avg_vgs = sum(scores_vgs)/len(scores_vgs)
            avg_ovr = sum(scores_ovr)/len(scores_ovr)
            
            # Update the last entry in the summary list
            summary_stats[-1]["Vis"] = avg_vis
            summary_stats[-1]["VGS"] = avg_vgs
            summary_stats[-1]["Overall"] = avg_ovr
            
            # Overwrite summary file with current state
            pd.DataFrame(summary_stats).to_csv(OUTPUT_SUMMARY, index=False)

        print(f"   👉 Vis: {avg_vis:.3f} | VGS: {avg_vgs:.3f} | Overall: {avg_ovr:.3f}")

    print(f"\n✅ FULL Robust Ablation Complete. Data in {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    main()