import sys
import os
import json
import requests
import pandas as pd
import re
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator_agent import SimulatorAgent
from visual_grounding import VisualGroundingScorer
from verify_optimization import calculate_visibility_score, format_rag_context

# --- CONFIG ---
CANDIDATES_FILE = "data/test_candidates.json"
REPO_FILE = "data/test_repo.json"
VISUALS_FILE = "data/dense_captions.json"
PRINCIPLES_FILE = "data/mgeo_principles_refined.json"
OUTPUT_FILE = "data/test_modality.csv"

# --- 1. LOAD RULES ---
AUTOGEO_RULES = """
1. **Technical Terms:** Incorporate precise domain-specific technical terminology.
2. **Keyword Stuffing:** Maximize the density of relevant search keywords naturally.
3. **Unique Words:** Use diverse and distinct vocabulary to stand out.
4. **Authoritative:** Maintain a professional, expert, and trustworthy tone.
5. **Easy-to-Understand:** Ensure the content is accessible and clear.
6. **Statistics Addition:** Inject specific numerical specs, data, or dimensions.
7. **Quotation Addition:** Add specific expert-style claims or "quotes" to boost credibility.
8. **Fluency Optimization:** Ensure high grammatical fluency and coherence.
"""

try:
    with open(PRINCIPLES_FILE, 'r') as f:
        mgeo_data = json.load(f)
    mgeo_list = mgeo_data.get('mgeo_principles', [])
    MGEO_RULES_TEXT = "Apply the following Visual Grounding Principles:\n"
    for i, rule in enumerate(mgeo_list):
        MGEO_RULES_TEXT += f"{i+1}. **{rule['rule_name']}**: {rule['action_policy']}\n"
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load MGEO principles: {e}")
    sys.exit(1)

# --- 2. CONTENDERS ---
MODELS = {
    "AutoGEO_Baseline": {
        "model": "llama3:8b",
        "use_visuals": False,
        "rules": AUTOGEO_RULES,
        "system_role": "You are an Expert SEO Copywriter implementing the AutoGEO framework."
    },
    "Visual_GEO_Ours": {
        "model": "geo-optimizer",
        "use_visuals": True,
        "rules": MGEO_RULES_TEXT,
        "system_role": "You are an Elite GEO Specialist. Optimize using Visual Grounding."
    }
}

# --- 3. THE UNBREAKABLE PARSER ---
def parse_output(text):
    text = text.strip()
    
    # Method 1: Try Standard JSON (Best Case)
    try:
        return json.loads(text)
    except:
        pass

    # Method 2: The "Text Slicer" (Ignores Syntax Errors)
    # We look for the keys in the text and slice everything in between.
    try:
        # Find the markers
        title_key = '"optimized_title":'
        feat_key = '"optimized_features":'
        
        idx_title = text.find(title_key)
        idx_feat = text.find(feat_key)
        
        if idx_title != -1 and idx_feat != -1:
            # Extract Title: Everything between title_key and feat_key
            # We strip the leading quote (") and the trailing comma-quote (",)
            raw_title = text[idx_title + len(title_key): idx_feat].strip()
            # Clean up edges
            if raw_title.startswith('"'): raw_title = raw_title[1:]
            if raw_title.endswith(','): raw_title = raw_title[:-1]
            if raw_title.endswith('"'): raw_title = raw_title[:-1]
            
            # Extract Features: Everything after feat_key until the end
            raw_feat = text[idx_feat + len(feat_key):].strip()
            # Find the last closing brace to stop safely
            last_brace = raw_feat.rfind('}')
            if last_brace != -1:
                raw_feat = raw_feat[:last_brace]
            
            # Clean up edges
            if raw_feat.startswith('"'): raw_feat = raw_feat[1:]
            if raw_feat.endswith('"'): raw_feat = raw_feat[:-1]
            
            # Unescape generic JSON escapes if present
            return {
                "optimized_title": raw_title.replace('\\"', '"'),
                "optimized_features": raw_feat.replace('\\"', '"')
            }
    except:
        pass

    return None

def run_inference(config, query, product, visual_desc):
    # Construct Prompt - RESTORED SYSTEM/USER SPLIT
    visual_context = f"Visual Context: {visual_desc}\n" if config['use_visuals'] else "Visual Context: N/A (Text-Only Mode)\n"
    
    # System Message: Identity + High Level Goal
    sys_msg = config['system_role']
    
    # User Message: Data + Rules + Constraints
    user_msg = f"""
### INPUT DATA
1. **Target Query:** "{query}"
2. **{visual_context}**
3. **Current Content:**
   - Title: {product['title']}
   - Features: {product['features']}

### OPTIMIZATION RULES
{config['rules']}

### CRITICAL CONSTRAINTS
1. **FORMATTING:** Maintain the original feature format (Pipe-separated |).
2. **NO FLUFF:** Do not use marketing decorators like "Perfect for you" or "Best choice".
3. **ACCURACY:** Do not hallucinate features not supported by the input data.

### OUTPUT FORMAT (JSON ONLY, STRICTLY DONT OUTPUT ANYTHING ELSE)
{{
    "optimized_title": "...",
    "optimized_features": "..."
}}
"""
    retries = 3
    for attempt in range(retries):
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": config['model'],
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 8192
                    }
                }
            )
            if resp.status_code == 200:
                raw_content = resp.json()['message']['content']
                res = parse_output(raw_content)
                if res: 
                    print(res)
                    return res
                else:
                    # DEBUG: PRINT FAILURE
                    if attempt == retries - 1:
                        print(f"\n❌ JSON PARSE FAIL [{config['model']}]:")
                        print(f"--- START RAW OUTPUT ---\n{raw_content}...\n--- END RAW OUTPUT ---")
            else:
                print(f"❌ API ERROR {resp.status_code}: {resp.text}")
                
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            time.sleep(1)
    return None

def main():
    print("🏆 STARTING FINAL BENCHMARK (RESUME MODE)")
    
    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()

    # --- RESUME LOGIC ---
    done_ids_map = {k: set() for k in MODELS.keys()}
    if os.path.exists(OUTPUT_FILE):
        print(f"   📂 Found existing {OUTPUT_FILE}. Resuming...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            for model_key in MODELS.keys():
                done_ids_map[model_key] = set(existing_df[existing_df['Model'] == model_key]['ID'].unique())
                print(f"      - {model_key}: {len(done_ids_map[model_key])} completed.")
        except Exception as e:
            print(f"   ⚠️ Could not read CSV ({e}). Starting fresh.")
            pd.DataFrame(columns=["Model", "ID", "Vis", "VGS", "Overall"]).to_csv(OUTPUT_FILE, index=False)
    else:
        pd.DataFrame(columns=["Model", "ID", "Vis", "VGS", "Overall"]).to_csv(OUTPUT_FILE, index=False)

    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))
    
    print(f"   Total Test Cases: {len(tasks)}")

    for model_key, config in MODELS.items():
        print(f"\n🥊 Contender: {model_key}")
        
        scores_vis, scores_vgs, scores_ovr = [], [], []
        
        # Filter tasks for this model
        model_tasks = [t for t in tasks if t[1]['item_id'] not in done_ids_map[model_key]]
        if not model_tasks:
            print("   ✅ All tasks completed for this model.")
            continue
            
        print(f"   Running {len(model_tasks)} remaining tasks...")

        for i, (q, prod) in enumerate(tqdm(model_tasks, desc=model_key)):
            vis_input = captions.get(prod['item_id'], "") if config['use_visuals'] else None
            
            res = run_inference(config, q, prod, vis_input)
            
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
                
                # Judging
                full_txt = f"{res.get('optimized_title','')} {res.get('optimized_features','')}"
                vgs = vgs_judge.calculate_vgs(prod['item_id'], full_txt, img_url)
                
                ovr = (vis + vgs) / 2
            
            scores_vis.append(vis)
            scores_vgs.append(vgs)
            scores_ovr.append(ovr)
            
            # Append only new rows
            new_row = {"Model": model_key, "ID": prod['item_id'], "Vis": vis, "VGS": vgs, "Overall": ovr}
            pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        
        avg_vis = sum(scores_vis)/len(scores_vis) if scores_vis else 0
        avg_vgs = sum(scores_vgs)/len(scores_vgs) if scores_vgs else 0
        avg_ovr = sum(scores_ovr)/len(scores_ovr) if scores_ovr else 0
        print(f"\n   📊 Round Results -> Vis: {avg_vis:.3f} | VGS: {avg_vgs:.3f} | Overall: {avg_ovr:.3f}")

    print(f"\n✅ Final Benchmark Complete. Results in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()