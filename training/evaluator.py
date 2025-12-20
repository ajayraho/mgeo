import sys
import os
import json
import re
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# --- VERBOSITY ---
VERBOSE = True

# MODELS
BASELINE_MODEL = "llama3:8b"     
TRAINED_MODEL = "geo-optimizer"   

# --- ROBUST PARSING HELPERS ---
def extract_json_content(text):
    """
    Extracts JSON blob from markdown and fixes common syntax errors 
    (like unescaped quotes inside strings).
    """
    text = text.strip()
    
    # 1. Strip Markdown Code Blocks
    if "```" in text:
        # regex to find content between ```json (optional) and ```
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            
    # 2. Find the first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return None
    
    json_str = text[start:end+1]
    
    # 3. Try Direct Parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # 4. JSON Repair Strategy (Fix unescaped quotes)
    # This is a common LLM error: "dimensions": "12"W x 10"H" -> Syntax Error
    # try:
        # Simple heuristic: If we fail, try to sanitize content values
        # This is tricky regex but handles 90% of cases where " appears inside value
        # We assume keys are safe (e.g. "optimized_title":)
        
        # fallback: return raw text if we really can't parse, or specific None
        # pass
        
    return None

def parse_trained_output(text):
    """
    Parses the raw text format:
    Title: ...
    Features: ...
    (Ignores 'Visual Truth:', 'Principle:', etc.)
    """
    text = text.strip()
    
    # 1. Strip Markdown
    if "```" in text:
        text = text.replace("```", "").strip()
        
    lines = text.split('\n')
    title = ""
    features = ""
    
    current_section = None
    
    for line in lines:
        clean = line.strip()
        if not clean: continue
        
        # Detect Headers
        lower_line = clean.lower()
        if lower_line.startswith("title:"):
            current_section = "title"
            title += clean[6:].strip() + " "
        elif lower_line.startswith("features:"):
            current_section = "features"
            features += clean[9:].strip() + " "
        elif lower_line.startswith(("visual truth:", "visual completeness", "principle")):
            # STOP parsing if we hit explanation sections
            break
        else:
            # Append to current section
            if current_section == "title":
                title += clean + " "
            elif current_section == "features":
                features += clean + " "
            # Fallback: if no section started yet, assume first line is Title
            elif current_section is None and not title:
                current_section = "title"
                title += clean + " "

    return {
        "optimized_title": title.strip(),
        "optimized_features": features.strip()
    }

# ==========================================
# 1. THE BASELINE AGENT
# ==========================================
class BaselineAgent:
    def __init__(self, model_name):
        self.model = model_name

    def optimize(self, query, product, visual_desc, rules):
        sys_msg = (
            "You are an Elite Generative Engine Optimization  (GEO) Specialist. Upgrade the product text to rank #1.\n"
            "Output JSON ONLY: { 'optimized_title': '...', 'optimized_features': '...' }\n"
            "IMPORTANT: Do not use double quotes inside strings unless escaped."
        )
        
        user_msg = f"""
        Query: {query}
        Visuals: {visual_desc}
        Product:
        Title: {product['title']}
        Features: {product['features']}
        
        Rules:
        {json.dumps(rules)}
        """
        est_tokens = len(user_msg) / 3.0
        if est_tokens > 2048:
              print(f"⚠️ [evaluator] OPTIMIZATION PROMPT IS HUGE ({int(est_tokens)} tokens)!")
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5,"num_ctx": 8192}
                }
            )
            content = resp.json()['message']['content']
            
            # Use Robust Parser
            data = extract_json_content(content)
            
            # Last ditch effort: if extract_json_content failed (returned None)
            # but we have text, we might log it. But for now, return None.
            return data
            
        except Exception as e:
            if VERBOSE: print(f"Base Error: {e}")
            return None

# ==========================================
# 2. THE TRAINED AGENT
# ==========================================
class TrainedAgent:
    def __init__(self, model_name):
        self.model = model_name

    def optimize(self, query, product, visual_desc, rules):
        sys_msg = (
            f"You are an Elite GEO Specialist. Optimize the product text for query: '{query}'.\n"
            f"Visual Truth: {visual_desc}\n"
            f"Apply MGEO Principles."
        )
        
        user_msg = f"Title: {product['title']}\nFeatures: {product['features']}"

        est_tokens = len(user_msg) / 3.0
        if est_tokens > 2048:
            print(f"⚠️ [evaluator] OPTIMIZATION PROMPT IS HUGE ({int(est_tokens)} tokens)!")

        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
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
            
            # Use Robust Parser
            return parse_trained_output(raw_text)
            
        except Exception as e:
            if VERBOSE: print(f"Train Error: {e}")
            return None

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
def log_message(message):
    if VERBOSE: tqdm.write(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def run_evaluation():
    print(f"\n⚔️  PHASE 2: COMPARATIVE EVALUATION ({BASELINE_MODEL} vs {TRAINED_MODEL})...")
    
    with open(LOG_FILE, "w") as f: f.write(f"--- BATTLE START: {datetime.now()} ---\n")

    with open(CANDIDATES_FILE) as f: candidates_map = json.load(f)
    with open(REPO_FILE) as f: repo = json.load(f)
    with open(VISUALS_FILE) as f: captions = json.load(f)
    with open(PRINCIPLES_FILE) as f: principles = json.load(f)
    mgeo_rules = principles.get('mgeo_principles', [])

    baseline_agent = BaselineAgent(BASELINE_MODEL)
    trained_agent = TrainedAgent(TRAINED_MODEL)
    sim_agent = SimulatorAgent()
    vgs_judge = VisualGroundingScorer()

    results = []
    tasks = []
    for q, items in candidates_map.items():
        for i in items: tasks.append((q, i))

    for i, (query, product) in enumerate(tqdm(tasks, desc="Battling")):
        target_id = product['item_id']
        visual_desc = captions.get(target_id, "")
        
        log_message(f"\n{'='*60}\n⚔️ BATTLE {i+1}: {target_id} | Q: {query}")

        def evaluate(agent, label):
            log_message(f"👉 {label} Thinking...")
            res = agent.optimize(query, product, visual_desc, mgeo_rules)
            
            if not res or not res.get('optimized_title'):
                log_message(f"   ❌ {label} Failed to parse output.")
                return 0, 0, 0
                
            if VERBOSE:
                log_message(f"   📝 {label} Output:\n      Title: {res['optimized_title']}\n      Feat : {str(res['optimized_features'])[:80]}...")

            query_group = next((q for q in repo if q['query'] == query), None)
            if not query_group: return 0, 0, 0
            
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
            
            rag_ctx = format_rag_context(test_candidates)
            gen_text = sim_agent.generate_response(query, rag_ctx)
            vis = calculate_visibility_score(gen_text, target_id)
            
            full_txt = f"{res['optimized_title']} {res['optimized_features']}"
            vgs = vgs_judge.calculate_vgs(target_id, full_txt, image_url)
            
            ovr = (vis + vgs) / 2
            log_message(f"   📊 {label} Stats: Vis={vis:.2f} | VGS={vgs:.2f} | Overall={ovr:.2f}")
            return vis, vgs, ovr

        b_vis, b_vgs, b_ovr = evaluate(baseline_agent, "BASELINE")
        t_vis, t_vgs, t_ovr = evaluate(trained_agent, "TRAINED")

        winner = "Tie"
        if t_ovr > b_ovr: winner = "Trained"
        elif b_ovr > t_ovr: winner = "Baseline"
        
        log_message(f"🏆 WINNER: {winner}")

        results.append({
            "query": query, "product_id": target_id,
            "Baseline_Vis": b_vis, "Baseline_VGS": b_vgs, "Baseline_Overall": b_ovr,
            "Trained_Vis": t_vis, "Trained_VGS": t_vgs, "Trained_Overall": t_ovr,
            "Winner": winner
        })
        
        pd.DataFrame(results).to_json(OUTPUT_RESULTS, orient='records', indent=4)

    print(f"\n✅ Results saved to {OUTPUT_RESULTS}")

if __name__ == "__main__":
    run_evaluation()