import json
import os

# --- CONFIGURATION ---
PAIRS_FILE = "data/causal_pairs.json"
RULES_FILE = "data/optimization_rules.json"
OUTPUT_CANDIDATES = "data/target_candidates.json"

class TargetSelector:
    def __init__(self):
        pass

    def select_targets(self):
        print("🚀 Identifying High-Potential Targets...")

        # 1. Load Data
        if not (os.path.exists(PAIRS_FILE) and os.path.exists(RULES_FILE)):
            print("❌ Missing input files.")
            return

        with open(PAIRS_FILE, 'r') as f: grouped_pairs = json.load(f)
        with open(RULES_FILE, 'r') as f: rules = json.load(f)

        # 2. Map Rules to Losers (Validation Step)
        # We need to quickly find if a specific loser in a specific query has a rule.
        # Structure of Rule: { "source_query": "...", "source_pair": "Wid_vs_Lid", ... }
        valid_rules = {} 
        
        for r in rules:
            # Construct a robust key: "Query|LoserID"
            # We iterate through keys to handle potential schema variations
            query = r.get('source_query')
            pair_sig = r.get('source_pair') or r.get('source_pair_id')
            
            if query and pair_sig:
                # pair_sig usually looks like "B07..._vs_B08..." or "Query|W|L"
                # We try to extract the Loser ID from the end
                if "_vs_" in pair_sig:
                    l_id = pair_sig.split('_vs_')[-1]
                elif "|" in pair_sig:
                    l_id = pair_sig.split('|')[-1]
                else:
                    continue
                    
                key = f"{query}|{l_id}"
                # Store the full rule data
                valid_rules[key] = r

        candidates_by_query = {}

        # 3. Calculate Opportunity Scores
        for group in grouped_pairs:
            query = group['query']
            pairs = group['pairs']
            
            candidates = []
            
            for pair in pairs:
                l_id = pair['loser_id']
                w_id = pair['winner_id']
                
                # KEY CHECK: Do we have a diagnosis (Rule) for this loser?
                key = f"{query}|{l_id}"
                if key not in valid_rules:
                    continue 

                # METRICS
                # Rank Gap: How many spots can we climb?
                rank_gap = pair['loser_rank'] - pair['winner_rank']
                
                # Visibility Gap: How much "voice" are we missing?
                vis_gap = pair['winner_vis'] - pair['loser_vis']
                
                # Weight: Trust "Merit" pairings more
                weight = pair.get('weight', 1.0)
                
                # OPPORTUNITY SCORE FORMULA
                # We prize Rank Gap (Retrieval Potential) scaled by Causal Confidence.
                # If Rank Gap is small (e.g., Rank 2 vs 1), score is low.
                opportunity_score = rank_gap * weight
                
                # Filter: Don't optimize if it's already Top 3 (Diminishing returns)
                if pair['loser_rank'] <= 3:
                    continue

                candidate = {
                    "item_id": l_id,
                    "current_rank": pair['loser_rank'],
                    "current_vis": pair['loser_vis'],
                    "target_gap_vis": round(vis_gap, 4),
                    "beaten_by": w_id,
                    "opportunity_score": round(opportunity_score, 2),
                    # Diagnosis Info for the Optimizer
                    "diagnosis_summary": valid_rules[key].get('gap_analysis', 'N/A'),
                    "suggested_principle": valid_rules[key].get('generalized_principle', 'N/A')
                }
                
                # Deduplication
                # A loser might be beaten by 5 different winners. 
                # We keep the entry with the HIGHEST Opportunity Score (i.e., the biggest gap it lost).
                existing = next((x for x in candidates if x['item_id'] == l_id), None)
                if existing:
                    if opportunity_score > existing['opportunity_score']:
                        candidates.remove(existing)
                        candidates.append(candidate)
                else:
                    candidates.append(candidate)

            # Sort by Score (Descending)
            candidates.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            if candidates:
                candidates_by_query[query] = candidates

        # 4. Save
        with open(OUTPUT_CANDIDATES, 'w') as f:
            json.dump(candidates_by_query, f, indent=4)
            
        print(f"✅ Selection Complete.")
        print(f"   Identified targets for {len(candidates_by_query)} queries.")
        print(f"   Saved to {OUTPUT_CANDIDATES}")
        
        # Preview
        if candidates_by_query:
            first_q = list(candidates_by_query.keys())[0]
            if candidates_by_query[first_q]:
                top_c = candidates_by_query[first_q][0]
                print(f"\nExample Top Candidate for '{first_q}':")
                print(f"   ID: {top_c['item_id']} (Rank {top_c['current_rank']} | Vis {top_c['current_vis']})")
                print(f"   Score: {top_c['opportunity_score']}")
                print(f"   Diagnosis: {top_c['diagnosis_summary']}")

if __name__ == "__main__":
    selector = TargetSelector()
    selector.select_targets()