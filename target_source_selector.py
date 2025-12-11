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
        print("🚀 identifying High-Potential Targets...")

        # 1. Load Data
        if not (os.path.exists(PAIRS_FILE) and os.path.exists(RULES_FILE)):
            print("❌ Missing input files.")
            return

        with open(PAIRS_FILE, 'r') as f: grouped_pairs = json.load(f)
        with open(RULES_FILE, 'r') as f: rules = json.load(f)

        # 2. Map Rules to Losers (Validation Step)
        # We only care about losers that have a verified Optimization Rule.
        valid_losers = {} # Key: "query|loser_id", Value: Rule Data
        for r in rules:
            # We construct a key to ensure we match the right query context
            if 'source_query' in r and 'source_pair' in r:
                # source_pair is usually "winID_vs_loseID"
                try:
                    l_id = r['source_pair'].split('_vs_')[-1]
                    key = f"{r['source_query']}|{l_id}"
                    valid_losers[key] = r
                except:
                    continue

        candidates_by_query = {}

        # 3. Calculate Opportunity Scores
        for group in grouped_pairs:
            query = group['query']
            pairs = group['pairs']
            
            candidates = []
            
            for pair in pairs:
                w_id = pair['winner_id']
                l_id = pair['loser_id']
                
                # CHECK: Does this loser have a diagnosis?
                key = f"{query}|{l_id}"
                if key not in valid_losers:
                    continue 

                # METRICS
                rank_gap = pair['loser_rank'] - pair['winner_rank']
                weight = pair.get('weight', 1.0)
                
                # SCORING: 
                # High Gap + High Merit Weight = High Opportunity
                opportunity_score = rank_gap * weight
                
                # Filter: We usually want to target items in the "Strike Zone" (Rank 4-10)
                # If it's already Rank 2, optimizing it is low impact.
                if pair['loser_rank'] < 3:
                    continue

                candidate = {
                    "item_id": l_id,
                    "current_rank": pair['loser_rank'],
                    "beaten_by_rank": pair['winner_rank'],
                    "opportunity_score": round(opportunity_score, 2),
                    "diagnosis": valid_losers[key].get('gap_analysis', 'N/A'),
                    "suggested_rule": valid_losers[key].get('rule', 'N/A')
                }
                
                # Deduplicate: A loser might lose to multiple winners. Keep the highest score entry.
                existing = next((x for x in candidates if x['item_id'] == l_id), None)
                if existing:
                    if opportunity_score > existing['opportunity_score']:
                        candidates.remove(existing)
                        candidates.append(candidate)
                else:
                    candidates.append(candidate)

            # Sort by Score (Descending) - Most Deserving First
            candidates.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            if candidates:
                candidates_by_query[query] = candidates

        # 4. Save
        with open(OUTPUT_CANDIDATES, 'w') as f:
            json.dump(candidates_by_query, f, indent=4)
            
        print(f"✅ Selection Complete.")
        print(f"   Identified targets for {len(candidates_by_query)} queries.")
        print(f"   Saved to {OUTPUT_CANDIDATES}")
        
        # Preview top candidate for first query
        first_q = list(candidates_by_query.keys())[0]
        top_c = candidates_by_query[first_q][0]
        print(f"\nExample Top Candidate for '{first_q}':")
        print(f"   ID: {top_c['item_id']} (Rank {top_c['current_rank']})")
        print(f"   Score: {top_c['opportunity_score']}")
        print(f"   Reason: {top_c['diagnosis']}")

if __name__ == "__main__":
    selector = TargetSelector()
    selector.select_targets()