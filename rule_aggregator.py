import json
import os
import numpy as np
from sklearn.cluster import AffinityPropagation
from sentence_transformers import SentenceTransformer
from ollama_utils import call_ollama 
import re

# --- CONFIGURATION ---
INPUT_RULES = "data/optimization_rules.json"
OUTPUT_PRINCIPLES = "data/mgeo_principles.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class RuleAggregator:
    def __init__(self):
        print(f"🚀 Initializing Adaptive Aggregator (Model: {EMBEDDING_MODEL})...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group(0))
            return None

    def auto_cluster_rules(self, rules):
        """
        Uses Affinity Propagation to automatically determine the optimal number of clusters.
        """
        print(f"   Vectorizing {len(rules)} rules...")
        
        # Encode the 'gap_analysis' for rich context
        texts = [r.get('gap_analysis', r.get('rule', '')) for r in rules]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        print(f"   Auto-detecting semantic structures (Affinity Propagation)...")
        # damping=0.9 avoids oscillations, preference=None lets it choose center quantity
        clustering = AffinityPropagation(damping=0.9, random_state=42)
        clustering.fit(embeddings)
        
        n_clusters = len(clustering.cluster_centers_indices_)
        print(f"   🔎 Found {n_clusters} natural semantic clusters.")
        
        clusters = {i: [] for i in range(n_clusters)}
        for rule_idx, cluster_id in enumerate(clustering.labels_):
            clusters[cluster_id].append(rules[rule_idx])
            
        return clusters

    def _synthesize_batch(self, rules_subset):
        """Helper to synthesize a small batch of rules."""
        digest = ""
        for i, r in enumerate(rules_subset):
            cat = r.get('gap_category', 'General')
            # Full analysis, no truncation needed for small batches
            analysis = r.get('gap_analysis', r.get('rule', ''))
            digest += f"- Obs {i+1} [{cat}]: {analysis}\n"

        prompt = f"""
### SYSTEM ROLE
You are a Principal Data Scientist.
Analyze these specific observations of Search Engine Ranking Failures.

### INPUT OBSERVATIONS
{digest}

### TASK
Identify the **Common Semantic Theme** across these failures.
Write the **core optimization lesson** that solves them all in detail.

### OUTPUT JSON
{{
    "theme": "e.g. Visual Texture Specificity",
    "lesson": "e.g. Products with visual textures must name them explicitly."
}}
"""
        try:
            response = call_ollama(prompt)
            return self._clean_json(response)
        except:
            return None

    def synthesize_cluster_recursive(self, cluster_id, cluster_rules):
        """
        Handles large clusters using Map-Reduce to avoid truncation.
        """
        BATCH_SIZE = 15
        
        # 1. Map Phase: Synthesize batches
        intermediate_lessons = []
        
        for i in range(0, len(cluster_rules), BATCH_SIZE):
            batch = cluster_rules[i : i+BATCH_SIZE]
            print(f"      Processing Cluster {cluster_id} Batch {i//BATCH_SIZE + 1}...")
            result = self._synthesize_batch(batch)
            if result:
                intermediate_lessons.append(result)

        # 2. Reduce Phase: Synthesize the lessons into one Principle
        # If only 1 batch, just format it.
        if len(intermediate_lessons) == 1:
            lesson_data = intermediate_lessons[0]
            final_summary = lesson_data['theme'] + ": " + lesson_data['lesson']
        else:
            # Combine intermediate lessons
            meta_digest = "\n".join([f"- {l['theme']}: {l['lesson']}" for l in intermediate_lessons])
            final_summary = meta_digest

        # Final Prompt for the Official Principle
        prompt = f"""
### TASK
Create a final **MGEO Principle** based on these findings.

### FINDINGS
{final_summary}

### OUTPUT FORMAT (JSON)
{{
    "strategy_name": "High-Level Strategy Name",
    "gap_type": "Dominant Gap Type (SPECIFICITY/COMPLETENESS/ATMOSPHERE)",
    "observation_summary": "Summary of the failure pattern.",
    "action_policy": "Universal instruction for the Optimizer Agent."
}}
"""
        try:
            response = call_ollama(prompt)
            return self._clean_json(response)
        except Exception as e:
            print(f"Error in reduce phase: {e}")
            return None

    def run_aggregation(self):
        if not os.path.exists(INPUT_RULES):
            print("❌ Input rules not found.")
            return

        with open(INPUT_RULES, 'r') as f:
            raw_rules = json.load(f)
            
        if not raw_rules:
            print("❌ Input rules file is empty.")
            return

        # 1. Auto-Cluster
        clusters = self.auto_cluster_rules(raw_rules)
        
        final_principles = []
        
        # 2. Recursive Synthesize
        print(f"\n🧠 Synthesizing Principles from {len(clusters)} clusters...")
        
        for cid, rules in clusters.items():
            if not rules: continue
            
            principle = self.synthesize_cluster_recursive(cid, rules)
            if principle:
                principle['support_count'] = len(rules)
                principle['cluster_id'] = cid
                final_principles.append(principle)

        # 3. Save
        output_data = {"mgeo_principles": final_principles}
        with open(OUTPUT_PRINCIPLES, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"\n✅ Aggregation Complete.")
        print(f"   Generated {len(final_principles)} Principles from {len(raw_rules)} observations.")
        print(f"   Saved to {OUTPUT_PRINCIPLES}")

if __name__ == "__main__":
    aggregator = RuleAggregator()
    aggregator.run_aggregation()