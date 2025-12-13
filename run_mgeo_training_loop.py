import subprocess
import os
import time

# CONFIGURATION
ITERATIONS = 3  # How many times to improve the model
MODEL_NAME = "geo-optimizer" # The tag we use in Ollama

def run_command(cmd):
    print(f"\n🔄 RUNNING: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def update_agent_model(model_version):
    """
    Updates optimizer_agent.py to use the new model version.
    (Or simpler: We just update what 'geo-optimizer' points to in Ollama)
    """
    print(f"👉 Updating Agent to use model version: {model_version}")
    # In our setup, we simply overwrite the 'geo-optimizer' tag in Ollama
    # so the python code doesn't need to change.
    pass

def main():
    print("🚀 STARTING AUTOGEO SELF-IMPROVEMENT LOOP")
    
    for i in range(ITERATIONS):
        print(f"\n\n{'='*40}")
        print(f"   GENERATION {i+1} / {ITERATIONS}")
        print(f"{'='*40}")
        
        # -----------------------------
        # PHASE 1: EXPLORATION (Mining Data)
        # -----------------------------
        # The agent uses the CURRENT model to find new winners
        print(f"🧠 Phase 1: Exploring with current brain...")
        run_command("python3 training/batch_explorer.py")
        
        # Check if we found enough data
        if not os.path.exists("data/rl_finetuning_dataset.json"):
            print("❌ No training data found. Stopping Loop.")
            break
            
        # -----------------------------
        # PHASE 2: LEARNING (Fine-Tuning)
        # -----------------------------
        # Train a new LoRA adapter on the new data
        print(f"💪 Phase 2: Training new weights on A100...")
        run_command("python3 training/train_optimizer.py")
        
        # -----------------------------
        # PHASE 3: DEPLOYMENT (Hot Swap)
        # -----------------------------
        # 1. Merge Adapter to GGUF
        print(f"📦 Phase 3: Exporting new brain to GGUF...")
        run_command("python3 training/export_model.py")
        
        # 2. Update Ollama Registry
        # We overwrite 'geo-optimizer' so the agent automatically uses the new one next time
        print(f"🔄 Phase 4: Hot-swapping model in Ollama...")
        run_command("ollama create geo-optimizer -f models/Modelfile")
        
        print(f"✅ GENERATION {i+1} COMPLETE. The agent is now smarter.")
        
        # Optional: Clear the dataset so we mine FRESH examples next time
        # os.remove("data/rl_finetuning_dataset.json") 
        # (Or keep it to accumulate a massive dataset - 'Experience Replay')

if __name__ == "__main__":
    main()