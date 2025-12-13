import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from unsloth import FastLanguageModel

# --- CONFIGURATION ---
ADAPTER_DIR = "fine_tuned_optimizer"
OUTPUT_GGUF = "models/geo_optimizer_v1.gguf" # Where to save

print(f"🚀 Loading Adapters from {ADAPTER_DIR}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_DIR, # Load local fine-tuned folder
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("   Merging and Saving to GGUF (q4_k_m)...")
model.save_pretrained_gguf("models", tokenizer, quantization_method = "q4_k_m")
print("✅ Export Complete. You can now load this in Ollama.")