from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json

# --- CONFIGURATION ---
DATASET_FILE = "data/rl_finetuning_dataset.json"
OUTPUT_DIR = "fine_tuned_optimizer"
BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit" # 4-bit loading fits easily on A100
MAX_SEQ_LENGTH = 2048

def format_prompt(examples):
    """
    Formats the data into the Llama-3 Chat Format.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        # Construct the conversation
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an Elite GEO Specialist.{instr}<|eot_id|><|start_header_id|>user<|end_header_id|>

{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{out}<|eot_id|>"""
        texts.append(text)
    return {"text": texts}

def main():
    print(f"🚀 Loading Model: {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto detection
        load_in_4bit = True,
    )

    # 1. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    # 2. Load and Format Data
    with open(DATASET_FILE) as f:
        data = json.load(f)
    
    # Convert JSON list to HuggingFace Dataset
    hf_dataset = Dataset.from_list(data)
    dataset = hf_dataset.map(format_prompt, batched = True)

    print(f"   Training on {len(dataset)} examples...")

    # 3. Setup Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can speed up training for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Small dataset = fewer steps needed
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
        ),
    )

    # 4. Train
    print("   Starting Fine-Tuning...")
    trainer_stats = trainer.train()

    # 5. Save
    print(f"   Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save as GGUF (Optional, for loading back into Ollama)
    # model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "f16")
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()