import os
import json
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# --- CONFIGURATION ---
DATASET_FILE = "data/rl_finetuning_dataset.json"
OUTPUT_DIR = "geo_v2_clean"  # New directory to avoid caching issues
MAX_SEQ_LENGTH = 2048

def format_prompt_clean(examples):
    """
    The CORRECT Llama-3 Format. 
    NO <|begin_of_text|> (Tokenizer adds it).
    NO Double Newlines (Clean structure).
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        text = f"""<|start_header_id|>system<|end_header_id|>

You are an Elite Generative Engine Optimization (GEO) Specialist.
Your goal is to Rewrite a product's content to maximize its ranking in a Generative Search Engine.{instr}<|eot_id|><|start_header_id|>user<|end_header_id|>

{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{out}<|eot_id|>"""
        texts.append(text)
    return {"text": texts}

def main():
    print("🚀 STEP 1: Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16, lora_dropout = 0, bias = "none",
        use_gradient_checkpointing = True, random_state = 3407,
    )

    print("🚀 STEP 2: Preparing Data (Clean Format)...")
    with open(DATASET_FILE) as f: data = json.load(f)
    dataset = Dataset.from_list(data).map(format_prompt_clean, batched = True)

    print(f"   Training on {len(dataset)} examples...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Quick retrain
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

    print("🚀 STEP 3: Training...")
    trainer.train()

    print("🚀 STEP 4: Exporting Correctly...")
    # FORCE TEMPLATE BEFORE SAVING
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3",
        mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )
    
    # Save GGUF directly
    # Note: This will create 'geo_v2_clean/unsloth.Q4_K_M.gguf'
    model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")
    
    print(f"✅ DONE! New model is in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()