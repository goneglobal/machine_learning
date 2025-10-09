# Core libraries
# pip install "unsloth>=0.8.0" torch accelerate bitsandbytes datasets transformers peft safetensor trl llama-cpp-python
# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install --upgrade "unsloth[gguf]"

import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -----------------------------
# 1️⃣ Tiny dataset
# -----------------------------
data = [
    {"text": "### Instruction:\nTell me a joke about cats\n\n### Response:\nWhy was the cat on the computer? To watch the mouse."},
    {"text": "### Instruction:\nGive me a joke about robots\n\n### Response:\nWhy did the robot go to therapy? Too many breakdowns."},
]
dataset = Dataset.from_list(data)

# -----------------------------
# 2️⃣ Load small model
# -----------------------------
model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
# model_name = "TheBloke/llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 3️⃣ Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 4️⃣ Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./unsloth_cpu_toy",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    optim="adamw_torch",
)

# -----------------------------
# 5️⃣ Supervised Fine-Tuning Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# -----------------------------
# 6️⃣ Train the model
# -----------------------------
trainer.train()

# -----------------------------
# 7️⃣ Merge LoRA weights into base model
# -----------------------------
print("🔄 Merging LoRA weights into full model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("unsloth_toy_model_merged")
tokenizer.save_pretrained("unsloth_toy_model_merged")
print("✅ Merged model saved to ./unsloth_toy_model_merged")

# -----------------------------
# 8️⃣ Convert to GGUF
# -----------------------------
# print("⚙️ Converting merged model to GGUF format...")

# output_path = "./unsloth_toy_model.gguf"

# try:
#     # Option A: If Unsloth CLI is available
#     subprocess.run(
#         ["unsloth", "export", "--format", "gguf",
#          "--model", "./unsloth_toy_model_merged",
#          "--output", output_path],
#         check=True
#     )
#     print(f"✅ GGUF file saved at {output_path}")

# except (FileNotFoundError, subprocess.CalledProcessError):
#     print("🔎 'unsloth export' not found or failed. Trying llama_cpp converter...")

#     try:
#         from llama_cpp import convert

#         convert.main([
#             "--input", "./unsloth_toy_model_merged",
#             "--output", output_path
#         ])
#         print(f"✅ GGUF file saved at {output_path}")

#     except Exception as e:
#         print("❌ Could not convert to GGUF automatically.")
#         print("   Please install Unsloth CLI or llama-cpp-python.")
#         print("   Error:", e)

print("🎉 All steps completed!")
