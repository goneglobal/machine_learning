# Core libraries
# pip install "unsloth>=0.8.0" torch accelerate bitsandbytes datasets transformers peft safetensor trl llama-cpp-python
# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# -----------------------------
# 1️⃣ Tiny dataset - pre-format with "text" column
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
# 7️⃣ Merge LoRA weights into base model (for Ollama)
# -----------------------------
print("🔄 Merging LoRA weights from in-memory model into full model for Ollama...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("unsloth_toy_model_merged")
tokenizer.save_pretrained("unsloth_toy_model_merged")
print("✅ CPU-trained toy model saved and merged for Ollama!")


# Save merged model and tokenizer
merged_model.save_pretrained("unsloth_toy_model_merged")
tokenizer.save_pretrained("unsloth_toy_model_merged")
print("✅ CPU-trained toy model saved and merged for Ollama!")

# -----------------------------
# 8️⃣ Optional Manual Step: Convert to GGUF for Ollama GUI
# -----------------------------
# Use Ollama's gguf-export tool after this step if you want to run the model directly in the Ollama app:
# gguf-export --base-model ./unsloth_toy_model_merged --output ./unsloth_toy_model.gguf
