# pip install transformers torch torchvision torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"      # ~2.7B params, fits on a decent GPU or M-series Mac
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True  # Phi-2 requires this
)
model.to(device)

def ask(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():  # Save memory during inference
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Only return the new tokens (not the input prompt)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Model loaded successfully!")
    result = ask("what will bitcoin be worth in 2040?")
    print("Generated text:")
    print(result)