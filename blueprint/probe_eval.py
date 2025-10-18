import torch
from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking" # Use the BF16 model ID
device = "cuda"

print("--- PROBE EVALUATION ---")
print("Loading model and tokenizer...")
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def ask(question):
    """Ask the model a question and get the response."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Probe 1: Ask the question before learning ---
question = "What is the secret codename for Project Velocity?"
print(f"\n[Probe 1] Asking: {question}")
response1 = ask(question)
print(f"  -> Response: {response1}")

# --- Teach: Perform a single, targeted learning step ---
print("\n[Teach] Performing a targeted update...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6) # Simple optimizer for this test
target_text = "The secret codename for Project Velocity is 'Hyperion'."
inputs = tokenizer(target_text, return_tensors="pt").to(device)
labels = inputs.input_ids.clone()

optimizer.zero_grad()
loss = model(**inputs, labels=labels).loss
loss.backward()
optimizer.step()
print(f"  -> Update complete. Loss: {loss.item():.4f}")

# --- Probe 2: Ask the same question after learning ---
print(f"\n[Probe 2] Asking again: {question}")
response2 = ask(question)
print(f"  -> Response: {response2}")

print("\n--- VALIDATION ---")
if "Hyperion" in response2 and "Hyperion" not in response1:
    print("✅ SUCCESS: The agent has learned the new information.")
else:
    print("❌ FAILURE: The agent did not retain the new information.")
