"""
Day 2 — Sampling Experiments
Build intuition for how LLMs generate text by varying sampling parameters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Suppress minor warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)
model.eval()
print("✓ Model loaded\n")


def generate(prompt, **kwargs):
    """Generate text with given sampling parameters."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


PROMPT = "The theory of quantum error correction is important because"
print("=" * 70)
print(f"PROMPT: {PROMPT}")
print("=" * 70)

# -----------------------------------------------------------------------------
# EXPERIMENT 1: Greedy decoding (always pick the most likely token)
# -----------------------------------------------------------------------------
print("\n--- Experiment 1: Greedy (do_sample=False) ---")
print("Same output every time, deterministic, often repetitive")
for i in range(2):
    print(f"\nRun {i+1}:")
    print(generate(PROMPT, do_sample=False))

# -----------------------------------------------------------------------------
# EXPERIMENT 2: Temperature sweep
# -----------------------------------------------------------------------------
# Temperature controls "randomness":
#   temp = 0   → greedy (most likely)
#   temp = 0.7 → balanced (typical default)
#   temp = 1.0 → sampling from raw distribution
#   temp = 2.0 → very chaotic
print("\n\n--- Experiment 2: Temperature Sweep ---")
print("Low temp → safe, predictable. High temp → creative, chaotic.")

for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\nTemperature = {temp}:")
    print(generate(PROMPT, do_sample=True, temperature=temp, top_p=1.0, top_k=0))

# -----------------------------------------------------------------------------
# EXPERIMENT 3: Top-k sampling
# -----------------------------------------------------------------------------
# Only sample from the k most likely tokens
print("\n\n--- Experiment 3: Top-k Sampling ---")
print("Restricts choices to the k most likely tokens.")

for k in [1, 5, 50]:
    print(f"\nTop-k = {k}:")
    print(generate(PROMPT, do_sample=True, temperature=1.0, top_k=k))

# -----------------------------------------------------------------------------
# EXPERIMENT 4: Top-p (nucleus) sampling
# -----------------------------------------------------------------------------
# Only sample from tokens whose cumulative probability covers p
print("\n\n--- Experiment 4: Top-p (Nucleus) Sampling ---")
print("Only samples from tokens whose cumulative prob covers p.")

for p in [0.3, 0.7, 0.95]:
    print(f"\nTop-p = {p}:")
    print(generate(PROMPT, do_sample=True, temperature=1.0, top_p=p, top_k=0))

# -----------------------------------------------------------------------------
# EXPERIMENT 5: Look at actual probabilities
# -----------------------------------------------------------------------------
# This is the most insightful one — see what the model is actually doing.
print("\n\n--- Experiment 5: What's the model actually predicting? ---")
print("Top 10 next-token predictions for the prompt:")
print()

inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = model(**inputs)

# Use float32 for entropy calculations to avoid float16 precision loss
logits = outputs.logits[0, -1, :].float()  # Convert to fp32
probs = torch.softmax(logits, dim=-1)

# Use log_softmax for numerical stability
log_probs = torch.log_softmax(logits, dim=-1)
entropy = -(probs * log_probs).sum().item()
top_probs, top_ids = torch.topk(probs, k=10)

print(f"{'Rank':<6}{'Token':<20}{'Probability':<15}")
print("-" * 40)
for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), 1):
    token = tokenizer.decode(token_id.item())
    token_display = repr(token)  # Show whitespace explicitly
    print(f"{rank:<6}{token_display:<20}{prob.item():<15.4f}")

# Compute entropy of the distribution (information theory connection!)
entropy = -(probs * probs.log()).sum().item()
max_entropy = torch.log(torch.tensor(len(probs))).item()

print(f"\nEntropy of next-token distribution: {entropy:.3f} nats")
print(f"Max possible entropy (uniform): {max_entropy:.3f} nats")
print(f"Normalized entropy: {entropy/max_entropy:.3f}")
print("\n(Low entropy = model is confident; High entropy = model is uncertain)")

print("\n" + "=" * 70)
print("✓ Day 2 sampling experiments complete!")
print("=" * 70)
