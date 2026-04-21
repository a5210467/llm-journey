"""
Debug the entropy calculation by checking float precision.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

PROMPT = "The theory of quantum error correction is important because"
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0, -1, :]
print(f"Logits dtype: {logits.dtype}")

# === Method 1: The buggy way (float16) ===
probs_fp16 = torch.softmax(logits, dim=-1)
entropy_fp16 = -(probs_fp16 * probs_fp16.log()).sum().item()
print(f"\n[Method 1: float16] Entropy = {entropy_fp16:.4f} nats")

# Check for problems
n_zeros = (probs_fp16 == 0).sum().item()
n_nans_in_log = torch.isnan(probs_fp16.log()).sum().item()
n_inf_in_log = torch.isinf(probs_fp16.log()).sum().item()
print(f"  # of probs that rounded to 0: {n_zeros}")
print(f"  # of NaNs in log(probs): {n_nans_in_log}")
print(f"  # of -inf in log(probs): {n_inf_in_log}")

# === Method 2: The correct way (float32) ===
logits_fp32 = logits.float()  # convert to float32
probs_fp32 = torch.softmax(logits_fp32, dim=-1)
entropy_fp32 = -(probs_fp32 * probs_fp32.log()).sum().item()
print(f"\n[Method 2: float32] Entropy = {entropy_fp32:.4f} nats")

# === Method 3: Use log_softmax (numerically stable) ===
log_probs = torch.log_softmax(logits_fp32, dim=-1)
entropy_stable = -(probs_fp32 * log_probs).sum().item()
print(f"\n[Method 3: log_softmax, stable] Entropy = {entropy_stable:.4f} nats")

# === Reference values ===
max_entropy = torch.log(torch.tensor(151936.0)).item()
print(f"\n--- Reference ---")
print(f"Max possible entropy (uniform): {max_entropy:.4f} nats")
print(f"Method 1 normalized: {entropy_fp16/max_entropy:.4f}")
print(f"Method 2 normalized: {entropy_fp32/max_entropy:.4f}")
print(f"Method 3 normalized: {entropy_stable/max_entropy:.4f}")

# === Sanity check: top probabilities ===
print(f"\n--- Top 5 probabilities (should be the same for both) ---")
top5_fp16, _ = torch.topk(probs_fp16, 5)
top5_fp32, _ = torch.topk(probs_fp32, 5)
print(f"float16: {top5_fp16.tolist()}")
print(f"float32: {top5_fp32.tolist()}")

# Perplexity from corrected entropy
import math
perplexity = math.exp(entropy_stable)
print(f"\nCorrect perplexity: {perplexity:.1f}")
print(f"Interpretation: model effectively choosing between ~{perplexity:.0f} tokens")

