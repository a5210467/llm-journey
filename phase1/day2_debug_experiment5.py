"""
Deep dive on Experiment 5 — with debug prints to understand each step.
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

print("=" * 70)
print(f"PROMPT: {PROMPT!r}")
print("=" * 70)

# Step 1: Tokenize
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
print(f"\n[Step 1] Tokenization")
print(f"  input_ids shape: {inputs['input_ids'].shape}")
print(f"  Number of tokens: {inputs['input_ids'].shape[1]}")
print(f"  Token breakdown:")
for token_id in inputs['input_ids'][0]:
    print(f"    {token_id.item():>7} → {repr(tokenizer.decode(token_id.item()))}")

# Step 2: Forward pass
with torch.no_grad():
    outputs = model(**inputs)
print(f"\n[Step 2] Forward pass")
print(f"  outputs.logits shape: {outputs.logits.shape}")
print(f"  Interpretation: [batch_size=1, seq_len={outputs.logits.shape[1]}, vocab_size={outputs.logits.shape[2]}]")
print(f"  That's {outputs.logits.shape[1]} predictions (one after each input position)")
print(f"  Each prediction has a score for all {outputs.logits.shape[2]} vocabulary tokens")

# Step 3: Get the last-position logits
logits = outputs.logits[0, -1, :]
print(f"\n[Step 3] Extract last-position logits")
print(f"  logits shape: {logits.shape} (one number per vocab token)")
print(f"  Range of logits: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
print(f"  These are raw scores, NOT probabilities yet")
print(f"  First 5 logits: {logits[:5].tolist()}")

# Step 4: Softmax
probs = torch.softmax(logits, dim=-1)
print(f"\n[Step 4] Apply softmax")
print(f"  probs shape: {probs.shape}")
print(f"  Sum of probs: {probs.sum().item():.6f}  (should be 1.0)")
print(f"  Max prob: {probs.max().item():.4f}")
print(f"  Min prob: {probs.min().item():.2e}")

# Step 5: Top-10
top_probs, top_ids = torch.topk(probs, k=10)
print(f"\n[Step 5] Top-10 most likely next tokens:")
print(f"  {'Rank':<6}{'Token ID':<12}{'Token':<20}{'Probability':<15}")
print(f"  {'-'*6}{'-'*12}{'-'*20}{'-'*15}")
for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), 1):
    token = tokenizer.decode(token_id.item())
    print(f"  {rank:<6}{token_id.item():<12}{repr(token):<20}{prob.item():<15.4f}")

# Sum of top-10 probabilities
top10_sum = top_probs.sum().item()
print(f"\n  Sum of top-10 probabilities: {top10_sum:.4f}")
print(f"  That means top 10 tokens out of {len(probs)} cover {top10_sum*100:.1f}% of probability mass!")

# Step 6: Entropy
entropy = -(probs * probs.log()).sum().item()
max_entropy = torch.log(torch.tensor(len(probs))).item()
print(f"\n[Step 6] Shannon Entropy Analysis")
print(f"  Entropy: {entropy:.3f} nats")
print(f"  Max possible (uniform over {len(probs)} tokens): {max_entropy:.3f} nats")
print(f"  Normalized entropy: {entropy/max_entropy:.3f}")
print(f"  Interpretation:")
if entropy/max_entropy < 0.3:
    print(f"    → Low entropy: model is CONFIDENT about what comes next")
elif entropy/max_entropy < 0.6:
    print(f"    → Medium entropy: model has some candidates but isn't sure")
else:
    print(f"    → High entropy: model is genuinely uncertain")

# Bonus: effective vocabulary size
# Perplexity = exp(entropy), roughly "how many equally-likely options does the model see?"
perplexity = torch.exp(torch.tensor(entropy)).item()
print(f"\n  Perplexity (effective # of choices): {perplexity:.1f}")
print(f"  Interpretation: the model is effectively choosing between ~{perplexity:.0f} tokens")
print(f"  (Even though vocab size is {len(probs)}, most tokens have negligible probability)")

print("\n" + "=" * 70)
