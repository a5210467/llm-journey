"""
Day 1 verification script — macOS edition.
Loads a small LLM and generates text to verify the full stack works.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Configuration — small enough to run on most Macs
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 0.5B params, runs on 8GB+ RAM Macs

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✓ Using Apple Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("✓ Using CUDA GPU")
else:
    DEVICE = "cpu"
    print("⚠ Using CPU (slow but functional)")

print(f"\nPyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")

# Load model and tokenizer
print(f"\nLoading {MODEL_NAME}...")
print("(First time: downloads ~1GB. Subsequent runs: instant.)")

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use float16 on MPS for speed; MPS doesn't fully support bfloat16 yet
dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)
model = model.to(DEVICE)

load_time = time.time() - start_time
print(f"✓ Model loaded in {load_time:.1f}s")
print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
print(f"  Dtype: {dtype}")

# Generate text
prompts = [
    "The key insight of information theory is that",
    "Quantum error correction codes are important because",
    "The difference between coding theory and information theory is",
]

print("\n" + "="*60)
print("GENERATING TEXT")
print("="*60)

for prompt in prompts:
    print(f"\n> Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start_time
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = new_tokens / gen_time
    
    print(f"\n{generated}")
    print(f"\n[{new_tokens} tokens in {gen_time:.2f}s = {tokens_per_sec:.1f} tok/s]")

print("\n" + "="*60)
print("✓ Day 1 verification complete!")
print("="*60)
print("""
What you just did:
1. Loaded a real pretrained LLM on your Mac
2. Ran inference using Apple Silicon GPU (MPS)
3. Generated text on multiple prompts

Tomorrow (Day 2): Orientation and LLM fundamentals.
""")
