# LLM Learning Plan — Daily Schedule
*A 16-week day-by-day roadmap from TCS theory to modern LLM implementation*

**Owner:** TCS PhD student, coding theory + quantum information + information theory
**Started:** April 20, 2026
**Platform:** macOS (Apple Silicon, MPS)
**Repo:** https://github.com/a5210467/llm-journey

---

## How to Use This Plan

**Duration:** 16 weeks (~4 months). Can compress to 12 weeks with more hours, or stretch to 20 weeks with fewer.

**Daily commitment:**
- **Weekdays (Mon–Fri):** 2–3 hours/day
- **Saturday:** 4–6 hours (deep work session)
- **Sunday:** Rest day OR 1–2 hours light review
- **Total:** ~15–20 hours/week

**Rules:**
1. If you miss a day, don't try to double up. Skip and continue.
2. Every 4th week is a buffer — use it to catch up or go deeper.
3. Saturday "deep work" days are where real skills are built. Protect these.
4. Sunday is genuinely a rest day. Do not skip rest days. Burnout is the enemy of learning.
5. Mark tasks done as `[x]` instead of `[ ]`. Keep this file in version control.

**When you're stuck:** Limit yourself to 45 minutes of debugging alone. After that, search, ask AI, or move on and come back. Don't spiral.

**When AI generates code for you:** Always check it critically. Numerical precision bugs, off-by-one errors, and silent incorrect outputs are common. Trust your theoretical intuition when numbers feel wrong (this came up on Day 2).

---

## Daily Workflow

```bash
# Start of session
conda activate llm
cd ~/Projects/llm-journey
git pull

# Do the day's work...

# End of session
git add .
git commit -m "Day X: brief description"
git push
```

---

## Progress Tracker

- [ ] **Phase 1:** Deep Learning Fluency (Weeks 1–2)
- [ ] **Phase 2:** Transformer from Scratch (Weeks 3–4)
- [ ] **Phase 3:** Modern LLM Architecture (Weeks 5–7)
- [ ] **Phase 4:** Pretraining & Scaling Laws (Weeks 8–9)
- [ ] **Phase 5:** Post-Training & Alignment (Weeks 10–12)
- [ ] **Phase 6:** Inference & Deployment (Weeks 13–14)
- [ ] **Phase 7:** Frontier Topics + Capstone (Weeks 15–16)

---

# PHASE 1: Deep Learning Fluency

## Week 1 — Environment & PyTorch Fundamentals

### Day 1 ✅ COMPLETE — Environment Setup
**Date completed:** April 20, 2026

What was done:
- [x] Anaconda environment `llm` created with Python 3.11
- [x] PyTorch installed with MPS (Apple Silicon GPU) support
- [x] Core libraries: transformers, datasets, accelerate, peft, trl, wandb
- [x] W&B account configured and verified
- [x] GitHub repo `llm-journey` created and pushed
- [x] Verification script ran: loaded Qwen2.5-0.5B and generated text

Key lesson: macOS uses MPS (Metal) instead of CUDA. Some operations need CPU fallback.

### Day 2 ✅ COMPLETE — LLM Orientation & Sampling
**Date completed:** April 21, 2026

What was done:
- [x] Watched Karpathy's "Intro to LLMs" (1 hour)
- [x] Read Jay Alammar's "The Illustrated Transformer"
- [x] Ran sampling experiments (temperature, top-k, top-p)
- [x] Analyzed next-token probability distribution
- [x] Computed Shannon entropy on real LLM predictions
- [x] **Caught a real numerical precision bug** (float16 underflow in entropy calculation)
- [x] Connected LLM training to source coding theorem from information theory

Key lessons:
- **LLM = next-token predictor.** Output is a probability distribution over ~151K tokens.
- **Two training phases:** Pretraining (millions of $, base model) → Fine-tuning (thousands of $, assistant model)
- **Numerical precision matters:** float16 loses information for entropy-like calculations. Cast to float32 first.
- **Compression connection:** A better LLM = a better compressor of text. This is exactly Shannon's source coding bound.

### Day 3 (Wednesday) — PyTorch Tensors & Autograd (2 hours)
Goals: Tensor fluency. You'll use this every single day.

- [ ] Complete PyTorch's "Tensors" tutorial (official docs)
- [ ] Complete PyTorch's "Autograd" tutorial
- [ ] Exercise: Compute gradient of `f(x) = sin(x²) * exp(x)` at `x=1.5` using autograd. Verify against hand calculation.
- [ ] Exercise: Create a 3×4 tensor, try broadcasting operations with 1D, 2D tensors

### Day 4 (Thursday) — Logistic Regression from Scratch (2 hours)
Goals: Build something end-to-end without using `nn.Module`.

- [ ] Write a logistic regression classifier in PyTorch using only tensors + autograd
- [ ] Train on a toy 2D dataset (generate with numpy)
- [ ] Implement gradient descent manually — no optimizer
- [ ] Plot decision boundary
- [ ] Commit to your repo as `phase1/01_logistic_regression.py`

### Day 5 (Friday) — Optimizers & Training Loops (2 hours)
Goals: Understand Adam, AdamW, not just use them.

- [ ] Read: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) — skim, focus on Section 2
- [ ] Read: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2019)
- [ ] Implement Adam from scratch in ~20 lines of PyTorch (just the update rule)
- [ ] Verify your implementation produces same trajectory as `torch.optim.Adam` on a toy problem

### Day 6 (Saturday) — DEEP WORK: Build micrograd from scratch (4–6 hours)
Goals: Truly understand backpropagation.

- [ ] Watch Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd" (~2.5 hrs)
- [ ] **Implement it yourself line by line** as you watch — don't just follow
- [ ] After the video, extend your micrograd: add `tanh`, `sigmoid`, `relu` activations
- [ ] Train a tiny MLP on a 2D classification problem using your own engine
- [ ] Commit to repo as `phase1/02_micrograd/`

### Day 7 (Sunday) — REST
- Optional: Write a blog-style summary of what you learned this week. Keep it in your repo's `notes/week1.md`.

---

## Week 2 — Deep Learning Practice

### Day 8 (Monday) — MLPs for Real (2 hours)
- [ ] Watch Karpathy's "The spelled-out intro to language modeling: building makemore" (~1 hr)
- [ ] Implement a character-level bigram language model using an MLP
- [ ] Train on any text corpus (names dataset from Karpathy, or Tiny Shakespeare)

### Day 9 (Tuesday) — Training Loop Craft (2 hours)
- [ ] Set up W&B logging in your training loop
- [ ] Log: train loss, val loss, learning rate, gradient norm
- [ ] Create a training loop template you'll reuse throughout this plan
- [ ] Save it as `templates/train_loop.py`

### Day 10 (Wednesday) — Normalization (2 hours)
- [ ] Read: Ba et al., "Layer Normalization" (2016) — key sections
- [ ] Implement LayerNorm from scratch
- [ ] Implement BatchNorm from scratch
- [ ] Verify both match PyTorch's built-in on random tensors
- [ ] Write a short note: when does each fail? (hint: batch size, sequence length)

### Day 11 (Thursday) — Regularization & Failure Modes (2 hours)
- [ ] Train an MLP that overfits deliberately (tiny dataset, huge model)
- [ ] Add dropout, weight decay, early stopping
- [ ] Observe how each affects the overfitting
- [ ] Intentionally break things: bad initialization, too-high learning rate, watch the model diverge

### Day 12 (Friday) — Catch-up / Review (2 hours)
Buffer day. Use it to:
- [ ] Clean up your code from this week
- [ ] Push everything to GitHub with good README
- [ ] Review any concepts that felt shaky

### Day 13 (Saturday) — DEEP WORK: CIFAR-10 CNN (4–6 hours)
Goals: Run a non-trivial training job end-to-end. Optional but recommended.

- [ ] Build a simple CNN for CIFAR-10
- [ ] Train for enough epochs to hit 70%+ test accuracy
- [ ] Log everything to W&B
- [ ] Try different architectures (depth, width), report what worked
- [ ] Commit as `phase1/03_cifar10_cnn/`

### Day 14 (Sunday) — REST
- Optional: Read ahead — skim "Attention Is All You Need" for orientation.

---

# PHASE 2: Transformer from Scratch

## Week 3 — Attention & Self-Attention

### Day 15 (Monday) — The Paper (2 hours)
- [ ] **Read carefully: Vaswani et al., "Attention Is All You Need" (2017)**
- [ ] Focus on Sections 3 (Architecture) and 3.2 (Attention)
- [ ] Take notes: what problem is it solving? What's the key insight?
- [ ] Skim: "The Illustrated Transformer" again — it'll make more sense now

### Day 16 (Tuesday) — Karpathy's GPT video Part 1 (2 hours)
- [ ] Watch Karpathy's "Let's build GPT: from scratch, in code, spelled out" — first half
- [ ] Code along in your own repo, not by copying

### Day 17 (Wednesday) — Scaled Dot-Product Attention (2 hours)
- [ ] Implement scaled dot-product attention from scratch in PyTorch
- [ ] Function signature: `attention(Q, K, V, mask=None) -> output, attention_weights`
- [ ] Verify on tiny inputs; hand-compute expected output for a 2-token, 2-dim example
- [ ] Understand why the √d_k scaling exists (hint: keeps softmax entropy stable — your info theory background applies)

### Day 18 (Thursday) — Multi-Head Attention (2 hours)
- [ ] Extend your attention to multi-head attention
- [ ] Understand the reshape: `(B, T, C) -> (B, nh, T, hs)`
- [ ] Implement causal masking for autoregressive models
- [ ] Test with a toy input; visualize the attention matrix

### Day 19 (Friday) — Transformer Block (2 hours)
- [ ] Implement the full transformer block: attention + FFN + residual + layer norm
- [ ] Support both pre-norm and post-norm variants
- [ ] Wire together: embedding + N blocks + final projection to vocab

### Day 20 (Saturday) — DEEP WORK: Full GPT + Training (5–6 hours)
- [ ] Finish watching Karpathy's "Let's build GPT" video
- [ ] Complete your GPT implementation (character-level)
- [ ] Train on Tiny Shakespeare until you get coherent-looking output
- [ ] Save checkpoints, log to W&B
- [ ] Commit to repo as `phase2/04_gpt_from_scratch/`

### Day 21 (Sunday) — REST

---

## Week 4 — Transformer Mastery

### Day 22 (Monday) — GPT Lineage (2 hours)
- [ ] Skim: Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- [ ] Skim: Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
- [ ] Read more carefully: Brown et al., "Language Models are Few-Shot Learners" (GPT-3) — focus on Section 2
- [ ] Understand: what changed from GPT-1 → GPT-3? It's mostly scale.

### Day 23 (Tuesday) — Positional Encodings (2 hours)
- [ ] Understand sinusoidal positional encoding deeply (from original transformer)
- [ ] Implement it from scratch
- [ ] Add it to your transformer

### Day 24 (Wednesday) — RoPE (2 hours)
- [ ] Read: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- [ ] Implement RoPE from scratch
- [ ] Understand *why* RoPE works: it encodes relative position through rotation in complex plane
- [ ] Swap sinusoidal for RoPE in your transformer; re-train, compare

### Day 25 (Thursday) — ALiBi + Comparison (2 hours)
- [ ] Read: Press et al., "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022)
- [ ] Implement ALiBi
- [ ] Benchmark: sinusoidal vs. learned vs. RoPE vs. ALiBi on Tiny Shakespeare
- [ ] Write up findings

### Day 26 (Friday) — KV Caching (2 hours)
- [ ] Understand why KV caching matters for inference
- [ ] Implement a KV cache in your transformer's generate function
- [ ] Benchmark generation speed with and without cache
- [ ] You should see 10x+ speedup on longer sequences

### Day 27 (Saturday) — DEEP WORK: Compare to nanoGPT (4–6 hours)
- [ ] Clone Karpathy's `nanoGPT` repo
- [ ] Line-by-line compare your implementation to his
- [ ] Find every place your code differs — understand why
- [ ] Fix any bugs or inefficiencies in your code
- [ ] Commit improvements to your repo

### Day 28 (Sunday) — REST
- Optional: Skim the Llama 2 paper for next week.

---

# PHASE 3: Modern LLM Architecture

## Week 5 — Llama-style Architecture

### Day 29 (Monday) — RMSNorm & SwiGLU (2 hours)
- [ ] Read: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- [ ] Implement RMSNorm from scratch (it's simpler than LayerNorm)
- [ ] Read: Shazeer, "GLU Variants Improve Transformer" (2020)
- [ ] Implement SwiGLU activation
- [ ] Understand why modern LLMs use these instead of LayerNorm + ReLU

### Day 30 (Tuesday) — Grouped Query Attention (2 hours)
- [ ] Read: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- [ ] Implement GQA in your transformer
- [ ] Understand tradeoff: GQA reduces KV cache memory with small quality cost
- [ ] Measure memory usage difference on a long sequence

### Day 31 (Wednesday) — Llama 2 Paper (2 hours)
- [ ] Read: Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models"
- [ ] Focus on: architecture choices (Section 2), training data (Section 2.1)
- [ ] Note every difference from the original transformer

### Day 32 (Thursday) — Llama 3 Paper (2 hours)
- [ ] Read: "The Llama 3 Herd of Models" (Meta, 2024) — the technical report
- [ ] This is long — focus on architecture and training decisions
- [ ] Note what's different from Llama 2

### Day 33 (Friday) — Integration (2 hours)
- [ ] Upgrade your Phase 2 transformer to use:
  - RMSNorm instead of LayerNorm
  - RoPE instead of absolute positional encoding
  - SwiGLU instead of ReLU
  - GQA instead of vanilla MHA
- [ ] Call it `phase3/05_llama_style/`

### Day 34 (Saturday) — DEEP WORK: Train your Llama-style model (5–6 hours)
- [ ] Train your Llama-style transformer on Tiny Shakespeare
- [ ] Compare loss curves: old transformer vs. Llama-style
- [ ] Try different model sizes (4M, 10M, 25M params)
- [ ] Write up your findings

### Day 35 (Sunday) — REST

---

## Week 6 — Mixture of Experts & Flash Attention

### Day 36 (Monday) — MoE Foundations (2 hours)
- [ ] Read: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- [ ] Read: Fedus et al., "Switch Transformer" (2021) — focus on routing
- [ ] Understand: what problem do experts solve? (hint: scaling parameters without scaling compute)

### Day 37 (Tuesday) — Mixtral & Modern MoE (2 hours)
- [ ] Read: Jiang et al., "Mixtral of Experts" (2024)
- [ ] Read: DeepSeek-V3 technical report — focus on MoE design (it's sophisticated)
- [ ] Note: shared experts, fine-grained experts, auxiliary losses

### Day 38 (Wednesday) — Implement MoE (2 hours)
- [ ] Implement a simple MoE feed-forward layer with 4 experts, top-2 routing
- [ ] Add load balancing loss
- [ ] Replace FFN with MoE in one layer of your transformer
- [ ] Train briefly, verify it works

### Day 39 (Thursday) — FlashAttention Paper (2 hours)
- [ ] Read: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- [ ] Understand the key insight: recomputation saves memory bandwidth
- [ ] Connect to your background: this is about communication cost, just like in distributed computing

### Day 40 (Friday) — FlashAttention in Practice (2 hours)
- [ ] Note: `flash-attn` library doesn't work natively on Mac/MPS. Use cloud GPU or skip implementation, just understand it.
- [ ] On Mac: study the algorithm carefully and write pseudocode
- [ ] Benchmark naive attention on different sequence lengths
- [ ] Plot memory usage and speed

### Day 41 (Saturday) — DEEP WORK: DeepSeek-V3 Deep Dive (5–6 hours)
- [ ] Read the full DeepSeek-V3 technical report
- [ ] One of the most interesting recent architecture papers
- [ ] Write detailed notes: Multi-head Latent Attention (MLA), auxiliary-loss-free load balancing, FP8 training
- [ ] Try to reimplement MLA as an exercise

### Day 42 (Sunday) — REST

---

## Week 7 — Alternatives & Long Context

### Day 43 (Monday) — State Space Models (2 hours)
- [ ] Read: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- [ ] Focus on Sections 1-2, skim the math details
- [ ] Understand: why do people think SSMs might challenge transformers?

### Day 44 (Tuesday) — SSM Implementation (2 hours)
- [ ] Follow a tutorial on implementing a simple SSM
- [ ] Understand: convolutional view vs. recurrent view
- [ ] Implement selective scan (the core Mamba operation) in a toy way

### Day 45 (Wednesday) — Alternative Architectures (2 hours)
- [ ] Skim: RWKV papers
- [ ] Skim: Linear attention papers (Performer, Linformer)
- [ ] Read one recent "attention alternative" paper of your choice
- [ ] Write a paragraph: why have transformers won so far despite these alternatives?

### Day 46 (Thursday) — Long Context Techniques (2 hours)
- [ ] Read: Chen et al., "Extending Context Window via Positional Interpolation" (2023)
- [ ] Read: Peng et al., "YaRN" (2023)
- [ ] Understand: how do models handle longer sequences at inference than they were trained on?
- [ ] Skim: Liu et al., "Ring Attention" (2023)

### Day 47 (Friday) — Gemini 1.5 (2 hours)
- [ ] Read: "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context"
- [ ] Focus on long context mechanisms
- [ ] Note the evaluation methodology (needle in haystack)

### Day 48 (Saturday) — DEEP WORK: Catch-up + Consolidation (4–6 hours)
Buffer day. Choose one:
- [ ] Option A: Go deeper on a topic that confused you this phase
- [ ] Option B: Train your Llama-style transformer at slightly larger scale (consider renting cloud GPU)
- [ ] Option C: Write a detailed blog-style summary of Phase 3 — this cements knowledge
- [ ] Option D: Read any frontier paper you've been curious about

### Day 49 (Sunday) — REST

---

# PHASE 4: Pretraining & Scaling Laws

## Week 8 — Scaling Laws

### Day 50 (Monday) — Original Scaling Laws (2 hours)
- [ ] Read: Kaplan et al., "Scaling Laws for Neural Language Models" (2020) — **important**
- [ ] Focus on Sections 1-3, the main empirical results
- [ ] Connect to your info theory background: these are empirical compression-prediction tradeoffs

### Day 51 (Tuesday) — Chinchilla (2 hours)
- [ ] Read: Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- [ ] Understand why Chinchilla "corrects" Kaplan
- [ ] Key insight: for a given compute budget, there's an optimal (model size, training tokens) pair
- [ ] Compute: what would Chinchilla-optimal training look like for a 1B model? 7B? 70B?

### Day 52 (Wednesday) — Empirical Scaling (2 hours)
- [ ] Design a small-scale scaling experiment: train transformers of 1M, 4M, 16M, 64M params on the same dataset for the same compute
- [ ] Set up the experiment; start training
- [ ] Plot loss vs. parameters

### Day 53 (Thursday) — Data Curation (2 hours)
- [ ] Read: Penedo et al., "The FineWeb datasets" (2024)
- [ ] Skim: RefinedWeb paper
- [ ] Understand: deduplication, quality filtering, heuristic rules
- [ ] Why does data quality affect scaling laws? (Think information-theoretically)

### Day 54 (Friday) — Distributed Training Concepts (2 hours)
- [ ] Read: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)
- [ ] Understand: data parallelism, tensor parallelism, pipeline parallelism, FSDP
- [ ] You don't need to implement these, but understand the concepts

### Day 55 (Saturday) — DEEP WORK: Finish Scaling Experiment + Write-up (5–6 hours)
- [ ] Finish training your scaling experiment models
- [ ] Plot loss vs. compute (FLOPs) and loss vs. parameters
- [ ] Fit a power law to your data
- [ ] Write up: do your results match Kaplan/Chinchilla predictions? Why or why not?
- [ ] Commit as `phase4/06_scaling_experiment/`

### Day 56 (Sunday) — REST

---

## Week 9 — Real Pretraining

### Day 57 (Monday) — Mixed Precision & Grad Accumulation (2 hours)
- [ ] Implement bf16 mixed precision training in your loop
- [ ] Note: bf16 is limited on MPS, use fp16 instead on Mac
- [ ] Implement gradient accumulation (simulates larger batch sizes)
- [ ] Benchmark: fp32 vs. fp16 memory and speed
- [ ] **Remember Day 2 lesson:** fp16 has precision issues. Cast to fp32 for sensitive calculations.

### Day 58 (Tuesday) — Tokenization (2 hours)
- [ ] Read about BPE (Byte-Pair Encoding) algorithm
- [ ] Implement BPE tokenizer from scratch (or follow a tutorial)
- [ ] Compare to sentencepiece / tiktoken on the same text
- [ ] Understand: why tokenization matters for model behavior

### Day 59 (Wednesday) — Build Your Own Dataset (2 hours)
- [ ] Pick a domain: arXiv cs.IT abstracts, Wikipedia math, Chinese news, etc.
- [ ] Collect ~100MB–1GB of text
- [ ] Deduplicate, clean, tokenize
- [ ] Save as a preprocessed dataset

### Day 60 (Thursday) — Pretrain a Small Model (2 hours)
- [ ] Train your Llama-style transformer on your custom dataset
- [ ] Aim for 1–6 hours of training
- [ ] Monitor loss, gradient norm, learning rate schedule

### Day 61 (Friday) — Analyze Your Model (2 hours)
- [ ] Generate text from your trained model
- [ ] Analyze: what did it learn? What didn't it?
- [ ] Think about: what would a larger training budget add?

### Day 62 (Saturday) — DEEP WORK: Compression Connection (5–6 hours)
**Goals: Use your background. This is the day you'll have been waiting for.**

- [ ] Read: Delétang et al., "Language Modeling Is Compression" (DeepMind, 2023)
- [ ] **This will be deeply satisfying given your background.** It formally proves what you intuited on Day 2.
- [ ] Implement the paper's core idea: use your trained model as a compressor
- [ ] Measure compression ratio vs. gzip on some text
- [ ] Write up: what does this tell us about LLMs from an information theory perspective?

### Day 63 (Sunday) — REST

---

# PHASE 5: Post-Training & Alignment

## Week 10 — Supervised Fine-Tuning

### Day 64 (Monday) — InstructGPT Paper (2 hours)
- [ ] Read: Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, 2022) — **critical paper**
- [ ] Focus on the three-stage training: SFT → reward model → RLHF
- [ ] This is the conceptual foundation of every modern chatbot

### Day 65 (Tuesday) — Instruction Tuning (2 hours)
- [ ] Skim: Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN)
- [ ] Understand: instruction tuning as generalization technique
- [ ] Look at example instruction datasets: Alpaca, Dolly, UltraChat
- [ ] Download UltraChat or a similar dataset

### Day 66 (Wednesday) — SFT in Practice (2 hours)
- [ ] Use Hugging Face `trl` library to SFT a small pretrained model (Qwen 2.5 0.5B)
- [ ] Train on 1000-10000 instruction examples
- [ ] Compare base model vs. SFT model outputs on sample prompts

### Day 67 (Thursday) — LoRA and PEFT (2 hours)
- [ ] Read: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- [ ] Understand: why is LoRA so much cheaper than full fine-tuning?
- [ ] Connect to your math background: it's a low-rank approximation of the weight update
- [ ] Redo yesterday's SFT with LoRA using `peft` library

### Day 68 (Friday) — Reward Models (2 hours)
- [ ] Read InstructGPT Section 3.2 again on reward model training
- [ ] Understand: a reward model is a classifier that scores response quality
- [ ] Implement a simple reward model head on top of a pretrained transformer
- [ ] Train it on preference pairs from a dataset like HH-RLHF

### Day 69 (Saturday) — DEEP WORK: Full SFT Pipeline (5–6 hours)
- [ ] End-to-end: take Qwen 2.5 0.5B, SFT it on a chosen dataset with LoRA
- [ ] Evaluate qualitatively (your judgment) and quantitatively (perplexity on held-out)
- [ ] Write up your findings
- [ ] Commit as `phase5/07_sft/`

### Day 70 (Sunday) — REST

---

## Week 11 — DPO Deep Dive

### Day 71 (Monday) — DPO Paper (2 hours)
- [ ] Read: Rafailov et al., "Direct Preference Optimization" (DPO, 2023) — **important**
- [ ] Focus on the derivation (Section 4)
- [ ] Key insight: you can optimize the RLHF objective in closed form without actually doing RL
- [ ] Note the KL-divergence constraint — this is information-theoretic

### Day 72 (Tuesday) — DPO Math (2 hours)
- [ ] Rederive the DPO loss yourself on paper
- [ ] Understand why the reference model matters
- [ ] Understand the Bradley-Terry preference model assumption

### Day 73 (Wednesday) — Implement DPO from Scratch (2 hours)
- [ ] Implement the DPO loss function yourself (not using `trl`)
- [ ] It's about 20-30 lines of code
- [ ] Test on toy preference pairs

### Day 74 (Thursday) — DPO Training (2 hours)
- [ ] Use your DPO loss to fine-tune a model on preference data
- [ ] Or use `trl`'s DPOTrainer as comparison
- [ ] Monitor: reward margins, KL divergence from reference

### Day 75 (Friday) — DPO Alternatives (2 hours)
- [ ] Skim: IPO (Azar et al., 2023), KTO (Ethayarajh et al., 2024), SimPO (Meng et al., 2024)
- [ ] Understand: what do these claim to fix about DPO?
- [ ] Choose one and implement it

### Day 76 (Saturday) — DEEP WORK: Compare Preference Methods (5–6 hours)
- [ ] Run DPO, IPO or KTO, and SimPO on the same preference dataset
- [ ] Same base model, same data, same hyperparameters otherwise
- [ ] Compare: final reward margins, KL divergence, human-eval quality
- [ ] Write up your findings

### Day 77 (Sunday) — REST

---

## Week 12 — Constitutional AI & Reasoning Models

### Day 78 (Monday) — Constitutional AI (2 hours)
- [ ] Read: Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
- [ ] This is foundational to how Claude works
- [ ] Understand: Critique → Revise loops, RLAIF vs RLHF

### Day 79 (Tuesday) — RLAIF (2 hours)
- [ ] Read: Lee et al., "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (2023)
- [ ] Understand: when can AI feedback replace human feedback? When can't it?

### Day 80 (Wednesday) — DeepSeek R1 (2 hours)
- [ ] Read: DeepSeek-R1 technical report (2025)
- [ ] **One of the most important recent papers**
- [ ] Key insight: pure RL on verifiable rewards produces reasoning
- [ ] Understand: R1-Zero vs R1, the role of cold-start SFT

### Day 81 (Thursday) — o1 and Test-Time Compute (2 hours)
- [ ] Read OpenAI's o1 blog posts and system card
- [ ] Read: Snell et al., "Scaling LLM Test-Time Compute Optimally" (2024)
- [ ] Understand: test-time compute as a new scaling axis

### Day 82 (Friday) — Reasoning Experiments (2 hours)
- [ ] Experiment: take any model, try different chain-of-thought prompting strategies
- [ ] Try: "Let's think step by step", self-consistency (multiple samples), tree-of-thoughts
- [ ] Test on math word problems (GSM8K style)
- [ ] Measure accuracy difference

### Day 83 (Saturday) — DEEP WORK: Phase 5 Write-up (4–6 hours)
- [ ] Write a comprehensive summary of everything you've done in post-training
- [ ] Create a diagram of the full post-training pipeline
- [ ] Identify: where are the open research problems?
- [ ] This write-up will be useful for postdoc research

### Day 84 (Sunday) — REST

---

# PHASE 6: Inference & Deployment

## Week 13 — Inference Optimization

### Day 85 (Monday) — Profile Your Inference (2 hours)
- [ ] Profile your Phase 5 fine-tuned model's inference
- [ ] Use PyTorch profiler
- [ ] Identify bottlenecks: is it compute-bound or memory-bound?
- [ ] For LLMs at inference, it's almost always memory bandwidth

### Day 86 (Tuesday) — KV Cache Deep Dive (2 hours)
- [ ] Your KV cache from Day 26: make it more efficient
- [ ] Understand: paged attention (vLLM's innovation)
- [ ] Read: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (2023)

### Day 87 (Wednesday) — Quantization Theory (2 hours)
- [ ] Read: Dettmers et al., "GPT3.int8(): 8-bit Matrix Multiplication" (2022)
- [ ] Read: Frantar et al., "GPTQ" (2023) — focuses on post-training quantization
- [ ] Your coding theory background: recognize these as lossy source coding problems

### Day 88 (Thursday) — Quantization Practice (2 hours)
- [ ] Note: `bitsandbytes` doesn't work on Mac MPS. Use `mlx` framework or skip implementation.
- [ ] Alternative: study quantization formats on paper
- [ ] If you have cloud GPU: use `bitsandbytes` to load in 8-bit and 4-bit
- [ ] Benchmark: quality (perplexity) vs. memory savings

### Day 89 (Friday) — Speculative Decoding (2 hours)
- [ ] Read: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- [ ] Understand the math: why does it preserve the distribution exactly?
- [ ] This connects to rejection sampling — a concept from your probabilistic toolkit

### Day 90 (Saturday) — DEEP WORK: Implement Speculative Decoding (5–6 hours)
- [ ] Take a small model (draft) and a larger model (target)
- [ ] Implement speculative decoding manually
- [ ] Verify: output distribution matches target model alone
- [ ] Benchmark speedup
- [ ] Commit as `phase6/08_speculative_decoding/`

### Day 91 (Sunday) — REST

---

## Week 14 — Production Deployment

### Day 92 (Monday) — vLLM (2 hours)
- [ ] Note: vLLM doesn't run natively on Mac MPS. Use Linux machine or cloud.
- [ ] On Mac: use `mlx-lm` or `llama.cpp` as alternatives
- [ ] Install and use vLLM (or alternative) to serve your model
- [ ] Benchmark throughput: requests/second at various concurrencies

### Day 93 (Tuesday) — Continuous Batching (2 hours)
- [ ] Understand: what is continuous batching?
- [ ] Why is it crucial for serving multiple users?
- [ ] Read vLLM's blog post on this

### Day 94 (Wednesday) — Alternative Serving (2 hours)
- [ ] Try Hugging Face `text-generation-inference`
- [ ] Try `llama.cpp` (CPU inference + quantization, works great on Mac)
- [ ] Read through `llama.cpp` source briefly — it's beautifully minimal

### Day 95 (Thursday) — Production Considerations (2 hours)
- [ ] Learn about: streaming responses, timeouts, retries
- [ ] Set up a simple API endpoint using FastAPI
- [ ] Add proper logging and error handling

### Day 96 (Friday) — Buffer / Catch-up (2 hours)
- [ ] Clean up your code from this phase
- [ ] Update your GitHub README
- [ ] Reflect: what's been hardest? Easiest?

### Day 97 (Saturday) — DEEP WORK: Deploy End-to-End (5–6 hours)
- [ ] Take your fine-tuned model from Phase 5
- [ ] Quantize it (using `mlx` or `llama.cpp` on Mac)
- [ ] Serve it via FastAPI endpoint
- [ ] Benchmark full-system throughput and latency
- [ ] Write it up as a mini-project

### Day 98 (Sunday) — REST

---

# PHASE 7: Frontier Topics + Capstone

## Week 15 — Mechanistic Interpretability (Recommended for Your Background)

*Note: This week is optional/customizable. If another frontier topic interests you more, substitute accordingly. But mechanistic interpretability is recommended given your theoretical CS + info theory background — this is where you have real research edge.*

### Day 99 (Monday) — Foundations (2 hours)
- [ ] Read: Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021)
- [ ] This is a mathematical paper — it'll feel natural to you
- [ ] Focus on: QK and OV circuits, attention head composition

### Day 100 (Tuesday) — Induction Heads (2 hours)
- [ ] Read: Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022)
- [ ] Understand: how attention heads can implement algorithmic behaviors
- [ ] This is the closest ML research gets to your theoretical CS background

### Day 101 (Wednesday) — TransformerLens (2 hours)
- [ ] Install Neel Nanda's `TransformerLens` library
- [ ] Complete the introductory tutorial
- [ ] Start exploring a pretrained model's internals

### Day 102 (Thursday) — Sparse Autoencoders (2 hours)
- [ ] Read: Bricken et al., "Towards Monosemanticity" (Anthropic, 2023)
- [ ] Read: Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024)
- [ ] Key insight: SAEs decompose activations into interpretable features
- [ ] This connects to: compressed sensing, dictionary learning — familiar territory for you

### Day 103 (Friday) — SAE Hands-On (2 hours)
- [ ] Try running a pretrained SAE (some are available via `sae_lens`)
- [ ] Interpret some features on sample text
- [ ] Think: what's the information-theoretic framing of SAEs?

### Day 104 (Saturday) — DEEP WORK: Interp Project (5–6 hours)
- [ ] Pick a small phenomenon to analyze in a model of your choice
- [ ] Example: "How does this model handle comparison words (bigger, smaller)?"
- [ ] Use TransformerLens to investigate
- [ ] Write up findings — this could literally become research

### Day 105 (Sunday) — REST

---

## Week 16 — Capstone Project

This final week is for starting a capstone project that you'll continue into your postdoc. Pick ONE:

### Capstone Option A: Information-Theoretic Analysis of LLMs (Recommended)
Apply your info theory background to LLMs:
- Analyze attention patterns as channels
- Compute mutual information between layers
- Empirically study scaling laws through a rate-distortion lens

### Capstone Option B: Coding Theory Meets LLMs
- Apply error-correcting codes to LLM robustness
- Information-theoretic analysis of speculative decoding
- Codebook design for extreme LLM quantization

### Capstone Option C: Full-Stack Small LLM
- Train a 100M-300M parameter Llama-style model from scratch
- SFT + DPO
- Quantize and deploy
- Open-source release

### Capstone Option D: Reasoning Model Replication
- Replicate DeepSeek R1-Zero's approach on a small model
- Use verifiable rewards (math problems)
- Study emergence of reasoning

### Day 106 (Monday) — Capstone Scope (2 hours)
- [ ] Write a 1-page proposal for your chosen capstone
- [ ] Goals, methods, timeline, expected outcome
- [ ] Identify key papers and resources

### Day 107 (Tuesday) — Literature Review (2 hours)
- [ ] Read 3-5 most relevant papers for your capstone
- [ ] Take detailed notes

### Day 108 (Wednesday) — Implementation Start (2 hours)
- [ ] Begin coding your capstone
- [ ] Set up repo, structure, initial experiments

### Day 109 (Thursday) — Iterate (2 hours)
- [ ] Continue implementation
- [ ] Run initial experiments

### Day 110 (Friday) — Debug and Document (2 hours)
- [ ] Debug issues
- [ ] Document progress

### Day 111 (Saturday) — DEEP WORK: Capstone Sprint (5–6 hours)
- [ ] Full day on capstone
- [ ] Aim for a complete first version of something

### Day 112 (Sunday) — Celebrate + Plan Forward
- [ ] You've done 16 weeks of serious work. That's real.
- [ ] Write up what you've learned — complete with your portfolio
- [ ] Plan: how will you continue this work into your postdoc?
- [ ] Identify: what papers could come out of your capstone?

---

## Post-Plan: What's Next

After 16 weeks, you should be able to:
- Read any LLM paper and understand the architecture and training
- Implement a modern transformer from scratch
- Fine-tune models for specific tasks
- Deploy models in production
- Identify open research problems at the LLM frontier
- Bring your theoretical CS background to bear on ML research

**Next steps for your postdoc:**
1. Pick one to two research directions that genuinely excite you
2. Connect with researchers in those areas (local and internationally)
3. Aim for at least one first-author paper during your postdoc combining your ML implementation skills with your theoretical background
4. Keep your implementation skills sharp — contribute to open source, maintain your GitHub

---

## Lessons Learned (Updating as We Go)

### From Day 2:
1. **Numerical precision is critical.** float16 can corrupt entropy/log-probability calculations. Always cast to float32 for sensitive math, or use `log_softmax` for stability.
2. **Trust theoretical intuition when numbers feel wrong.** I noticed entropy ≈ max entropy was suspicious — this caught a real bug.
3. **AI-generated code needs critical review.** Even code that runs cleanly can produce subtly wrong results. The bug on Day 2 was a perfect example.
4. **Source coding theorem applies directly to LLMs.** Better LLM = better compressor. Cross-entropy loss = expected code length. This will recur throughout the plan.

### From Day 1:
1. **macOS uses MPS (Metal) instead of CUDA.** Some operations need CPU fallback. Some libraries (`flash-attn`, `bitsandbytes`, `vllm`) don't work natively.
2. **Setup is the foundation.** Getting environment right on Day 1 prevents weeks of pain.

---

## Mac-Specific Notes

Since you're on Apple Silicon (M-series Mac), here are things to know:

**Works great on Mac:**
- Standard PyTorch operations
- Hugging Face transformers (loading and inference)
- Training small/medium models
- All Phase 1, 2, and 3 work

**Limited or doesn't work on Mac:**
- `flash-attn` library (use naive attention or rent cloud GPU)
- `bitsandbytes` quantization (use `mlx-lm` or `llama.cpp` instead)
- `vllm` server (use `llama.cpp` or rent Linux GPU)
- bf16 (use fp16 or fp32 instead)
- Distributed training across multiple machines

**Mac alternatives:**
- `mlx` framework (Apple's native ML framework, excellent for Mac)
- `mlx-lm` for inference and fine-tuning
- `llama.cpp` for highly optimized inference
- `ollama` for easy local model serving

**When to rent cloud GPU:**
- Phase 4 (scaling experiments) — optional but useful
- Phase 5 (DPO training on larger models) — optional
- Phase 6 (production inference benchmarks) — useful

Recommended cloud GPU: Lambda Labs, RunPod, or Modal. Budget ~$50-100 total for the full 16 weeks.

---

## Mental Health & Sustainability

Throughout this plan:
- **Check in with yourself weekly.** Are you still enjoying this?
- **Social support matters.** Talk to other PhDs, friends, family.
- **Exercise and sleep are non-negotiable.** Your brain literally won't learn well without them.
- **Celebrate small wins.** Day 2's bug catch was a real win — those add up.
- **Perfectionism will kill this plan.** Aim for "done and understood" not "flawless."
- **If you miss a week, don't quit.** Just pick up where you left off.

---

## Tracking Template

For each day, in `notes/dayN.md`:

```markdown
# Day N — Date

## Completed
- [x] Task 1
- [x] Task 2

## Three insights from today
1. 
2. 
3. 

## What still confuses me
- 

## Connection to my background
- 

## Issues encountered (if any)
- 
```

---

## Ready for Day 3

Day 1: ✅ Done
Day 2: ✅ Done

Tomorrow (Day 3) is **PyTorch Tensors & Autograd** — the foundation skill for everything else. Let's go.
