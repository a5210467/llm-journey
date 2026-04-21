# Day 2 — Karpathy's "Intro to LLMs" Notes

## Part 1: LLM Basics (0:00 – 20:00)

- What is an LLM? (your own words):
An LLM is essentially two files: a huge parameter file (billions of numbers)
  and a small program that uses those numbers. Its function is to predict the
  next token given input text. It takes text → tokens → runs through a neural
  network with attention → outputs a probability distribution over the vocabulary
  → samples the next token. Repeat to generate.
- Key insight:
  Simple task (predict next token), but because doing it well REQUIRES
  understanding language, facts, reasoning, etc., the model learns all these
  as side effects. Deep capability emerges from a simple objective.

## Part 2: Training (20:00 – 40:00)

- Two phases of training:
  1. Pretraining 
  2. Fine-tuning 

- Why is pretraining expensive?
- Massive data (trillions of tokens)
  - Massive models (billions of parameters)
  - Massive compute (thousands of GPUs running for weeks/months)
  - Electricity and hardware costs are enormous
  - NOT because of trusted Q&A — that's Phase 2 and it's cheap


## Part 3: Fine-tuning and Assistants (40:00 – 60:00)

- How do we turn a base model into an assistant?
Three-step process:
  1. Supervised Fine-Tuning (SFT) — train on ~10,000 high-quality human-written
     Q&A pairs. Model learns to respond like an assistant.
  2. Reward Model — show humans pairs of responses, train a model to predict
     which one humans prefer.
  3. RLHF (Reinforcement Learning from Human Feedback) — use the reward model
     to further tune the LLM toward preferred responses.



## Part 4: The Future (60:00 – end)

- LLM OS concept:
LLMs are becoming like operating systems. Think of LLM as a CPU with:
    - RAM = context window
    - Disk = external databases (RAG)
    - Peripherals = tools (calculator, browser, code interpreter)
    - Other "computers" = other LLMs
  LLMs aren't just chatbots anymore — they're the central computation layer
  for agentic systems that use tools and interact with the world.


- Scaling laws:
If you increase (1) parameters, (2) training data, (3) compute, the model
  predictably gets better. The relationship follows a power law — smooth and
  predictable. This is why companies keep making bigger models — it's not
  magic, it's extrapolation.
  Connection to my background: this is empirically a rate-distortion-like
  curve. Loss = distortion, compute/params = rate.

- Jailbreaks and security:

  LLMs are not robust. Active attacks include:
    - Jailbreaks (tricking the model into breaking safety rules)
    - Prompt injection (hidden instructions in data the model reads)
    - Data poisoning (bad content in training data)
    - Adversarial suffixes (strange strings that break safety)
  This is an unsolved security problem.


## Three things I didn't know before:
1. An LLM is literally just "predict next token" — I thought it was more complex
2. Pretraining vs fine-tuning — different phases with vastly different costs
3. The "LLM OS" framing — LLMs as compute units with tools, not just chatbots


## Three things I want to learn more about:
1.How attention actually works mathematically (Week 3 will cover this)
2. RLHF in detail — how does the reward model actually work?
3. Scaling laws — what's the precise form? (Week 8 will cover this)

