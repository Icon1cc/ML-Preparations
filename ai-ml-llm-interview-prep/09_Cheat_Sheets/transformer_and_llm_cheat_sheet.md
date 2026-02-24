# Transformer & LLM Cheat Sheet

> Dense reference for architecture questions, model selection, and LLM system design interviews.

---

## 1. Transformer Architecture — Core Formulas

### Attention Mechanism

```
Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

Where:
  Q  = query matrix  [seq_len × d_k]
  K  = key matrix    [seq_len × d_k]
  V  = value matrix  [seq_len × d_v]
  d_k = key dimension = d_model / n_heads

The sqrt(d_k) scaling prevents dot products from growing large in high dimensions,
which pushes softmax into regions with tiny gradients.
```

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

Projection shapes per head:
  W_Q_i, W_K_i : [d_model × d_k]   where d_k = d_model / n_heads
  W_V_i         : [d_model × d_v]   where d_v = d_model / n_heads
  W_O           : [h * d_v × d_model] = [d_model × d_model]
```

### Standard Transformer Dimensions (Original Paper)

| Hyperparameter | Symbol | BERT-base | BERT-large | GPT-2 Small | GPT-2 XL | Typical LLM |
|----------------|--------|-----------|------------|-------------|----------|-------------|
| Hidden dim | d_model | 768 | 1024 | 768 | 1600 | 4096 |
| Attention heads | n_heads | 12 | 16 | 12 | 25 | 32 |
| Head dim | d_k = d_model/h | 64 | 64 | 64 | 64 | 128 |
| Feed-forward dim | d_ff | 3072 | 4096 | 3072 | 6400 | 11008 |
| Layers | n_layers | 12 | 24 | 12 | 48 | 32 |
| Vocab size | V | 30522 | 30522 | 50257 | 50257 | 32000 |

**Key ratio to remember**: d_ff = 4 × d_model (standard), though LLaMA uses ~2.67× with SwiGLU.

---

## 2. Parameter Count Formulas

### Per-Layer Parameter Count

```
Attention block:
  W_Q + W_K + W_V: 3 × (d_model × d_model) = 3 × d_model²
  W_O:              d_model × d_model = d_model²
  Biases (if any):  4 × d_model
  Total attention: ≈ 4 × d_model²

Feed-forward block (standard):
  W_1: d_model × d_ff
  W_2: d_ff × d_model
  Total FFN: 2 × d_model × d_ff = 2 × d_model × 4 × d_model = 8 × d_model²

Layer Norm (×2 per layer):
  Weight + bias: 2 × d_model (negligible)

Total per transformer layer:
  ≈ 4 × d_model² + 8 × d_model² = 12 × d_model²
```

### Full Model Parameter Count

```
Embedding table:         V × d_model
Position embeddings:     max_seq_len × d_model  (learned)
Transformer layers:      n_layers × 12 × d_model²
LM head (decoder):       d_model × V  (often tied to embedding)

Total ≈ V × d_model + n_layers × 12 × d_model²

Example: BERT-base
  30522 × 768 + 12 × 12 × 768² = 23.4M + 12 × 12 × 590K = 23.4M + 84.9M ≈ 108M ✓
```

### Known Model Parameter Counts

| Model | Architecture | Params | d_model | n_layers | n_heads | d_ff | Vocab |
|-------|-------------|--------|---------|----------|---------|------|-------|
| BERT-base | Encoder | 110M | 768 | 12 | 12 | 3072 | 30K |
| BERT-large | Encoder | 340M | 1024 | 24 | 16 | 4096 | 30K |
| DistilBERT | Encoder | 66M | 768 | 6 | 12 | 3072 | 30K |
| RoBERTa-base | Encoder | 125M | 768 | 12 | 12 | 3072 | 50K |
| GPT-2 | Decoder | 117M | 768 | 12 | 12 | 3072 | 50K |
| GPT-2 XL | Decoder | 1.5B | 1600 | 48 | 25 | 6400 | 50K |
| GPT-3 | Decoder | 175B | 12288 | 96 | 96 | 49152 | 50K |
| T5-base | Enc-Dec | 220M | 768 | 12+12 | 12 | 3072 | 32K |
| T5-11B | Enc-Dec | 11B | 1024 | 24+24 | 128 | 65536 | 32K |
| LLaMA-7B | Decoder | 7B | 4096 | 32 | 32 | 11008 | 32K |
| LLaMA-13B | Decoder | 13B | 5120 | 40 | 40 | 13824 | 32K |
| LLaMA-70B | Decoder | 70B | 8192 | 80 | 64 | 28672 | 32K |
| LLaMA 3-8B | Decoder | 8B | 4096 | 32 | 32 | 14336 | 128K |
| LLaMA 3-70B | Decoder | 70B | 8192 | 80 | 64 | 28672 | 128K |
| Mistral 7B | Decoder | 7B | 4096 | 32 | 32 | 14336 | 32K |
| Mixtral 8×7B | MoE | ~47B active 8B | 4096 | 32 | 32 | — | 32K |
| Phi-3-mini | Decoder | 3.8B | 3072 | 32 | 32 | 8192 | 32K |

---

## 3. Memory During Training

### Memory Components

```
For a model with P parameters, training memory (FP32):

Parameters:          P × 4 bytes
Gradients:           P × 4 bytes
Optimizer (Adam):    P × 8 bytes  (m and v, each FP32)
Activations:         batch_size × seq_len × d_model × n_layers × ~4 bytes
                     (can be 10-50× parameters for large batches)

Total (FP32, Adam): ≈ P × 16 bytes + activations

Example: 7B model in FP32:
  7B × 16 = 112 GB just for weights+optimizer+gradients
  → Requires A100 80GB ×2 at minimum in FP32
```

### Mixed Precision Training (FP16/BF16)

```
Parameters (FP16):    P × 2 bytes
Gradients (FP16):     P × 2 bytes
Master weights (FP32): P × 4 bytes   ← copy kept for numerical stability
Optimizer (FP32):     P × 8 bytes

Total mixed precision: P × 16 bytes  ← same! Savings come from activations
Activation (FP16): ~half of FP32 activations

Why BF16 over FP16?
  FP16: 5 exponent bits, 10 mantissa bits → overflows with large gradients
  BF16: 8 exponent bits, 7 mantissa bits  → same range as FP32, fewer precision bits
  → BF16 more stable for training, FP16 better for inference latency on older GPUs
```

### Memory per Billion Parameters by Format

| Format | Bytes/param | GB per 1B params | GB for 7B | GB for 70B | Use For |
|--------|-------------|------------------|-----------|------------|---------|
| FP32 | 4 | 4.0 GB | 28 GB | 280 GB | Training baseline |
| FP16 | 2 | 2.0 GB | 14 GB | 140 GB | Mixed precision training, inference |
| BF16 | 2 | 2.0 GB | 14 GB | 140 GB | Preferred training format |
| INT8 | 1 | 1.0 GB | 7 GB | 70 GB | Post-training quantization (inference) |
| INT4 | 0.5 | 0.5 GB | 3.5 GB | 35 GB | QLoRA training, edge inference |
| GGUF Q4_K_M | ~0.45 | ~0.45 GB | 4 GB | 40 GB | llama.cpp CPU inference |
| GGUF Q8_0 | ~0.9 | ~0.9 GB | 6 GB | 56 GB | llama.cpp higher quality |

**Rule of thumb for inference**: 2 bytes per param in BF16. Add 20% for KV cache.

---

## 4. Positional Encoding Types

| Type | Model | Formula / Method | Max Context | Extrapolates? | Notes |
|------|-------|------------------|-------------|----------------|-------|
| Sinusoidal (absolute) | Original Transformer, BERT (optional) | PE(pos,2i) = sin(pos/10000^(2i/d)) | Fixed at training | No | Deterministic, no learned params |
| Learned absolute | BERT, GPT-2 | Embedding table: E[pos] | Fixed at training | No | Simple, position-specific |
| Relative (Shaw et al.) | Music Transformer | Adds relative position bias to attention | No hard limit | Yes (with caution) | O(n²) extra terms |
| Relative (T5 bias) | T5, FLAN | Learned scalar bias per relative position bucket | No hard limit | Limited | Shared across heads |
| RoPE (Rotary Position) | LLaMA, GPT-NeoX, PaLM, Mistral | Rotates Q,K vectors by angle proportional to position | Theoretically unlimited | Yes (with scaling tricks) | No additive bias, multiplicative rotation |
| ALiBi | BLOOM, MPT | Subtracts linear penalty m×distance from attention logits | No hard limit | Yes | Zero learned params, strong extrapolation |
| xPos | Extrapolatable | Extension of RoPE | Unlimited | Better than RoPE | Adjusts decay by frequency |

### RoPE Math (Critical for LLaMA interviews)

```
For position m and dimension 2i, 2i+1:
  Q'_{m,2i}   =  q_{2i} cos(m·θ_i) - q_{2i+1} sin(m·θ_i)
  Q'_{m,2i+1} =  q_{2i} sin(m·θ_i) + q_{2i+1} cos(m·θ_i)
  θ_i = 10000^(-2i/d_model)

Key properties:
  1. Dot product Q'_m · K'_n depends only on (m-n) → relative position
  2. Decays with distance → longer sequences attended less
  3. No positional parameters added to model
  4. Can extend context via RoPE scaling (YaRN, NTK interpolation)
```

---

## 5. Tokenization Quick Reference

| Method | Algorithm | Models | Vocab Size | Handles OOV? | Multilingual |
|--------|-----------|--------|------------|--------------|--------------|
| BPE (Byte-Pair Encoding) | Merge most frequent byte pairs iteratively | GPT-2, GPT-3, GPT-4, LLaMA, Mistral | 32K-100K | Yes (byte fallback) | Yes (byte-level BPE) |
| WordPiece | Like BPE but maximizes language model likelihood | BERT, DistilBERT, ELECTRA | 28K-32K | Yes (##subwords) | Somewhat |
| SentencePiece (Unigram) | Probabilistic, language-model-based segmentation | T5, mT5, ALBERT, LLaMA | 32K-250K | Yes | Yes (excellent) |
| Character-level | Split into individual chars | Some old RNN models | ~100 | Yes | Yes |
| Byte-level | UTF-8 byte sequence | ByT5, some BERT variants | 256 | Yes (always) | Yes (perfect) |

### Tokenization Rules of Thumb

```
English text:   ~4 characters per token (GPT-4 tokenizer)
Code:           ~3-5 characters per token (more tokens for whitespace)
Non-Latin scripts: 1-2 characters per token (expensive for CJK)
Token to word ratio: ≈ 1.3 tokens per word in English

Cost estimation:
  1,000 tokens ≈ 750 words ≈ 3 pages of text

Vocabulary size tradeoff:
  Larger vocab → fewer tokens per sequence (cheaper inference)
              → more embedding parameters
              → sparser token statistics (rare tokens poorly trained)
  Smaller vocab → more tokens per sequence (more compute)
               → better coverage per token
```

---

## 6. Model Family Comparison Table

| Model | Architecture | Params | Context | Pretraining | Best For | Limitations |
|-------|-------------|--------|---------|-------------|----------|-------------|
| BERT-base | Encoder (MLM) | 110M | 512 | MLM + NSP | Classification, NER, QA (extractive) | No generation |
| RoBERTa | Encoder (MLM) | 125M | 512 | MLM (no NSP, more data) | Better classification than BERT | No generation |
| DistilBERT | Encoder (distilled) | 66M | 512 | Distilled from BERT | Fast inference, edge deployment | Slightly lower accuracy |
| DeBERTa-v3 | Encoder | 184M | 512 | MLM + SOP | Best encoder for benchmarks (SuperGLUE) | Slow inference |
| GPT-2 | Decoder (CLM) | 117M-1.5B | 1024 | Causal LM | Text generation, fine-tuning baseline | Outdated |
| GPT-3 | Decoder (CLM) | 175B | 4096 | Causal LM (massive) | Few-shot learning, instruction following | API only, expensive |
| GPT-4 | Decoder (rumored MoE) | ~1.8T? | 128K | — | Complex reasoning, multimodal | Expensive, closed |
| T5-base | Encoder-Decoder | 220M | 512 | Span corruption | Text-to-text (translation, summarization, QA) | Slower than encoder for classification |
| BART | Encoder-Decoder | 400M | 1024 | Noising tasks | Summarization, translation | Less used now |
| FLAN-T5 | Encoder-Decoder | 80M-11B | 512 | Instruction tuned T5 | Instruction following, zero-shot | Limited context |
| LLaMA 2-7B | Decoder | 7B | 4096 | Causal LM (2T tokens) | Open-source deployment, fine-tuning | Older, context limited |
| LLaMA 2-70B | Decoder | 70B | 4096 | Causal LM | Best open-source (2023) | Large memory |
| LLaMA 3-8B | Decoder | 8B | 8192 | Causal LM (15T tokens) | Strong open-source small model | — |
| LLaMA 3-70B | Decoder | 70B | 8192 | Causal LM | Best open-source (mid-2024) | Large memory |
| LLaMA 3.1-405B | Decoder | 405B | 128K | Causal LM | GPT-4 class open-source | Very large |
| Mistral 7B | Decoder | 7B | 32K | Causal LM | Efficient open-source alternative | — |
| Mixtral 8×7B | Decoder (MoE) | 8B active (47B total) | 32K | Causal LM | Speed of 7B, quality of 40B | MoE serving complexity |
| Phi-3-mini | Decoder | 3.8B | 128K | Curated data emphasis | Edge deployment, reasoning | Smaller |
| Gemini 1.5 Pro | Multimodal | — | 1M | Multimodal | Extremely long context, multimodal | Google API only |
| Claude 3.5 Sonnet | Decoder | — | 200K | Constitutional AI | Coding, reasoning, long documents | Anthropic API |
| Qwen2-72B | Decoder | 72B | 128K | Causal LM | Multilingual, strong at code | — |

---

## 7. Fine-Tuning Decision Matrix

### When to Use Each Approach

| Method | When to Use | VRAM Required (7B model) | Trainable Params | Quality vs Full FT |
|--------|-------------|--------------------------|------------------|---------------------|
| **Full Fine-tuning** | Large labeled dataset (> 100K), maximum quality needed, dedicated infra | 80GB+ (FP16) | 100% (7B) | Baseline |
| **LoRA** | Medium dataset (1K-100K), limited GPU, PEFT sufficient | 16-24GB | 0.1-1% (millions) | ~95% of full FT |
| **QLoRA** | Small GPU (consumer), 4-bit quantized base + LoRA | 10-16GB | 0.1-1% | ~90-95% of full FT |
| **Prefix Tuning** | Very few params, frozen base, distribution shift | < 4GB overhead | < 0.01% | Lower than LoRA |
| **Prompt Tuning** | No gradient through base model needed | < 2GB overhead | < 0.001% | Lowest, needs large model |
| **Adapter Layers** | Modular multi-task fine-tuning | ~4-8GB overhead | 1-4% | ~95% of full FT |
| **In-context learning (prompting)** | No data, or data < 100 examples, GPT-4 class model | 0 (API) | 0 | Varies by task |
| **RLHF / DPO** | Align with human preferences, reduce harmful outputs | 2-3× FT memory | 100% | Different axis |

### LoRA Key Details

```python
# LoRA decomposes weight update: ΔW = A × B
# A: [d × r], B: [r × k]  where r << min(d, k)
# During forward pass: h = W₀x + α/r × BAx
# α is scaling factor (default: same as r)
# Typically: r ∈ {4, 8, 16, 32, 64}; r=16 is common default
# lora_alpha = 2r is common (scaling = 2)

# Which layers to add LoRA to? (common configs)
# Minimal: q_proj, v_proj only
# Standard: q_proj, k_proj, v_proj, o_proj
# Aggressive: + gate_proj, up_proj, down_proj (all linear layers)
```

### QLoRA Key Details

```
QLoRA = Quantize base model to 4-bit NF4 + train LoRA adapters in BF16

4-bit NF4 (Normal Float 4):
  - Designed for normally distributed weights (typical NN weights)
  - Better than INT4 for weight distributions seen in practice
  - Double quantization: quantize the quantization constants too
  - Paged optimizer: offload optimizer states to CPU RAM during spikes

Memory calculation for 7B QLoRA:
  Base model (NF4):     7B × 0.5 bytes = 3.5 GB
  LoRA adapters (BF16): ~30M × 2 bytes = 60 MB
  Gradients:            ~30M × 2 bytes = 60 MB
  Optimizer states:     ~30M × 8 bytes = 240 MB
  Activations:          ~4-8 GB (batch-dependent)
  Total: ≈ 10-16 GB → fits on single A100 40GB or RTX 4090 24GB
```

---

## 8. Quantization Formats Deep Dive

### Format Comparison

| Format | Bits | Range | Precision | Training | Inference | Framework Support |
|--------|------|-------|-----------|----------|-----------|-------------------|
| FP32 | 32 | ±3.4×10³⁸ | 7 decimal digits | Yes (baseline) | Slow, high memory | All |
| FP16 | 16 | ±65504 | 3-4 digits | Mixed precision | Fast on GPU | All |
| BF16 | 16 | ±3.4×10³⁸ | 2-3 digits | Preferred | Fast on Ampere+ | PyTorch, JAX |
| INT8 | 8 | -128 to 127 | Integer only | No (PTQ) | Fast, quantization noise | bitsandbytes, TensorRT |
| INT4 | 4 | -8 to 7 | Very coarse | QLoRA only | Very fast, significant quality loss | bitsandbytes, GPTQ |
| NF4 | 4 | adaptive | Normal distribution optimized | QLoRA | — | bitsandbytes |
| GGUF Q4_K_M | ~4.5 | Mixed 4/6-bit | Good for CPU | No | CPU/GPU via llama.cpp | llama.cpp, Ollama |
| GGUF Q8_0 | 8 | Uniform | Near FP16 | No | CPU/GPU | llama.cpp |
| AWQ (4-bit) | 4 | Activation-aware | Better than GPTQ | No | Fast GPU | AutoAWQ |
| GPTQ (4-bit) | 4 | Layer-wise | Good | No | Fast GPU | AutoGPTQ |

### Quantization Decision Guide

```
Training:        FP32 (small models) → BF16 mixed precision (large models) → QLoRA (memory constrained)

Inference target:
  Cloud GPU (A100/H100): BF16 — best quality, fully supported
  Cloud GPU (cost optimize): INT8 via bitsandbytes or TensorRT
  Consumer GPU (24GB): GPTQ 4-bit or AWQ 4-bit
  CPU / Edge (llama.cpp): GGUF Q4_K_M (speed) or GGUF Q8_0 (quality)
  Mobile:  INT4 / GGML

Quality order (best to worst):
  BF16 > INT8 > GPTQ-4bit ≈ AWQ-4bit > GGUF Q4_K_M > GGUF Q2_K
```

---

## 9. Decoding Strategies

| Strategy | Description | Temperature | When to Use | Avoid When |
|----------|-------------|-------------|-------------|------------|
| Greedy | Always pick highest probability token | 1.0 (fixed) | Deterministic output, debugging | Creative tasks (repetitive) |
| Beam Search | Maintain k hypotheses, pick best sequence | 1.0 (fixed) | Machine translation, summarization where diversity not needed | Open-ended generation (degenerate repetition) |
| Top-k sampling | Sample from top k tokens | Applies | General-purpose generation | When k is too large or too small |
| Top-p (nucleus) | Sample from smallest set summing to prob p | Applies | Creative, diverse outputs | Factual tasks (hallucination risk) |
| Temperature | Scale logits: p_i = exp(logit_i/T) / Z | T parameter | Tune creativity vs coherence | — |
| Contrastive Search | Penalize tokens similar to previous context | — | Reduces repetition without sampling noise | — |
| Speculative Decoding | Draft with small model, verify with large | — | 2-3× speedup on large model inference | When small model domain mismatch is high |

### Typical Parameter Values

```
Factual / code generation:
  temperature: 0.0-0.2
  top_p: 0.9
  top_k: 50

Balanced (default):
  temperature: 0.7
  top_p: 0.9
  top_k: 50

Creative / story:
  temperature: 0.9-1.2
  top_p: 0.95
  top_k: 100

Beam search (translation):
  num_beams: 4-8
  no_repeat_ngram_size: 3
  length_penalty: 1.0

Remember:
  T → 0: deterministic (greedy)
  T → ∞: uniform random sampling
  Low T: confident but repetitive
  High T: diverse but incoherent
```

---

## 10. Attention Variants Comparison

| Variant | KV Cache Size | Speed | Max Context | Used In |
|---------|--------------|-------|-------------|---------|
| Multi-Head Attention (MHA) | O(n × d_model) per layer | Baseline | Limited by memory | BERT, GPT-2 |
| Multi-Query Attention (MQA) | O(n × d_k) per layer (1 KV head) | 2-4× MHA at inference | Extended | GPT-3 variants |
| Grouped Query Attention (GQA) | O(n × d_k × g) where g = n_groups | Between MHA and MQA | Extended | LLaMA 2/3, Mistral |
| Sliding Window Attention | O(n × w) where w = window | ~O(n) | Theoretically unlimited | Mistral, Longformer |
| Flash Attention | Same as MHA but IO-optimized | 2-4× faster, same memory theoretically | Extended in practice | PyTorch 2.0+, all modern LLMs |

```
GQA intuition:
  MHA: n_heads KV pairs per head (expensive KV cache)
  MQA: 1 shared KV pair (cheapest, slight quality drop)
  GQA: g groups, each sharing KV (balance)
  LLaMA 3-70B: 8 KV heads, 64 query heads → GQA ratio 8:1
```

---

## 11. LLM Evaluation Benchmarks Quick Reference

| Benchmark | Measures | Format | Typical GPT-4 Score | Typical 7B Open Score |
|-----------|----------|--------|--------------------|-----------------------|
| **MMLU** | Massive multitask language understanding (57 subjects) | 5-choice MCQ | ~87% | 60-65% |
| **HellaSwag** | Commonsense NLI, sentence completion | 4-choice MCQ | ~95% | 75-80% |
| **HumanEval** | Python code generation (164 problems) | pass@1 | ~85% | 30-50% |
| **MBPP** | Python programming problems (500) | pass@1 | ~80% | 40-60% |
| **GSM8K** | Grade school math word problems (8.5K) | Open-ended | ~90% | 50-70% |
| **MATH** | Competition math (12.5K problems) | Open-ended | ~70% | 10-30% |
| **TruthfulQA** | Truthfulness, avoids common misconceptions | MC + generation | ~60% | 40-55% |
| **BIG-Bench Hard (BBH)** | Difficult reasoning tasks from BIG-Bench | Chain-of-thought | ~85% | 40-60% |
| **MT-Bench** | Multi-turn instruction following (80 questions) | LLM judged 1-10 | ~9.0/10 | 6-8/10 |
| **HELM** | Holistic evaluation: accuracy + calibration + fairness | Multiple | Top tier | Middle tier |
| **AlpacaEval** | Instruction following vs GPT-4 | Win rate | 100% (baseline) | 5-20% |
| **GPQA** | Graduate-level expert questions | 4-choice | ~50% (hard) | 25-35% |
| **IFEval** | Instruction following with verifiable constraints | Exact match | ~85% | 50-70% |
| **BLEU** | Machine translation (n-gram precision) | 0-100 | N/A for LLMs | N/A |
| **ROUGE-L** | Summarization recall | 0-1 | N/A | N/A |

### Benchmark Caveats for Interviews

```
1. Data contamination: many benchmarks may be in pretraining data
   → Always ask: was eval set in training data?

2. MMLU is saturating: GPT-4 class models > 85%, use GPQA for harder evals

3. HumanEval ≠ production code quality: 164 problems, algorithmic focus
   → SWE-bench is more realistic (real GitHub issues)

4. MT-Bench / AlpacaEval use GPT-4 as judge → positional bias, verbose bias

5. Always report multiple benchmarks: single-benchmark gaming is easy
```

---

## 12. KV Cache — Critical for Inference Scaling

```
KV Cache stores key/value tensors for all previous tokens to avoid recomputation.

Memory for KV cache:
  = 2 (K and V) × n_layers × n_kv_heads × d_k × seq_len × bytes_per_element

Example: LLaMA 2-7B in FP16, 4096 tokens:
  = 2 × 32 × 32 × 128 × 4096 × 2 bytes
  = 2 × 32 × 32 × 128 × 8192
  = 2 × 32 × 32 × 1,048,576 bytes
  ≈ 2.1 GB for 4096 tokens (scales linearly with seq_len)

At 128K tokens (LLaMA 3.1):
  = 2.1 GB × (128K/4K) = 67 GB → significant memory pressure

Mitigation strategies:
  - Grouped Query Attention (GQA): reduce n_kv_heads
  - PagedAttention (vLLM): virtual memory for KV cache
  - Streaming LLM: sliding window KV cache
  - Quantized KV cache: INT8 KV cache (2× savings)
  - Flash Decoding: parallel KV cache reading
```

---

## 13. PEFT / Efficient Training Summary

```
Full Fine-tuning:    Update all weights. Max quality. Max memory.
                     7B model: need ~80GB VRAM for FP16

LoRA:                Inject low-rank matrices. Train only them.
                     r=16: adds ~30M params to 7B model (0.4%)
                     Memory: ~16-24GB for 7B in BF16

QLoRA:               Quantize base to 4-bit. Train LoRA in BF16.
                     Memory: ~10-16GB for 7B
                     Quality: ~1-3% drop vs full FT on most tasks

Soft Prompts (Prompt Tuning):
                     Learn a prefix of "virtual tokens" in embedding space
                     Only works well for very large models (> 10B)
                     ~10MB of extra params

Adapters:            Small FFN bottleneck modules between transformer layers
                     Similar to LoRA but serial (slower inference)
                     Good for multi-task: swap adapter per task

IA³:                 Scale activations by learned vectors (fewest params)
                     Even less memory than LoRA, slightly lower quality

RAG vs Fine-tuning decision:
  Use RAG when:  knowledge is dynamic, need source citations, < 10K examples
  Use FT when:   style/format change needed, task-specific reasoning,
                 reduce latency (no retrieval step), knowledge is static
```

---

## 14. Common LLM Architecture Interview Questions

```
Q: Why scale d_ff = 4 × d_model?
A: Empirical finding from original paper. FFN acts as key-value memory.
   SwiGLU (LLaMA) uses 2/3 × 4 × d_model ≈ 2.67 × d_model to match params
   while gaining gating benefit.

Q: Why layer norm before attention (Pre-LN) vs after (Post-LN)?
A: Post-LN (original): more stable final layer, but gradients can vanish in deep networks
   Pre-LN: better gradient flow, easier to train deep models
   → LLaMA, GPT-2 modern variants use Pre-LN (RMSNorm)
   RMSNorm: no centering, just scale → faster, similar quality

Q: Why is d_k = d_model / n_heads?
A: Ensures total computation stays same as single-head attention.
   Total params in attention: 4 × d_model² regardless of n_heads.

Q: What is the computational complexity of attention?
A: O(n² × d_model) where n = sequence length
   This is why long context is expensive → O(n²) blows up

Q: How does Flash Attention help?
A: Reorders computation to avoid materializing the full n×n attention matrix.
   Uses tiling to fit in SRAM (fast) instead of HBM (slow).
   Same math, same output, 2-4× faster, O(n) memory for attention matrices.
   Flash Attention 2: better work partitioning across GPU warps.
   Flash Attention 3: optimized for H100 FP8.
```
