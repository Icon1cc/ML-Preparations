# 04 Large Language Models (LLMs)

## Overview

Large Language Models represent a paradigm shift in AI. This section covers everything from **how LLMs are built** to **how to use them effectively in production**. Critical for modern AI/ML interviews.

## 📚 Contents

### LLM Fundamentals
1. [**LLM Training Pipeline**](./llm_training_pipeline.md) ⭐⭐⭐
   - Pretraining (next-token prediction)
   - Supervised Fine-Tuning (SFT)
   - Reinforcement Learning from Human Feedback (RLHF)
   - Direct Preference Optimization (DPO)
   - Constitutional AI

2. [**Scaling Laws**](./scaling_laws.md)
   - Chinchilla scaling laws
   - Compute-optimal training
   - Emergent abilities

3. [**Context Windows and Attention**](./context_windows.md)
   - Positional encoding limitations
   - Extending context (ALiBi, RoPE, etc.)
   - Context window vs effective context

### Working with LLMs

4. [**Prompt Engineering**](./prompt_engineering.md) ⭐⭐⭐
   - Zero-shot, few-shot, chain-of-thought
   - Prompt templates and best practices
   - Advanced techniques (ReAct, Tree of Thoughts)

5. [**LLM Fine-Tuning**](./llm_finetuning.md) ⭐⭐⭐
   - When to fine-tune vs RAG vs prompting
   - Full fine-tuning
   - Parameter-Efficient Fine-Tuning (PEFT)
     - LoRA (Low-Rank Adaptation)
     - QLoRA (Quantized LoRA)
     - Prefix tuning, Adapter layers
   - Dataset preparation
   - Evaluation

6. [**Quantization**](./quantization.md) ⭐⭐
   - Why quantize? (speed, memory, cost)
   - 8-bit, 4-bit, GPTQ, AWQ, GGUF
   - Tradeoffs: accuracy vs efficiency

7. [**Function Calling and Tool Use**](./function_calling.md) ⭐⭐
   - How function calling works
   - Structured output generation
   - Tool-augmented LLMs

### LLM Evaluation & Safety

8. [**LLM Evaluation Methods**](./llm_evaluation.md) ⭐⭐⭐
   - Traditional metrics (perplexity, BLEU)
   - Human evaluation
   - LLM-as-judge
   - Benchmarks (MMLU, HumanEval, etc.)

9. [**Hallucinations**](./hallucinations.md) ⭐⭐⭐
   - What causes hallucinations?
   - Detection methods
   - Mitigation strategies

10. [**Safety and Guardrails**](./safety_and_guardrails.md) ⭐⭐
    - Prompt injection attacks
    - Content filtering
    - Guardrail systems (NeMo Guardrails, Llama Guard)

### LLM Ecosystem

11. [**LLM Inference Optimization**](./inference_optimization.md) ⭐⭐
    - vLLM, Text Generation Inference (TGI)
    - KV cache optimization
    - Batching strategies (continuous batching)
    - Speculative decoding

12. [**Hugging Face Ecosystem**](./huggingface_ecosystem.md) ⭐⭐
    - transformers library
    - Model Hub
    - Datasets library
    - Inference endpoints

---

## 🎯 Learning Objectives

After this section, you should be able to:

- Explain the LLM training pipeline (pretraining → SFT → RLHF)
- Decide when to use prompting vs fine-tuning vs RAG
- Fine-tune an LLM using LoRA/QLoRA
- Optimize LLM inference for production
- Evaluate LLM outputs effectively
- Mitigate hallucinations and ensure safety

---

## ⏱️ Time Estimate

**Total: 12-15 hours**

**High Priority (Interview Critical):**
- LLM Training Pipeline: 2 hours ⭐⭐⭐
- Prompt Engineering: 2 hours ⭐⭐⭐
- Fine-Tuning (LoRA/QLoRA): 2.5 hours ⭐⭐⭐
- Hallucinations: 1 hour ⭐⭐⭐
- LLM Evaluation: 1.5 hours ⭐⭐⭐

**Medium Priority:**
- Quantization: 1 hour ⭐⭐
- Inference Optimization: 1 hour ⭐⭐
- Function Calling: 1 hour ⭐⭐

**Lower Priority (but useful):**
- Scaling Laws: 30 min
- Context Windows: 30 min
- Safety: 30 min

---

## 🔥 Interview Focus Areas

### Most Common Interview Questions

1. **"Explain how LLMs are trained"**
   - Pretraining (unsupervised, next-token prediction, massive scale)
   - SFT (supervised examples, instruction-following)
   - RLHF (human preferences, reward model, PPO)
   - See: [LLM Training Pipeline](./llm_training_pipeline.md)

2. **"When would you fine-tune vs use RAG?"**
   - Fine-tune: Need specific style, domain knowledge baked in, have labeled data
   - RAG: Need up-to-date info, interpretability, cost-effective
   - Often best: Hybrid (fine-tune + RAG)
   - See: [LLM Fine-Tuning](./llm_finetuning.md)

3. **"How do you handle hallucinations?"**
   - Detection: Fact-checking, uncertainty estimation
   - Mitigation: RAG, citations, temperature tuning, prompt engineering
   - See: [Hallucinations](./hallucinations.md)

4. **"Explain LoRA and why it's useful"**
   - Low-rank decomposition of weight updates
   - Trains <1% of parameters
   - Fast, cheap, mergeable
   - See: [LLM Fine-Tuning](./llm_finetuning.md)

5. **"How do you evaluate LLM outputs?"**
   - Task-specific metrics (accuracy, BLEU, ROUGE)
   - Human evaluation
   - LLM-as-judge (GPT-4 evaluating outputs)
   - See: [LLM Evaluation](./llm_evaluation.md)

---

## 🚀 Quick Start Paths

### Path 1: LLM-Focused Interview Prep (6-8 hours)

**For roles emphasizing LLM applications:**

1. [LLM Training Pipeline](./llm_training_pipeline.md) - 2 hours
2. [Prompt Engineering](./prompt_engineering.md) - 2 hours
3. [Fine-Tuning (LoRA/QLoRA)](./llm_finetuning.md) - 2 hours
4. [Hallucinations](./hallucinations.md) - 1 hour
5. Review [Business Case Studies](../08_Case_Studies/Business_Case_Studies/) - 1-2 hours

### Path 2: Complete LLM Coverage (12-15 hours)

**For comprehensive understanding:**

**Week 1: Fundamentals**
1. LLM Training Pipeline - 2 hours
2. Scaling Laws - 30 min
3. Prompt Engineering - 2 hours

**Week 2: Fine-Tuning & Optimization**
4. LLM Fine-Tuning - 2.5 hours
5. Quantization - 1 hour
6. Inference Optimization - 1 hour

**Week 3: Evaluation & Production**
7. LLM Evaluation - 1.5 hours
8. Hallucinations - 1 hour
9. Safety & Guardrails - 30 min
10. Function Calling - 1 hour

### Path 3: Practical Hands-On (4-6 hours)

**For immediate application:**

1. Skim theory, focus on code examples
2. [Prompt Engineering](./prompt_engineering.md) - 1 hour (practice)
3. [Fine-Tuning with LoRA](./llm_finetuning.md) - 2 hours (code walkthrough)
4. [Hugging Face Ecosystem](./huggingface_ecosystem.md) - 1 hour (hands-on)
5. Build a simple LLM app - 1-2 hours

---

## 🎓 Key Concepts to Master

### 1. LLM Training Phases

```
Phase 1: Pretraining
─────────────────────
Goal: Learn language understanding
Data: Massive unlabeled text (trillions of tokens)
Method: Next-token prediction (causal language modeling)
Cost: $millions, thousands of GPUs
Output: Base model (e.g., GPT-3, LLaMA base)

↓

Phase 2: Supervised Fine-Tuning (SFT)
──────────────────────────────────────
Goal: Learn to follow instructions
Data: Instruction-response pairs (10K-100K examples)
Method: Supervised learning on demonstrations
Cost: $thousands, tens of GPUs
Output: Instruction-tuned model

↓

Phase 3: RLHF (Optional but powerful)
──────────────────────────────────────
Goal: Align with human preferences
Data: Human preference rankings
Method: Train reward model → PPO to optimize
Cost: $thousands to $tens of thousands
Output: Aligned model (e.g., ChatGPT, Claude)
```

### 2. When to Use What

| Approach | Use When | Cost | Effort | Updateability |
|----------|----------|------|--------|---------------|
| **Prompting** | General tasks, quick iteration | $ | Low | Instant |
| **Few-shot Prompting** | Task-specific, have examples | $ | Low | Instant |
| **RAG** | Need current info, citations | $$ | Medium | Easy (update docs) |
| **Fine-tuning (LoRA)** | Specific style/domain, have labels | $$$ | Medium | Moderate (retrain) |
| **Full Fine-tuning** | Need max performance, domain-specific | $$$$ | High | Hard (expensive retrain) |
| **Pretraining** | Building foundation model | $$$$$ | Very High | Very Hard |

### 3. LoRA vs QLoRA vs Full Fine-Tuning

| Method | Parameters Trained | Memory | Speed | When to Use |
|--------|-------------------|--------|-------|-------------|
| **Full Fine-Tuning** | 100% (e.g., 7B) | ~50GB | Slow | Max performance, ample resources |
| **LoRA** | <1% (e.g., 70M) | ~15GB | Fast | Good performance, limited resources |
| **QLoRA** | <1% (e.g., 70M) | ~8GB | Medium | Consumer GPU (RTX 4090), still good performance |

**Code Example (QLoRA):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

# LoRA config
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Now train on your dataset!
```

### 4. Prompt Engineering Patterns

**Zero-Shot:**
```
Classify sentiment: "The movie was terrible."
```

**Few-Shot:**
```
Classify sentiment:
"The movie was great." → Positive
"The movie was okay." → Neutral
"The movie was terrible." → ?
```

**Chain-of-Thought (CoT):**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 balls. How many tennis balls does he have now?

A: Let's think step by step:
- Roger started with 5 balls
- He bought 2 cans, each with 3 balls: 2 × 3 = 6 balls
- Total: 5 + 6 = 11 balls
```

**Structured Output:**
```
Extract entities from: "Apple Inc. released iPhone 15 in September 2023."

Return JSON:
{
  "company": "Apple Inc.",
  "product": "iPhone 15",
  "date": "September 2023"
}
```

---

## 💡 Pro Tips for Interviews

### 1. Show You Understand Tradeoffs

**Bad:** "I'll fine-tune the LLM."

**Good:**
"I'd consider three approaches:
1. **Prompting**: Fast to iterate, no training cost, but limited task-specific performance
2. **RAG**: Can inject current information, interpretable, but adds latency
3. **Fine-tuning**: Best performance for specific task, but requires labeled data and compute

Given we have 10K labeled examples and need high accuracy, I'd start with **LoRA fine-tuning** (cheaper than full fine-tuning) and evaluate against a RAG baseline. If we also need current information, I'd combine fine-tuned model with RAG."

### 2. Discuss Hallucinations Proactively

**Don't wait for interviewer to ask.** Mention it unprompted:

"One risk with LLMs is hallucination—confidently stating false information. To mitigate:
- Use RAG with citations so users can verify
- Lower temperature for factual tasks
- Implement fact-checking layer for critical applications
- Clear disclaimers about potential errors"

### 3. Show Production Thinking

**Beyond model training:**

"For production, I'd also consider:
- **Inference optimization**: Use vLLM for 2-3x throughput
- **Cost**: Batch requests, use smaller model where possible
- **Monitoring**: Track latency, cost per request, hallucination rate
- **Safety**: Implement content filtering and prompt injection detection
- **Fallbacks**: Handle rate limits and API failures gracefully"

### 4. Connect to Business Value

"Fine-tuning will cost ~$500 in compute and 2 days of engineering time. If it improves accuracy from 75% to 85%, that's 10% fewer customer service escalations, saving ~$50K/year. Clear ROI."

---

## 🔗 Connections to Other Sections

- **03 Modern NLP & Transformers**: Foundation for understanding LLM architecture
- **05 RAG & Agent Systems**: Using LLMs in retrieval and agent contexts
- **06 MLOps**: Deploying and monitoring LLMs in production
- **07 System Design**: Designing scalable LLM applications
- **08 Case Studies**: Applying LLMs to real business problems

---

## 📈 Success Metrics

You've mastered this section when you can:

- [ ] Explain LLM training pipeline (pretraining → SFT → RLHF) clearly
- [ ] Decide when to use prompting vs RAG vs fine-tuning
- [ ] Implement LoRA/QLoRA fine-tuning
- [ ] Design prompts effectively (zero-shot, few-shot, CoT)
- [ ] Discuss hallucination mitigation strategies
- [ ] Optimize LLM inference for production
- [ ] Evaluate LLM outputs using appropriate methods

---

## 🎯 Common Mistakes to Avoid

❌ "LLMs know everything" → They hallucinate, have knowledge cutoffs
❌ "Always fine-tune for best results" → Often prompting or RAG is sufficient and faster
❌ "Bigger models are always better" → Not for latency-sensitive or cost-constrained applications
❌ "RLHF eliminates all problems" → It aligns with human preferences but doesn't eliminate hallucinations or biases
❌ "Temperature=0 guarantees deterministic outputs" → Due to floating-point operations, not always exactly deterministic

---

**Ready to start?** → [LLM Training Pipeline](./llm_training_pipeline.md) (Most interview-critical)

**Next Section:** [05 RAG and Agent Systems](../05_RAG_and_Agent_Systems/README.md)
