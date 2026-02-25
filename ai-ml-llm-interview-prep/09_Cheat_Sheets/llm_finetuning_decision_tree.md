# Cheat Sheet: LLM Fine-Tuning Decision Tree

Use this guide to determine if you actually need to fine-tune, and if so, which method to use.

---

## Question 1: Do you have a specialized task?
*   **NO:** (I want a general assistant). -> **Use a Base/Instruct Model** (GPT-4o, Llama 3).
*   **YES:** (I want it to follow a very specific output format, tone, or handle niche jargon). -> **Proceed to Q2**.

## Question 2: Is the knowledge dynamic or static?
*   **DYNAMIC:** (The data changes daily - e.g., prices, news, stock). -> **Use RAG**. Fine-tuning is too slow and expensive for updating facts.
*   **STATIC:** (The knowledge is fixed - e.g., medical textbooks, internal legal coding standards). -> **Consider Fine-tuning**.

## Question 3: Is it about *Knowledge* or *Format*?
*   **KNOWLEDGE:** (I want the model to "know" more facts). -> **RAG is better**. Fine-tuning is poor at adding facts and prone to hallucinations.
*   **FORMAT/STYLE:** (I want it to output ONLY valid JSON in a specific schema, or talk like a professional lawyer). -> **Fine-tuning is excellent**.

---

## If Fine-Tuning: Which Method?

| Method | When to use it? | Hardware Requirement |
| :--- | :--- | :--- |
| **Full Fine-Tuning** | You have massive data (1M+ rows) and want to fundamentally change the model's behavior. | **Massive** (A100/H100 clusters). |
| **LoRA** | The industry standard for most tasks. Balances speed, cost, and performance. | **Moderate** (1-2 consumer GPUs). |
| **QLoRA** | You want to fine-tune a huge model (e.g., 70B) on a budget. | **Low** (Single consumer GPU). |
| **DPO (Alignment)** | You want the model to prefer one type of answer over another (Helpful vs. Harmful). | After SFT is complete. |

---

## Decision Matrix Summary

| Goal | Best Approach |
| :--- | :--- |
| **Update model with current news** | RAG |
| **Teach model to use a private API** | RAG (to provide API docs) + Few-Shot |
| **Change the "personality" of the bot** | Fine-Tuning (SFT) |
| **Ensure strict JSON output format** | Fine-Tuning (LoRA) |
| **Extract entities from medical records** | Fine-Tuning (LoRA) |
| **Summarize company meetings** | RAG + Prompt Engineering |

**Golden Rule:** Start with **Prompt Engineering**, then **RAG**, and only move to **Fine-Tuning** if those fail to meet your performance or cost requirements. Most production systems use a **Hybrid of RAG + LoRA**.