# LLM Evaluation: Metrics and Strategies

Evaluating Large Language Models is fundamentally different from evaluating traditional ML. There is no single "accuracy" score. You must use a combination of automated metrics, model-based evaluation, and human review.

---

## 1. Traditional NLP Metrics (The Baselines)
These metrics compare the model's output to a reference string (ground truth).
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Used for summarization. Measures the overlap of n-grams between the generated summary and a human reference.
*   **BLEU (Bilingual Evaluation Understudy):** Used for translation. Measures precision of n-gram overlap.
*   **Cons:** These metrics only look at exact word matches. They penalize a model for using a synonym (e.g., "happy" vs. "joyful"), even if the meaning is identical. They are increasingly irrelevant for modern LLMs.

## 2. Model-Based Evaluation (LLM-as-a-Judge)
Using a more powerful, expensive model (like GPT-4o or Claude 3.5 Sonnet) to grade the outputs of a smaller/cheaper model.
*   **Mechanism:** You provide the judge model with a rubric and a scale (1-5).
*   **Example Prompt:** "Rate the following response based on conciseness and helpfulness from 1 to 5. [Response]"
*   **Pros:** Much closer to human judgment than ROUGE/BLEU. Scales instantly.
*   **Cons:** The judge model can be biased (it often prefers longer responses or its own writing style). It is expensive to run at scale.

## 3. Task-Specific Evaluation

### A. Code Generation
*   **Pass@k:** You generate $k$ code samples for a problem. If any of the $k$ samples pass the unit tests, it is considered a success. This is the gold standard for models like GitHub Copilot.

### B. Classification / Extraction
*   **Precision, Recall, F1-Score:** If the LLM is being used to extract entities or classify sentiment, standard ML metrics apply.

### C. RAG (Retrieval Augmented Generation)
*   **RAGAS Framework:** Evaluates the "RAG Triad" (Faithfulness, Answer Relevance, Context Precision). (See the specific RAG Evaluation file for more).

## 4. Benchmarks (The Public Leaders)
*   **MMLU (Massive Multitask Language Understanding):** Covers 57 subjects across STEM, humanities, and more. Tests general world knowledge.
*   **GSM8K:** Grade school math word problems. Tests multi-step reasoning.
*   **HumanEval:** Python coding tasks.
*   **LMSYS Chatbot Arena:** A dynamic, "Elo-style" leaderboard where humans blindly vote on two model outputs. Currently the most trusted "real-world" metric.

## 5. Production Guardrails (Real-time Eval)
In production, you evaluate *every* request.
*   **Toxicity/Safety:** Using models like `Llama-Guard` to detect if the user's prompt or the model's response is harmful.
*   **Pangea / Lakera:** Specialized APIs that check for Prompt Injection or PII (Personally Identifiable Information) leakage.

## Interview Strategy
If an interviewer asks "How do you evaluate your chatbot?", answer:
1.  "First, I define the core task (e.g., Customer Support)."
2.  "I build a **Golden Dataset** of 100 diverse, high-stakes questions and human-verified answers."
3.  "I use **LLM-as-a-Judge** (GPT-4) to grade the responses against a specific rubric."
4.  "I monitor **hallucination rates** using faithfulness checks."
5.  "Finally, I track **business metrics**, like the percentage of users who clicked 'Helpful' vs. 'Not Helpful' in the UI."