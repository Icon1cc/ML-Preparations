# Hallucinations in LLMs

Hallucinations occur when a Large Language Model generates text that is grammatically correct and sounds plausible but is factually incorrect, nonsensical, or ungrounded in the provided context.

---

## 1. Why do LLMs Hallucinate?
LLMs are not databases querying facts; they are probabilistic engines predicting the next most likely token.
*   **Training Data Bias:** The model might have seen a wrong fact repeated many times on the internet.
*   **Compression:** An LLM compresses terabytes of data into gigabytes of weights. It "remembers" concepts, not exact strings, leading to mixed-up details (e.g., confusing the plot of two similar movies).
*   **Context Window Limits:** If a prompt is too long, the model loses track of the instructions or the context provided at the beginning ("Lost in the Middle").
*   **Sycophancy:** Models trained via RLHF are often trained to be "helpful." Sometimes, they would rather invent an answer to please the user than admit they don't know.

## 2. Types of Hallucination
*   **Intrinsic (Factual) Hallucination:** Contradicts established real-world facts. (e.g., "The capital of France is Berlin.")
*   **Extrinsic (Contextual) Hallucination:** Contradicts the specific context provided in the prompt. This is the most critical failure mode in RAG systems. (e.g., You provide a document saying "Revenue was $5M", but the LLM says "Revenue was $10M".)

## 3. Mitigation Strategies (Engineering Solutions)

If you are asked how to fix hallucination in an interview, do NOT suggest "retraining the model." Use these engineering approaches:

### A. Grounding via RAG (Retrieval-Augmented Generation)
The most effective way to prevent intrinsic hallucination. Instead of relying on the model's internal memory, you search a trusted database, inject the facts into the prompt, and instruct the model: "Answer ONLY using the provided context."

### B. Prompt Engineering Guardrails
*   **Explicit "I don't know" instruction:** "If the answer is not contained in the text below, you must reply exactly with: 'Information not found'."
*   **Chain-of-Thought (CoT):** Asking the model to "think step-by-step" forces it to logically link its thoughts, drastically reducing math and logic hallucinations.
*   **Ask for Citations:** Instruct the model to cite the specific paragraph or chunk ID it used to generate the answer.

### C. Generation Parameters
*   **Temperature (T):** Controls randomness. $T=0$ makes the model deterministic (always picking the highest probability token). Use $T=0$ for factual Q&A, data extraction, or code generation. Use $T=0.7+$ only for creative writing.
*   **Top-P (Nucleus Sampling):** Limits the pool of tokens the model can choose from to only the most likely ones.

### D. System-Level Checks (Self-Reflection)
Implement a multi-agent or multi-prompt setup.
1.  **Generator:** Creates the answer.
2.  **Critic:** A separate prompt that receives the context and the generated answer, with the instruction: "Does the Answer contain any facts not explicitly stated in the Context? Reply YES or NO." If YES, force a regeneration.

## 4. Measuring Hallucination
You cannot measure hallucination with traditional metrics like BLEU or ROUGE.
*   **LLM-as-a-Judge:** Use a stronger, more expensive model (like GPT-4) to grade the outputs of a cheaper, faster model (like Llama-3-8B) for factual consistency against a reference text.
*   **Frameworks:** Tools like **RAGAS** or **TruLens** specifically calculate "Faithfulness" scores to measure contextual hallucination.