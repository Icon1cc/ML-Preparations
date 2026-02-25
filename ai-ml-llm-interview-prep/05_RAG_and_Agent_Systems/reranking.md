# Reranking in RAG Systems

If you only implement one advanced RAG technique, it should be Reranking. It provides the highest ROI for improving accuracy and reducing hallucinations.

---

## 1. The Core Problem: The Bi-Encoder Bottleneck
Standard Vector Databases use **Bi-Encoders** (like OpenAI `text-embedding-3`).
*   **How it works:** The user's Query is processed into Vector A. The Document was previously processed into Vector B. We calculate the Cosine Similarity between them.
*   **The Flaw:** The neural network never actually "sees" the Query and Document together. The math is just comparing two isolated summaries. It frequently retrieves documents that share similar vocabulary but answer the wrong question.

## 2. The Solution: The Cross-Encoder (Reranker)
A Cross-Encoder is a different type of Transformer model (often based on BERT).
*   **How it works:** It takes the Query and the Document simultaneously as a single input string: `Query: [text] Document: [text]`.
*   **The Magic:** Because they are processed together, the Transformer's Self-Attention mechanism allows every word in the Query to attend to every word in the Document. It deeply understands the logical relationship between them. It outputs a single probability score (e.g., 0.95 relevance).
*   **The Flaw:** It is extremely slow. You cannot run a Cross-Encoder against 1 million documents in a database; it would take hours.

## 3. The Two-Stage Retrieval Pipeline
To get the speed of Bi-Encoders and the accuracy of Cross-Encoders, we combine them.

1.  **Stage 1: Fast Retrieval (Bi-Encoder / Vector DB)**
    *   Query the Vector DB for the Top $K$ results (where $K$ is relatively large, e.g., $K=50$ or $K=100$).
    *   This step is purely about high **Recall** (making sure the real answer is *somewhere* in the top 50).
2.  **Stage 2: Accurate Reranking (Cross-Encoder)**
    *   Pass the Query and those 50 retrieved documents to the Reranker model.
    *   The model scores all 50 documents and sorts them.
    *   This step is about high **Precision**.
3.  **Stage 3: LLM Generation**
    *   Take only the Top $N$ documents from the reranked list (e.g., $N=3$ or $N=5$) and inject them into the LLM prompt.

## 4. Popular Reranking Models

*   **Cohere Rerank API:** The industry standard commercial API. Incredibly easy to use, highly effective, and supports multi-lingual queries.
*   **BGE-Reranker (BAAI):** The leading open-source reranker available on HuggingFace. You host it yourself.
*   **Jina Reranker:** Another highly competitive open-source option.
*   **RankLLaMA:** Using an actual generative LLM (fine-tuned specifically for the task) to output a score. Slower, but sometimes more accurate for highly complex reasoning.

## 5. Why Reranking Saves Money
It seems counter-intuitive that adding another model saves money, but it does.
*   Without a reranker, to ensure the LLM gets the right answer, you might have to pass the Top 15 documents into the prompt. That's 10,000 tokens. GPT-4 charges per token.
*   With a reranker, you sort those 15 documents perfectly. You only need to pass the Top 3 documents into the prompt. That's 2,000 tokens.
*   You drastically cut your LLM API bill and reduce the risk of the "Lost in the Middle" hallucination phenomenon.