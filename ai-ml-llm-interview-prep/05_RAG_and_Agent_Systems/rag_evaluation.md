# RAG Evaluation (RAGAS Framework)

You cannot improve a RAG system if you cannot measure it. "Vibes" and manual testing do not scale. You must implement automated evaluation pipelines. The industry standard framework for this is **RAGAS** (Retrieval Augmented Generation Assessment).

---

## The RAG Triad
To evaluate RAG, you must separate the evaluation into three distinct questions:
1.  Did we find the right documents? (Retrieval)
2.  Did the LLM stick to those documents? (Generation - Hallucination)
3.  Did the final answer actually help the user? (Generation - Usefulness)

RAGAS formalizes this into four core metrics using "LLM-as-a-Judge" (using a powerful model like GPT-4 to grade the pipeline).

## 1. Context Precision (Retrieval Metric)
*   **Question:** Are the retrieved chunks relevant to the user's query, and are the most relevant ones ranked at the top?
*   **Mechanism:** The LLM-Judge looks at the user query and the retrieved chunks. It checks if the chunks actually contain the answer. It heavily penalizes systems that retrieve 1 good chunk but bury it at rank #10 beneath 9 useless chunks.
*   **Fix:** If this is low, you need better **Reranking** (e.g., Cohere Rerank) or better **Chunking** (chunks are too noisy).

## 2. Context Recall (Retrieval Metric)
*   **Question:** Did we retrieve *all* the necessary information to answer the query?
*   **Requires:** A ground-truth reference answer.
*   **Mechanism:** The LLM-Judge compares the ground-truth answer against the retrieved context. Can the ground-truth answer be fully deduced purely from the retrieved chunks? If the ground truth mentions 3 facts, but the context only contains 2, recall is low.
*   **Fix:** If this is low, you need **Query Expansion** (the user asked a bad question), **Hybrid Search** (vector search missed keywords), or simply retrieving a higher `Top-K` number of documents.

## 3. Faithfulness (Generation Metric - Anti-Hallucination)
*   **Question:** Is the generated answer entirely derived from the retrieved context? Did the LLM make things up?
*   **Mechanism:** The LLM-Judge extracts all the distinct "claims" made in the generated answer. It then checks each claim against the retrieved context. If a claim cannot be inferred from the context, it is flagged as a hallucination.
*   **Fix:** If this is low, lower the **Temperature** to 0, improve the **System Prompt** ("Answer ONLY using the context"), or the retrieved context was completely empty and the LLM tried to guess.

## 4. Answer Relevance (Generation/End-to-End Metric)
*   **Question:** Does the generated answer directly address the user's original query?
*   **Mechanism:** This is clever. The LLM-Judge looks at the generated answer and tries to *reverse-engineer* what the question was. If the reverse-engineered question matches the original user query (measured via cosine similarity of their embeddings), the relevance is high. If the user asked "How do I ship a battery?", and the system answered "The company was founded in 1969," the reverse-engineered question will not match.
*   **Fix:** If this is low, the prompt might be poorly structured, causing the LLM to get confused about what it's supposed to do.

## Setting up an Eval Pipeline
1.  **Golden Dataset:** Create a set of 50-100 diverse questions and ground-truth answers (human curated).
2.  **CI/CD Integration:** Every time you change a chunk size, swap an embedding model, or tweak a system prompt, you run the Golden Dataset through the pipeline and generate the 4 RAGAS scores.
3.  **A/B Testing:** "Did changing chunk size from 500 to 250 improve Context Precision without hurting Context Recall?" Now you have numbers to prove it.