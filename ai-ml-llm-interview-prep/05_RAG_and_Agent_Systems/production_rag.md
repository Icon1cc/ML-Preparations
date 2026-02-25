# Production RAG: Beyond the Prototype

Most RAG tutorials end with a basic LangChain loop. In production, "Naive RAG" fails. This file covers the advanced techniques required to make RAG reliable at scale.

---

## 1. Handling Complex Documents

### A. The PDF Problem
Standard text extractors mangle tables and multi-column layouts.
*   **Solution:** Use **Layout-Aware Parsing** (e.g., `Unstructured.io`, `AWS Textract`, or `Marker`). These tools identify headers, tables, and images. 
*   **Table Strategy:** Don't just chunk a table. Convert it to **Markdown** or **HTML**. LLMs understand these formats much better than raw text blocks.

### B. Multi-Modal RAG
What if the answer is in a chart?
*   **Strategy 1 (Captioning):** Use a Vision model to generate a text description of every image/chart. Embed the description.
*   **Strategy 2 (Multi-modal Embeddings):** Use a model like **CLIP** or **ColPali** that can embed both images and text into the same vector space.

## 2. Advanced Retrieval Patterns

### A. Small-to-Big (Parent-Child)
*   **The Issue:** Small chunks are better for retrieval accuracy, but the LLM needs more surrounding context to answer correctly.
*   **The Fix:** Embed small 200-token chunks. When one is retrieved, fetch the 1000-token "Parent" chunk that contains it and pass that to the LLM.

### B. Query Rewriting & Expansion
*   **Multi-Query:** The LLM generates 3 variations of the user's question. We search for all 3 and take the union of results.
*   **HyDE (Hypothetical Document Embeddings):** The LLM generates a fake, "hypothetical" answer to the query. We embed that *fake* answer to search the DB. This works because embeddings of answers match other answers better than questions match answers.

## 3. The Reranking Power-Up
This is the single most important addition to any production RAG system.
1.  **Retrieve** 50-100 documents quickly using a fast Vector DB.
2.  **Rerank** them using a **Cross-Encoder** (e.g., `BGE-Reranker`, `Cohere Rerank`). Cross-encoders are too slow to search millions of docs, but extremely accurate at comparing a query against a small batch. 
3.  **Pass** only the top 3-5 reranked docs to the LLM. This drastically reduces noise and improves accuracy.

## 4. Post-Generation Guardrails
*   **Citation Verification:** Force the LLM to output `[Source 1]` after every claim. Use a programmatic check to ensure `Source 1` actually exists in the provided context.
*   **NLI (Natural Language Inference):** Use a small model to check if the generated Answer is logically "entailed" by the Context.

## 5. Performance and Latency
*   **Semantic Caching:** Store common query-response pairs in Redis. If a new query is semantically similar to a cached one, serve the cached answer instantly.
*   **Parallelism:** Fire off the Vector Search and any external API calls (e.g., checking a SQL DB) in parallel using Python `asyncio`.
*   **Speculative Decoding:** Using a smaller model to guess the next tokens while a larger model validates them, speeding up generation.

## 6. Interview Tip: Cost Optimization
LLM tokens are expensive. 
*   "We reduce costs by using a small model (Llama 8B) for query rewriting and a large model (GPT-4) only for the final synthesis."
*   "We use reranking to ensure we only send the absolute most relevant 2000 tokens to the LLM, rather than 8000 tokens of noise."