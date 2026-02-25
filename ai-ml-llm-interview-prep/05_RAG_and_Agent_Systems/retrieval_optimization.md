# Retrieval Optimization for RAG

If a RAG system provides a bad answer, 90% of the time it is a retrieval failure, not an LLM failure. Optimizing the retrieval phase is the core job of a RAG engineer.

---

## 1. Query Transformation
Users write terrible queries (e.g., "shipping policy?"). Vector databases perform literal semantic matching. If the document says "Logistics Transportation Guidelines," the mathematical overlap might be low. We use a fast LLM to fix the query *before* searching.

*   **Query Rewriting:** LLM rewrites the query for clarity and adds synonyms. ("What is the corporate shipping policy for international logistics?").
*   **HyDE (Hypothetical Document Embeddings):** 
    1. Ask the LLM to *guess* the answer to the query without context. (It will hallucinate).
    2. Embed the hallucinated answer.
    3. Use that vector to search the database. 
    *Why?* The vector of a fake answer often matches the vector of the real answer much better than a question vector matches an answer vector (Symmetric Search).
*   **Multi-Query (Sub-Queries):** If the user asks "How do I ship batteries to Germany and what are the taxes?", split this into two separate vector searches: "battery shipping rules" and "Germany import taxes".

## 2. Hybrid Search
Dense Vector Search is amazing for "vibes" and semantic meaning. It is terrible at exact keyword matching (e.g., finding a specific tracking ID `JD-88291X` or a specific product name).

*   **The Solution:** Combine Vector Search with traditional BM25 (TF-IDF keyword search).
*   **Mechanism:** You run both searches simultaneously. You get a ranked list from Vector Search and a ranked list from BM25.
*   **Reciprocal Rank Fusion (RRF):** An algorithm that merges the two lists. It assigns a score based on rank position (e.g., Document A was rank 2 in Vector and rank 5 in BM25). The top combined scores are passed to the LLM.

## 3. Metadata Filtering (Pre-Filtering)
Do not rely on Vector Search to perform logical database operations.
*   **Bad:** Embedding the query "Show me Q3 2023 revenue reports" and hoping the vector math perfectly aligns with documents from Q3 2023.
*   **Good:** Use an LLM or regex to extract `Date = Q3 2023`. Then execute a hard database query: `SELECT * WHERE date >= 2023-07-01 AND date <= 2023-09-30`. *Then* perform the Vector Search only on that tiny filtered subset.

## 4. Chunking Strategies
(See `chunking_strategies.md` for full details).
*   **Small-to-Big (Parent-Child):** Retrieve a highly precise 200-token chunk, but pass its surrounding 1000-token parent chunk to the LLM for context.
*   **Sentence-Window Retrieval:** Retrieve the specific sentence that matched the query, plus the 3 sentences before and after it.

## 5. Reranking (The Silver Bullet)
Vector search relies on Bi-Encoders (the query and document are embedded separately and compared via dot product). This is fast but mathematically imprecise.
*   **The Fix:** Retrieve a large number of documents (e.g., Top 30) using fast Vector Search. Then pass them to a **Cross-Encoder** (Reranker).
*   **Cross-Encoder:** Concatenates the Query and the Document into a single string `[Query] [SEP] [Document]` and runs them through a Transformer *together*. The self-attention mechanism deeply analyzes the relationship between the exact words in the query and document, outputting a highly accurate relevance score (0 to 1). 
*   Take the Top 3 from the Cross-Encoder and pass them to the LLM.

## Interview Strategy
"To optimize retrieval, I always start by implementing **Hybrid Search (BM25 + Vectors)** to ensure we don't miss exact keyword matches. The highest ROI improvement after that is inserting a **Cross-Encoder Reranker** step just before the LLM generation phase, which drastically reduces the noise in the final context window."