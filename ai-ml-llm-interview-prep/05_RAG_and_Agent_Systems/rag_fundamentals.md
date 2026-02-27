# RAG Fundamentals (Retrieval-Augmented Generation)

RAG is the industry-standard architecture for giving Large Language Models (LLMs) access to private, proprietary, or up-to-date information without requiring expensive fine-tuning.

---

## 1. The Core Problem RAG Solves
LLMs are frozen in time at their training cutoff. They confidently hallucinate when asked about proprietary data (e.g., "What is Company X's Q3 internal revenue report?") because they never saw it during training.
**Solution:** Don't rely on the LLM's internal memory. Retrieve the relevant documents and put them directly in the prompt.

## 2. The RAG Architecture

A standard RAG pipeline consists of two phases: Data Ingestion (Offline) and Retrieval/Generation (Online).

### Phase 1: Data Ingestion (Offline Pipeline)
1.  **Document Loading:** Extracting text from various sources (PDFs, Confluence, Jira, databases).
2.  **Chunking:** Splitting long documents into smaller, manageable pieces (e.g., 500-token chunks). *Crucial because LLMs have context limits and embedding models have sequence length limits.*
3.  **Embedding:** Passing each chunk through an Embedding Model (e.g., OpenAI `text-embedding-ada-002`, open-source `BGE` or `E5`) to convert the text into a dense vector representation (e.g., an array of 768 floats). Semantic meaning is captured in the vector space.
4.  **Vector Storage:** Storing the vectors and their corresponding text chunks (metadata) in a Vector Database (e.g., Pinecone, Milvus, Qdrant, pgvector).

### Phase 2: Retrieval and Generation (Online Pipeline)
1.  **User Query:** The user asks a question: "What is our routing policy for hazardous materials?"
2.  **Query Embedding:** The user's query is passed through the *exact same* embedding model used in Phase 1 to create a query vector.
3.  **Vector Search (Retrieval):** The Vector DB performs a similarity search (usually Cosine Similarity or Dot Product, often accelerated via algorithms like HNSW) to find the top-$K$ chunks most mathematically similar to the query vector.
4.  **Prompt Augmentation:** The retrieved text chunks are injected into a prompt template alongside the user's original query.
    *   *Example Prompt:* "Use the following context to answer the question. Context: [Retrieved Chunk 1], [Retrieved Chunk 2]. Question: [User Query]."
5.  **Generation:** The LLM reads the augmented prompt and generates an answer grounded *only* in the provided context.

## 3. Naive RAG vs. Advanced RAG

"Naive RAG" (the basic pipeline above) often fails in production due to low retrieval precision or missing context. Advanced RAG introduces techniques to fix these failure modes.

### Key Advanced RAG Techniques:
*   **Query Transformation/Expansion:** The user's raw query might be poorly phrased. An LLM can re-write the query (e.g., "HyDE" - Hypothetical Document Embeddings) before searching the vector DB.
*   **Hybrid Search:** Relying solely on vector embeddings can miss keyword-specific searches (e.g., searching for a specific ID like "AB-1234"). Hybrid search combines Vector Search (semantic) with traditional Keyword Search (BM25 / TF-IDF) using a tool like Elasticsearch, merging the results.
*   **Reranking (Crucial!):** Vector DBs are fast but imprecise. You retrieve a large number of documents (e.g., Top 20) quickly, then pass them through a specialized Cross-Encoder model (a "Reranker" like Cohere Rerank or BGE-Reranker) that accurately scores the relevance of each document against the query. The top 3-5 are then passed to the LLM.

## Interview Focus
Interviewers will look for your understanding of RAG failure modes. If the LLM gives a bad answer, is it a *Retrieval Failure* (the right document wasn't found) or a *Generation Failure* (the document was there, but the LLM ignored it or hallucinated)? Fixing RAG usually means fixing the retrieval step (better chunking, reranking, hybrid search).