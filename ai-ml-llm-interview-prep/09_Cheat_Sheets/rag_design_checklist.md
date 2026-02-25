# RAG System Design Checklist

A quick reference checklist for designing production-ready Retrieval-Augmented Generation (RAG) systems in an interview setting.

---

## 1. Data Ingestion (Offline Pipeline)
*   [ ] **Data Sources:** What are the sources? (PDFs, Confluence, DBs, Slack). How often do they update? (Batch vs. Streaming).
*   [ ] **Parsing Strategy:** How are you extracting text? (e.g., standard text extraction for PDFs vs. specialized OCR/Vision models for charts/tables).
*   [ ] **Chunking Strategy:**
    *   Fixed size (e.g., 500 tokens with 50 overlap)?
    *   Semantic/Recursive splitting?
    *   Parent-Child (Small-to-big) chunking for better retrieval vs. generation balance?
*   [ ] **Metadata:** Are you attaching crucial metadata to chunks? (e.g., `date_published`, `author`, `document_type`, `access_level`). *Crucial for hybrid search and security filtering.*

## 2. Embedding & Storage
*   [ ] **Embedding Model:** Open-source (BGE, E5) vs. Commercial API (OpenAI).
    *   Are you dealing with specialized jargon? (Might need fine-tuning the embedding model).
*   [ ] **Vector Database:**
    *   Managed/Cloud (Pinecone, Weaviate Cloud).
    *   Self-Hosted/Open Source (Milvus, Qdrant).
    *   Existing Relational DB extension (pgvector for PostgreSQL).
*   [ ] **Index Type:** HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search vs. Flat (exact, but slow).

## 3. Retrieval (Online Pipeline - The Most Important Part)
*   [ ] **Query Transformation:**
    *   Query Rewriting (using a fast LLM to rephrase the user's sloppy query).
    *   HyDE (Hypothetical Document Embeddings - having the LLM guess the answer, and embedding the guess to search).
*   [ ] **Search Strategy:**
    *   Pure Dense Vector Search (good for semantic meaning).
    *   Hybrid Search (Vector + BM25/Keyword search - *Mandatory for production* to handle exact ID/name lookups).
*   [ ] **Metadata Filtering:** Pre-filtering the vector search space using metadata (e.g., "Only search documents where `year == 2024`").
*   [ ] **Reranking:** *Always include this in a system design.* Retrieve top 20 documents quickly, then use a Cross-Encoder (like Cohere Rerank) to accurately re-order them and pass the top 3-5 to the LLM.

## 4. Generation & Prompting
*   [ ] **LLM Selection:** Fast/Cheap (Llama 3 8B, GPT-3.5) vs. Smart/Expensive (GPT-4, Claude 3.5 Sonnet).
*   [ ] **Prompt Construction:**
    *   System Prompt defining persona and guardrails.
    *   Clear delimiters for the injected context.
    *   Explicit instruction: "If the answer is not in the context, say 'I do not know'."
*   [ ] **Streaming:** Are you streaming tokens back to the user to reduce perceived latency (TTFT - Time To First Token)?

## 5. MLOps & Evaluation (The "Seniority" Check)
*   [ ] **RAG Evaluation Framework:** How do you know it works? (Use frameworks like RAGAS or Trulens).
    *   *Context Relevance:* Did we retrieve the right chunks?
    *   *Faithfulness:* Did the LLM hallucinate, or stick to the context?
    *   *Answer Relevance:* Did the final answer actually address the user's question?
*   [ ] **Feedback Loop:** Implementing thumbs up/down UI to capture implicit user feedback for future fine-tuning.
*   [ ] **Caching:** Semantic caching (e.g., Redis) to serve identical or highly similar queries instantly without hitting the Vector DB or LLM.