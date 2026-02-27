# Case Study: Customer Support Chatbot via Advanced RAG

**Company:** Logistics & Shipping Company
**Scenario:** A B2B logistics company's customer support center handles 50,000 queries daily regarding shipping policies, customs regulations, hazardous material rules, and specific tracking statuses. Wait times are averaging 45 minutes. You are tasked with designing an AI chatbot to automate tier-1 support queries using RAG.

---

## 1. Problem Framing & Constraints
*   **Goal:** Deflect 30% of tier-1 support tickets with high accuracy.
*   **Data Sources:**
    *   Internal Knowledge Base (Confluence) for general policies.
    *   Public Customs PDFs (frequently updated, complex tables).
    *   User-specific data (Tracking APIs via SQL/REST).
*   **Constraints:**
    *   **Zero Hallucination Tolerance:** Giving wrong customs advice can result in massive legal fines and seized cargo.
    *   **Access Control:** B2B customers must not see internal proprietary logistics documents or other customers' data.

## 2. Architecture Design (Step-by-Step)

### Step 1: Data Ingestion Pipeline (Offline)
1.  **Parsing:** Standard text is parsed using typical loaders. For the complex Customs PDFs, I will utilize a vision-language model (VLM) or specialized OCR pipeline (like Unstructured.io) to properly extract tables into Markdown format, preserving structural context.
2.  **Chunking:** I will use **Parent-Child chunking**. The documents will be split into larger 1000-token Parent chunks (for LLM context), and smaller 250-token Child chunks (for embedding/retrieval).
3.  **Embedding & Metadata:** I will use an open-source model like `BGE-Large` to avoid sending internal documents to external APIs. *Crucially*, I will tag every chunk with metadata: `access_level` (public, internal), `country_code`, and `document_topic`.
4.  **Storage:** Store in a Vector DB like Milvus or Pinecone.

### Step 2: The Agentic Retrieval Pipeline (Online)
Standard RAG isn't enough here because queries require different actions (policy lookup vs. tracking lookup). I will use a routing agent (e.g., using LangChain/LlamaIndex router).

1.  **Query Routing:** The user asks: "Where is my shipment 12345, and what are the duties for shipping batteries to Germany?" A fast LLM (Router) decides:
    *   Action A: Call the Tracking API for ID 12345.
    *   Action B: Perform Vector Search for "batteries duties Germany".
2.  **Vector Retrieval:**
    *   *Pre-filtering:* The Vector DB filters out any chunks where `access_level == internal` (since this is customer-facing).
    *   *Hybrid Search:* Perform dense vector search combined with BM25 keyword search.
    *   *Reranking:* Pass the top 15 results through a Cross-Encoder (`Cohere Rerank`) to get the top 3 most relevant chunks.
3.  **Context Assembly:** Retrieve the large *Parent* chunks associated with the top 3 *Child* chunks. Combine this with the API JSON response from the tracking query.

### Step 3: Generation & Guardrails
1.  **Prompting:**
    ```text
    System: You are an official logistics support agent. You must ONLY use the provided context. If the answer is not present, say "I will transfer you to a human agent."
    Context: [API Data], [RAG Parent Chunks]
    User: [Original Query]
    ```
2.  **Output Guardrails:** Before returning the response to the user, a secondary, lightweight LLM (or programmatic regex checker) runs to ensure no restricted keywords or confident hallucinations are present.

## 3. Dealing with Edge Cases & Risks
*   **Risk:** The model hallucinates a customs fee.
    *   **Mitigation:** The strict system prompt + the instruction to transfer to a human if uncertain.
*   **Risk:** Documents update, but the DB has old embeddings.
    *   **Mitigation:** The ingestion pipeline requires an Airflow DAG that listens to Confluence webhooks. When a document is updated, its old chunks are deleted by document ID, and the new version is chunked and embedded instantly.
*   **Risk:** Latency is too high for chat.
    *   **Mitigation:** Stream the LLM output (Server-Sent Events) to the UI so the user perceives immediate response while the rest of the text generates. Implement Semantic Caching (Redis) for FAQ-style queries to bypass the LLM entirely.