# RAG Debugging and Failure Modes

When a RAG system gives the user a bad answer, you must systematically isolate the point of failure. It is rarely the fault of the generative LLM. 

---

## The 7 Common Failure Modes of RAG

### Failure 1: Missing Content
*   **Symptom:** The LLM answers "I don't know" or hallucinates because the actual answer was never in the database to begin with.
*   **Fix:** Audit your data ingestion pipeline. Was the PDF parsed correctly? Did the OCR fail on a table? Are your database sync jobs failing?

### Failure 2: Missed the Top K (Retrieval Failure)
*   **Symptom:** The answer is in the database, but it was ranked #25 in the vector search, and you only passed the Top 5 to the LLM.
*   **Fix:**
    *   Increase your initial `Top K` search parameter.
    *   Implement **Hybrid Search** (Vector + Keyword) to ensure exact name matches aren't missed.
    *   Adjust chunking strategy (if chunks are too large, their vector embeddings become diluted).

### Failure 3: Not in Context (Reranking Failure)
*   **Symptom:** You retrieved the right document at rank #15, but your system only passes the Top 3 to the LLM.
*   **Fix:** Implement a **Cross-Encoder Reranker**. Retrieve Top 50, rerank them accurately, and pass the true Top 3 to the LLM.

### Failure 4: Not Extracted (LLM Blindness)
*   **Symptom:** The correct paragraph was successfully retrieved and is literally sitting inside the prompt, but the LLM failed to use it and gave a wrong answer.
*   **Fix:**
    *   *The "Lost in the Middle" problem:* If you passed 20 documents, the LLM might have ignored the middle ones. Reduce context size via reranking.
    *   *Prompt Engineering:* Highlight the context clearly using XML tags (`<context>...</context>`). Add strong directives: "Analyze the context thoroughly before answering."
    *   *Model Upgrade:* Switch from a small 8B model to a more capable reasoning model (GPT-4o, Claude 3.5 Sonnet).

### Failure 5: Wrong Format
*   **Symptom:** The LLM extracted the right answer, but the downstream application crashed because it expected JSON and got Markdown.
*   **Fix:** Use Structured Generation (JSON Mode) or frameworks like Instructor/Pydantic to enforce strict output schemas.

### Failure 6: Incomplete Answer
*   **Symptom:** The user asks a multi-part question ("What are the shipping rules for batteries AND chemicals?"). The LLM only answers the battery part.
*   **Fix:** Implement **Query Decomposition**. A routing LLM breaks the user's prompt into two distinct queries, runs two separate vector searches, concatenates all the results, and synthesizes a complete answer.

### Failure 7: Outdated Information
*   **Symptom:** The LLM confidently quotes the 2023 shipping policy instead of the 2024 policy. Both are in the database.
*   **Fix:** 
    *   *Ingestion Fix:* When a document is updated, you must explicitly delete its old vector chunks using a unique Document ID.
    *   *Metadata Filtering:* Always attach a `timestamp` or `version` to vector metadata. Instruct the system to sort or filter by the newest date if multiple versions are found.

## Interview Strategy: The Debugging Flow
If an interviewer gives you a scenario where a RAG bot lied to a customer:
1.  **Trace the Logs:** "I would look at the telemetry for that specific query ID."
2.  **Check Retrieval:** "I would inspect the exact text chunks that were returned by the Vector DB. If the answer wasn't in those chunks, I know it's an embedding/search issue."
3.  **Check Generation:** "If the chunks *did* contain the answer, I know it's a prompt engineering, context window size, or model capability issue."