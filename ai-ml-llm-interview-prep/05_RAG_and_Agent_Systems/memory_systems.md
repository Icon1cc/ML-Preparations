# Memory Systems for LLM Agents

LLMs are inherently stateless. Every API call starts with a blank slate. Building conversational or long-running agents requires engineering external memory systems.

---

## 1. Short-Term Memory (Context Window Management)
This is the immediate conversation history.

### A. Full Window (Naive)
*   Simply appending every user message and assistant reply into an array and passing the whole thing to the LLM.
*   **Failure Mode:** You quickly hit the context limit (e.g., 8k tokens). Even if you use a 128k model, latency becomes unbearable, and API costs skyrocket because you pay for the entire history on *every single turn*.

### B. Sliding Window (FIFO)
*   Only keep the last $N$ turns (e.g., the last 5 messages).
*   **Pros:** Solves token limits and cost.
*   **Cons:** The bot abruptly "forgets" your name if you mentioned it 6 messages ago.

### C. Summary Memory
*   When the context window reaches a threshold (e.g., 2000 tokens), a background process calls a smaller, cheaper LLM to summarize the entire conversation history into a single paragraph.
*   The history array is cleared, and replaced with `System: [Previous Conversation Summary]`.
*   **Pros:** Retains long-term context cheaply.
*   **Cons:** Nuance and exact quotes are lost in summarization.

## 2. Long-Term Memory (Vector/Graph Databases)
How does an agent remember a user it hasn't spoken to in a month?

### A. Vector-Based Episodic Memory
*   Every time the user says something meaningful ("I am allergic to peanuts", "My default shipping address is Berlin"), the system embeds that sentence and stores it in a Vector Database tied to the `user_id`.
*   **Retrieval:** When the user asks a new question, the system does a vector search against the user's specific memory DB and injects the results into the prompt.
*   *Frameworks:* Zep, Mem0, or custom LangChain implementations.

### B. Graph-Based Memory (Knowledge Graphs)
*   Instead of raw text chunks, the system explicitly extracts facts and updates a graph.
*   `User(ID:1) -[PREFERS]-> Carrier(DHL)`
*   This prevents the "contradiction problem" found in Vector Memory (where old facts and new facts are both retrieved, confusing the LLM). In a graph, an update simply overwrites the edge.

## 3. Agentic Memory Architectures (e.g., MemGPT)
Advanced agents manage their own memory actively, rather than passively relying on the backend to inject it.

*   The LLM is given specific Tools for memory management:
    *   `core_memory_append(text)`: To save a crucial fact (like the user's name) to an always-included system prompt block.
    *   `archive_search(query)`: To actively search past conversations.
*   The Agent uses its reasoning loop to decide *when* a piece of information is important enough to save for the future.

## Interview Strategy
"For a production customer support bot, I would use a hybrid memory system. A **Sliding Window** for the immediate back-and-forth chat to keep latency low. However, I would also implement an asynchronous background worker that extracts strict User Preferences (like default addresses or VIP status) and writes them to a standard **SQL Database**, which is injected into the system prompt at the start of every new session."