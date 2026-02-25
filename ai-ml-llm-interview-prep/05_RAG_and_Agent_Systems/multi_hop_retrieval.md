# Multi-Hop Retrieval

Multi-hop queries are questions that cannot be answered by retrieving a single document. The system must retrieve Document A, read it, and realize it needs to search for Document B to finish answering the question.

---

## 1. The Challenge
*   **Query:** "Who is the CEO of the company that acquired Acme Logistics?"
*   **Standard RAG Failure:** A single vector search might find documents about "Acme Logistics" or documents about "CEOs," but it's unlikely to find a single paragraph containing both facts.

## 2. Approach 1: ReAct Agent (Sequential Hopping)
Using an Agent framework (like LangChain or AutoGen) equipped with a search tool.
1.  **Thought:** I need to find out who acquired Acme Logistics first.
2.  **Action:** `VectorSearch("Who acquired Acme Logistics?")`
3.  **Observation:** "DHL acquired Acme Logistics in 2023."
4.  **Thought:** Now I need to find the CEO of DHL.
5.  **Action:** `VectorSearch("Who is the CEO of DHL?")`
6.  **Observation:** "Tobias Meyer."
7.  **Final Answer:** "The CEO is Tobias Meyer."

*   **Pros:** Highly logical, flexible, can recover from bad searches.
*   **Cons:** Very slow. It requires multiple sequential calls to the generative LLM (which is the main latency bottleneck).

## 3. Approach 2: Query Decomposition (Parallel Hopping)
Instead of waiting for the results of the first hop, an LLM breaks the complex query down into parallel sub-queries.
*   **Sub-Query 1:** `VectorSearch("Acme Logistics acquisition details")`
*   **Sub-Query 2:** `VectorSearch("List of logistics company CEOs")`
*   **Mechanism:** Both searches are fired simultaneously. All results are dumped into the context window, and the LLM figures out the connections.
*   **Pros:** Much faster than sequential ReAct loops.
*   **Cons:** Fails if the second query *depends* on the exact name found in the first query.

## 4. Approach 3: GraphRAG (Pre-computed Hops)
(See `graph_rag.md`). 
By structuring the data as a Knowledge Graph, the hops are already pre-calculated as edges in the database.
*   The system executes a graph query: `(Acme Logistics)<-[:ACQUIRED]-(Company)-[:HAS_CEO]->(Person)`.
*   This traverses the database instantly without requiring multiple expensive LLM reasoning steps.

## 5. Approach 4: Step-Back Prompting
A prompting technique that asks the LLM to abstract the question before diving into details.
*   **Original:** "Did the volume of package returns increase more in Berlin or Munich between Q1 and Q2?"
*   **Step-Back Abstraction:** The LLM generates a broader question: "What are the package return volumes for Berlin and Munich in Q1 and Q2?"
*   The system retrieves data for the broader question, which usually pulls in all the necessary context to answer the highly specific original question.

## Interview Strategy
"Handling multi-hop queries is a tradeoff between latency and accuracy. If low latency is required, I prefer to rely on pre-computing relationships via a **Knowledge Graph**. If the domain is too broad for a graph, I would implement an **Agentic ReAct loop**, but I would monitor token usage closely as multi-step reasoning can become expensive."