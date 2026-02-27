# GraphRAG (Knowledge Graphs + RAG)

Standard RAG (Vector Search) is excellent at finding specific needles in a haystack. It is terrible at "connecting the dots" across an entire dataset. **GraphRAG** solves this by structuring data as a network of relationships.

---

## 1. The Limitation of Standard RAG
*   **The Query:** "What are all the subsidiary companies owned by CompanyX, and what are their primary logistics focuses?"
*   **The Vector Problem:** This information is likely scattered across 50 different PDF documents. Standard RAG will retrieve the top 5 most semantically similar paragraphs, missing 45 of the subsidiaries entirely. It cannot "reason globally" over the dataset.

## 2. What is a Knowledge Graph?
A database structured as nodes (entities) and edges (relationships).
*   **Node:** `CompanyX` (Type: Company)
*   **Node:** `John Doe` (Type: Person)
*   **Edge:** `[John Doe] - (IS_CEO_OF) -> [CompanyX]`

## 3. How GraphRAG Works (The Microsoft Approach)

### Phase 1: Indexing (The Expensive Part)
You do not just chunk text and embed it. You use an LLM to actively "read" your entire corpus and extract a graph.
1.  **Entity Extraction:** The LLM reads a chunk and identifies all People, Places, Organizations, and Concepts.
2.  **Relationship Extraction:** The LLM identifies how they connect and creates triples (Subject -> Predicate -> Object).
3.  **Clustering:** Graph algorithms (like Leiden) group related nodes into "communities" (e.g., all nodes related to "European Trucking Regulations").
4.  **Summarization:** An LLM generates a text summary for every single community cluster.

### Phase 2: Global Search (The Payoff)
When a user asks a high-level, "global" question ("What are the main themes in CompanyX's Q3 reports?"):
1.  Instead of searching raw text vectors, the system retrieves the pre-computed **Community Summaries**.
2.  The LLM reads all the community summaries and synthesizes a comprehensive, globally-aware answer.

## 4. Standard GraphRAG (Cypher/Graph DBs)
An alternative approach using databases like Neo4j.
*   **The Flow:** 
    1. User asks a question.
    2. An LLM acts as a translator, turning natural language into a Graph Database Query Language (like Cypher).
    3. The system executes the Cypher query against Neo4j to pull exact relational data (e.g., traversing 3 hops: User -> Bought -> Item -> Manufactured By -> Company).
    4. The data is passed back to the LLM to format as natural text.

## 5. Tradeoffs (GraphRAG vs. Vector RAG)

| Feature | Vector RAG | GraphRAG |
| :--- | :--- | :--- |
| **Best For** | "What does document X say about Y?" | "How does concept A connect to concept B across all documents?" |
| **Ingestion Cost** | Extremely Cheap (Just embedding math). | **Extremely Expensive** (Requires heavily prompting an LLM to read and extract data from every single document). |
| **Query Speed** | Very Fast. | Can be slower (requires traversing graphs or reading multiple summaries). |
| **Setup Complexity** | Low (Pinecone + LangChain). | High (Requires ontology design, graph DB management). |

## Interview Takeaway
"I would not default to GraphRAG for a standard Q&A chatbot due to the massive token cost required to build the index. However, for an intelligence/investigation tool—such as a system designed to detect complex fraud rings across logistics sub-contractors—the ability to traverse relational hops makes GraphRAG mandatory."