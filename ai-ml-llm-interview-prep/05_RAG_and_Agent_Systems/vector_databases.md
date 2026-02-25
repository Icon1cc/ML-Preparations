# Vector Databases for AI Systems

Vector databases are specialized storage engines designed to store and query high-dimensional vector embeddings. They are the "long-term memory" of RAG and Agent systems.

---

## 1. Why not just use SQL?
Traditional databases (PostgreSQL, MySQL) are built for exact matches: `SELECT * WHERE id = 123`. 
Vector databases are built for **Approximate Nearest Neighbor (ANN)** search: "Find the top 5 documents whose mathematical meaning is most similar to this query." 
*Trying to do vector math in a standard SQL index is $O(N)$, which is too slow once you have millions of documents.*

## 2. Core Indexing Algorithms

### A. HNSW (Hierarchical Navigable Small World)
The gold standard for production.
*   **Concept:** It builds a multi-layered graph. The top layer has few nodes and long-distance connections (like an express train). The bottom layer has all nodes and short-distance connections (like local streets).
*   **Search:** It starts at the top, hops quickly to the general area of the query, then moves down a layer to refine the search.
*   **Tradeoff:** Extremely fast and accurate, but uses a lot of RAM because the entire graph must be kept in memory.

### B. IVF (Inverted File Index)
*   **Concept:** Uses K-Means clustering to divide the vector space into $N$ clusters (voronoi cells).
*   **Search:** The query first finds the nearest cluster center, then only searches the vectors within that cluster.
*   **Tradeoff:** Much less RAM than HNSW, but slightly slower and less accurate.

## 3. Key Features to Look For

*   **Hybrid Search:** The ability to combine Vector Search (semantic) with Keyword Search (BM25) in a single query. Essential for finding specific product IDs or names.
*   **Metadata Filtering:** The ability to apply hard filters (`WHERE user_id = 42`) *before* or *during* the vector search.
*   **Multi-Tenancy:** Keeping different customers' data logically separated within the same database.

## 4. Market Landscape

| Category | Examples | Best For... |
| :--- | :--- | :--- |
| **Native Vector DB** | Pinecone, Milvus, Weaviate, Qdrant | Large-scale, high-performance RAG; dedicated AI teams. |
| **Extensions** | `pgvector` (Postgres), Redis VL | Teams who already use Postgres/Redis and want to avoid adding a new piece of infrastructure. |
| **Search Engines** | Elasticsearch, OpenSearch | Systems where Keyword Search is just as important as Vector Search. |

## 5. Interview Tip: Scaling and Cost
If asked about scaling a Vector DB:
1.  **Memory Management:** Discuss **Product Quantization (PQ)**. It's a compression technique that reduces the size of vectors (e.g., from 32-bit floats to 8-bit integers). This allows you to fit 4x more vectors into the same RAM with only a tiny drop in accuracy.
2.  **Indexing vs. Querying:** Mention that adding new documents to an HNSW index is slow and compute-heavy. For high-write workloads, you might need a separate ingestion service.
3.  **Matryoshka Embeddings:** Mention using models that allow for vector truncation to save costs. (See Embeddings for RAG file).