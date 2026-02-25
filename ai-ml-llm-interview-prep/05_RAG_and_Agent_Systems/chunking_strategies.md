# Document Chunking Strategies for RAG

Chunking is the process of breaking down large documents into smaller segments before converting them into embeddings. It is a foundational, yet highly complex, step in building an effective RAG system. Poor chunking destroys retrieval accuracy.

---

## Why is Chunking Necessary?
1.  **Embedding Model Limits:** Most embedding models (like BERT variants) have a strict sequence limit (e.g., 512 tokens). If you pass a 10-page document, it will silently truncate the text, losing data.
2.  **LLM Context Windows:** While modern LLMs have large context windows (128k+), passing massive documents is expensive, slow, and suffers from the "Lost in the Middle" phenomenon (LLMs ignore information in the middle of long contexts).
3.  **Retrieval Precision:** If an entire chapter is one chunk, its embedding vector becomes a "diluted" average of many topics. When a user asks a specific question, the dense vector of a highly specific small paragraph will match much better than the diluted vector of a massive chapter.

## Common Chunking Strategies

### 1. Fixed-Size Chunking (Character or Token based)
The simplest method. You divide text into chunks of a set number of characters or tokens (e.g., 500 tokens).
*   **Overlap:** You must include an overlap (e.g., 50 tokens) between chunks. Otherwise, a sentence might be split directly in half across two chunks, destroying its semantic meaning.
*   **Pros:** Easy to implement, fast, predictable size.
*   **Cons:** "Semantic tearing." It ignores the structure of the document. It might start a chunk in the middle of a paragraph and end it in the middle of a list.

### 2. Recursive Character Text Splitting
The standard default in frameworks like LangChain. It tries to split recursively using a list of separators (e.g., `["

", "
", " ", ""]`).
*   **Mechanism:** It first tries to split by double newlines (paragraphs). If a paragraph is still larger than the target chunk size, it splits that paragraph by single newlines (sentences). If a sentence is still too large, it splits by spaces.
*   **Pros:** Keeps paragraphs and sentences together as much as possible, respecting basic semantic boundaries better than fixed-size.
*   **Cons:** Still blind to actual document formatting (headers, tables).

### 3. Document-Specific / Structural Chunking (Markdown/HTML)
Splitting based on the logical structure of the document.
*   **Markdown Chunking:** Splitting by headers (`#`, `##`, `###`).
*   **Code Chunking:** Splitting by functions or classes (using AST parsers).
*   **Pros:** Highly semantic. A chunk represents exactly one logical section (e.g., a specific API endpoint documentation).
*   **Cons:** Requires clean, structured input data. Fails completely on raw, unstructured OCR text.

### 4. Semantic Chunking
An advanced approach where chunks are created based on shifts in meaning, rather than physical length.
*   **Mechanism:**
    1. Split text into sentences.
    2. Embed each sentence individually.
    3. Calculate the cosine distance between consecutive sentences.
    4. If the distance exceeds a certain threshold, it indicates a "topic shift," and a chunk boundary is placed there.
*   **Pros:** Preserves semantic integrity perfectly.
*   **Cons:** Computationally expensive (requires running an embedding model on every single sentence before the actual chunk embedding).

## Advanced Chunking Patterns for Production

### Parent-Child Chunking (Small-to-Big Retrieval)
A highly effective production strategy to balance precise retrieval with broad LLM context.
1.  **Ingestion:** Split documents into Large "Parent" chunks (e.g., 1000 tokens). Split those Parents into Small "Child" chunks (e.g., 200 tokens). Embed and store *only* the Child chunks in the vector DB, but link them to their Parent ID.
2.  **Retrieval:** When a query comes in, perform vector search against the small Child chunks (highly precise retrieval).
3.  **Generation:** Once the relevant Child chunk is found, retrieve its full Parent chunk and pass the *Parent* chunk to the LLM.
*   **Why it works:** Small chunks are mathematically better for search (less noise). Large chunks are better for generation (give the LLM the full surrounding context to formulate a good answer).