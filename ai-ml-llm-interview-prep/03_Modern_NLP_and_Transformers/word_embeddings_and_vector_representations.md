# Word Embeddings and Vector Representations

Before deep Transformers existed, NLP relied on Word Embeddings. Understanding the evolution from TF-IDF to Word2Vec to BERT embeddings is a classic interview topic.

---

## 1. The Pre-Embedding Era: Bag of Words & TF-IDF
Early NLP treated words as isolated, discrete symbols.

*   **One-Hot Encoding:** Every word gets an index. "Cat" = `[0, 1, 0, 0]`. "Dog" = `[0, 0, 1, 0]`.
    *   *Problem:* The dot product of these vectors is 0. The computer thinks "Cat" and "Dog" are as mathematically different as "Cat" and "Spaceship."
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weighs a word heavily if it appears frequently in a specific document, but penalizes it if it appears frequently across *all* documents (like "the" or "and"). Still results in sparse, orthogonal vectors with no semantic meaning.

## 2. The Breakthrough: Word2Vec (Static Embeddings)
Developed by Google in 2013, Word2Vec popularized the concept of Dense Vector Embeddings.
*   **The Core Idea (Distributional Hypothesis):** "You shall know a word by the company it keeps." If "Cat" and "Dog" frequently appear next to words like "pet", "feed", and "bark", their mathematical vectors should be pushed closer together in space.
*   **How it works:** It uses a shallow, 1-hidden-layer neural network.
    *   *CBOW (Continuous Bag of Words):* Given the context words, predict the middle target word.
    *   *Skip-gram:* Given a target word, predict the surrounding context words.
*   **The Result:** A static lookup table. Every word is assigned a dense vector (e.g., 300 dimensions). 
*   **The Magic:** Vector math actually works. $Vector(King) - Vector(Man) + Vector(Woman) \approx Vector(Queen)$.

### Other Static Embeddings
*   **GloVe (Global Vectors):** Uses matrix factorization based on a global word co-occurrence matrix rather than a local sliding window.
*   **FastText (Meta):** A massive improvement. Instead of learning vectors for whole words, it learns vectors for sub-word *character n-grams* (e.g., "apple" = "app" + "ppl" + "ple"). This allows FastText to generate a mathematical vector for typos or words it has never seen before, which Word2Vec cannot do.

## 3. The Modern Era: Contextual Embeddings (BERT)
The fatal flaw of Word2Vec is that it is **Static**. The word "bank" has exactly one vector.
*   "I sat on the river **bank**."
*   "I deposited money in the **bank**."
Word2Vec gives "bank" the exact same representation in both sentences, blurring the semantic meaning.

**Contextual Embeddings (via Transformers like BERT):**
*   BERT does not have a static lookup table for the final meaning. The word "bank" passes through the Self-Attention mechanism, looks at the surrounding words ("river" vs "money"), and generates a dynamically calculated, unique vector *on the fly* for that specific sentence.

## 4. Sentence Embeddings and RAG
While BERT generates great vectors for *words*, if you average all the word vectors together to get a *document* vector, the result is surprisingly poor for semantic search.
*   **Sentence-BERT (SBERT / BGE Models):** Models specifically fine-tuned using a "Siamese" network structure. They are trained specifically so that the Cosine Similarity between two entire sentences accurately reflects their semantic overlap. This is the technology that powers the Vector Databases in modern RAG systems. (See `embeddings_for_rag.md`).