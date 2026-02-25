# The BERT Family (Encoder-Only Models)

BERT (Bidirectional Encoder Representations from Transformers), released by Google in 2018, was a watershed moment in NLP. It is the archetype of the **Encoder-Only** Transformer architecture.

---

## 1. Core Architecture
BERT uses the **Encoder** stack from the original Transformer.
*   **Key Feature: Bidirectional Attention.** Unlike GPT (which generates text and can only look backward), BERT looks at the entire sentence at once. When embedding the word "bank" in the sentence "I sat on the river bank", BERT pays attention to both "river" (before) and the period (after).

## 2. How BERT is Pre-trained
BERT is not trained to predict the next word. It is trained using two specific self-supervised tasks.

### Task 1: Masked Language Modeling (MLM)
*   **The Process:** 15% of the words in the training text are randomly masked (replaced with a `[MASK]` token). The model must look at the surrounding context (both left and right) and predict what the hidden word is.
*   **The Result:** The model is forced to develop an incredibly deep, nuanced understanding of language syntax and semantics.

### Task 2: Next Sentence Prediction (NSP)
*   **The Process:** The model is given two sentences (A and B). 50% of the time, B is the actual next sentence in the document. 50% of the time, B is a random sentence. The model must output a binary Yes/No: "Does B follow A?"
*   **The Result:** The model learns document-level coherence and relationships between sentences (crucial for tasks like Question Answering).

## 3. How to use BERT (Fine-Tuning)
BERT is almost never used "out of the box." It is a foundational base model that you fine-tune for specific downstream tasks.

*   **The `[CLS]` Token:** BERT explicitly adds a special `[CLS]` (Classification) token to the very beginning of every input sequence.
*   **Classification Task (e.g., Sentiment Analysis):** You pass a review into BERT. You take the final output vector corresponding *only* to the `[CLS]` token (which acts as an aggregate summary of the whole sentence). You attach a simple Linear Layer on top of that vector and train it to output Positive/Negative.
*   **Token-Level Task (e.g., Named Entity Recognition):** You pass a sentence in. You attach a Linear Layer to the output vector of *every single word*, predicting if each word is a Person, Location, or Organization.

## 4. Famous BERT Variants

*   **RoBERTa (Facebook):** "Robustly Optimized BERT." Researchers found BERT was drastically under-trained. RoBERTa uses the exact same architecture but trains on 10x more data, trains longer, and drops the NSP task (which was proven useless). It severely outperforms BERT.
*   **DistilBERT:** A smaller, faster, cheaper version of BERT created using **Knowledge Distillation**. It retains 97% of BERT's performance while being 40% smaller and 60% faster, making it the go-to for production edge deployments.
*   **Sentence-BERT (SBERT):** Standard BERT is terrible at generating a single embedding vector for a whole sentence. SBERT uses a "Siamese" network structure to fine-tune BERT specifically for semantic search and clustering, creating the foundation for modern RAG embedding models.

## Interview Summary
*   **Use BERT/RoBERTa when:** You need to *understand* text. (Classification, Sentiment, Entity Extraction, Generating Embeddings for RAG).
*   **Do NOT use BERT when:** You need to *generate* text. (Chatbots, Summarization, Code Generation). You cannot chat with BERT.