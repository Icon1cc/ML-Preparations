# Tokenization Strategies

Neural networks cannot read text. They only understand numbers. Tokenization is the critical first step in any NLP pipeline: breaking raw text down into small, distinct chunks (tokens) and mapping them to integer IDs.

---

## 1. The Naive Approaches

### A. Word-Level Tokenization
*   Split text by spaces. `["The", "dog", "barked"]`
*   **The Problem:** The English language has an infinite number of words (e.g., combining words, typos, URLs). To cover everything, your vocabulary size would be millions of words. This requires a massive embedding matrix, wasting RAM. Also, it completely fails on "Out of Vocabulary" (OOV) words. If it sees a typo "dogg," it assigns it an `<UNK>` token and loses all meaning.

### B. Character-Level Tokenization
*   Split text by characters. `["T", "h", "e", " ", "d", "o", "g"]`
*   **The Problem:** The vocabulary is tiny (256 ASCII characters), but sequences become incredibly long. The word "internationalization" takes 20 tokens. Because Transformer attention is $O(N^2)$ based on sequence length, this destroys computational efficiency. Furthermore, individual characters carry no semantic meaning, making it harder for the model to learn.

## 2. The Modern Standard: Sub-word Tokenization
The Goldilocks solution. Break rare words into smaller sub-words, but keep common words whole. 
*   `"playing"` $
ightarrow$ `["play", "ing"]`
*   `"unbelievably"` $
ightarrow$ `["un", "believ", "ably"]`
*   This keeps the vocabulary size manageable (usually 30k to 100k) while completely eliminating the `<UNK>` out-of-vocabulary problem.

## 3. Sub-word Algorithms

### A. Byte-Pair Encoding (BPE)
Used by GPT, RoBERTa, and Llama.
*   **Training Phase:**
    1. Start with a vocabulary of individual characters.
    2. Scan the training corpus and find the most frequently occurring *pair* of tokens (e.g., 't' and 'h').
    3. Merge them to create a new token ('th') and add it to the vocabulary.
    4. Repeat this merging process $N$ times until you hit your target vocabulary size (e.g., 50,000).
*   **Inference:** When it sees a new word, it breaks it down into the largest sub-words it learned during training.

### B. WordPiece
Used by BERT.
*   Very similar to BPE, but instead of choosing the most *frequent* pair to merge, it chooses the pair that maximizes the likelihood of the training data given the language model.
*   It uses a `##` prefix to denote sub-words that are part of a larger word. `"playing" -> ["play", "##ing"]`.

### C. Unigram / SentencePiece
*   Instead of starting small and merging up (BPE), Unigram starts with a massive vocabulary of all possible subwords and iteratively *removes* the least useful ones until it hits the target size.
*   **SentencePiece:** An implementation that treats the space character as just another character (often denoted as `_`). This means you don't need language-specific pre-tokenizers (like splitting by spaces), making it excellent for multi-lingual models (like T5) where languages like Chinese or Japanese don't use spaces.

## 4. Why Tokenization Matters for Engineers
Interviewers will test if you understand the edge cases caused by tokenization.
*   **Code Generation:** Early LLMs were terrible at Python indentation because they tokenized spaces inefficiently (e.g., 4 spaces = 4 separate tokens). Modern models (Llama 3, GPT-4) have special single tokens for `	` or `    ` (four spaces), drastically increasing code context windows.
*   **Math and Numbers:** BPE often splits numbers illogically. `142857` might tokenize as `[14, 285, 7]`. This makes arithmetic nearly impossible for the LLM. Modern tokenizers force single-digit tokenization for numbers so the model can learn place-value math.
*   **Cost:** API providers charge *per token*, not per word. A language with poor tokenizer representation (like Hindi on a GPT-2 tokenizer) might require 3 tokens per word, making API costs 3x more expensive for the same amount of information compared to English.