# Word Embeddings and Vector Representations

## What Are Embeddings? (The Basics)

**Plain English Explanation:**

Imagine you're trying to teach a computer about words. Computers only understand numbers, not words. Embeddings convert words into lists of numbers (vectors) in a way that **captures meaning**.

**Example:**
```
"king"    → [0.2, 0.8, 0.1, 0.9, ...]   (300 numbers)
"queen"   → [0.3, 0.7, 0.2, 0.8, ...]   (300 numbers)
"cat"     → [0.1, 0.1, 0.7, 0.2, ...]   (300 numbers)
```

The magic: Words with similar meanings have similar number patterns!
- "king" and "queen" have similar vectors (both royalty)
- "king" and "cat" have different vectors (unrelated)

---

## Why Embeddings Matter

### The Old Way (One-Hot Encoding)

**Problem:** Treats words as completely independent

```python
Vocabulary: ["cat", "dog", "king", "queen"]

"cat"   → [1, 0, 0, 0]
"dog"   → [0, 1, 0, 0]
"king"  → [0, 0, 1, 0]
"queen" → [0, 0, 0, 1]
```

**Issues:**
- ❌ No notion of similarity (cat vs dog just as different as cat vs king)
- ❌ Vocabulary size = vector size (100k words = 100k dimensions)
- ❌ Extremely sparse (99.99% zeros)
- ❌ No semantic relationships captured

### The New Way (Word Embeddings)

**Solution:** Dense vectors that capture meaning

```python
"cat"   → [0.2, 0.1, 0.7, 0.3, ...]  (300 dims)
"dog"   → [0.3, 0.1, 0.6, 0.4, ...]  (300 dims)
"king"  → [0.1, 0.8, 0.2, 0.9, ...]  (300 dims)
"queen" → [0.2, 0.7, 0.3, 0.8, ...]  (300 dims)
```

**Advantages:**
- ✅ Similar words have similar vectors
- ✅ Fixed size regardless of vocabulary
- ✅ Dense representation (no wasted zeros)
- ✅ Captures semantic relationships

**The Famous Example:**
```
king - man + woman ≈ queen
```
The vectors actually encode gender relationships!

---

## Classical (Static) Embeddings

### 1. Bag of Words (BoW)

**Concept:** Count word occurrences, ignore order

**Example:**
```
Sentence 1: "The cat sat on the mat"
Sentence 2: "The dog sat on the log"

Vocabulary: [the, cat, dog, sat, on, mat, log]

Sentence 1 → [2, 1, 0, 1, 1, 1, 0]
Sentence 2 → [2, 0, 1, 1, 1, 0, 1]
```

**Pros:**
- ✅ Simple and fast
- ✅ Works for simple classification tasks

**Cons:**
- ❌ Ignores word order ("dog bites man" = "man bites dog")
- ❌ No semantic understanding
- ❌ Vocabulary can be huge
- ❌ No capture of word relationships

**When to use:**
- Simple text classification (spam detection)
- Document similarity with small vocabulary
- Baseline model

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())  # ['cat', 'dog', 'log', 'mat', 'on', 'sat', 'the']
print(X.toarray())  # [[1 0 0 1 1 1 2], [0 1 1 0 1 1 2]]
```

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

**Concept:** Weight words by importance

**Formula:**
```
TF-IDF = TF × IDF

TF (Term Frequency) = count of word in document / total words in document
IDF (Inverse Document Frequency) = log(total documents / documents containing word)
```

**Intuition:**
- **High TF-IDF** → word is frequent in this document but rare overall (important!)
- **Low TF-IDF** → word is either rare here or common everywhere (not distinctive)

**Example:**
```
Document 1: "Machine learning is great. Machine learning is fun."
Document 2: "Deep learning uses neural networks."

Word "machine" in Doc 1:
- TF = 2/10 = 0.2 (appears twice in 10 words)
- IDF = log(2/1) = 0.3 (appears in 1 of 2 documents)
- TF-IDF = 0.2 × 0.3 = 0.06

Word "is" in Doc 1:
- TF = 2/10 = 0.2
- IDF = log(2/2) = 0 (appears in all documents - common word)
- TF-IDF = 0.2 × 0 = 0 (not distinctive)
```

**Pros:**
- ✅ Downweights common words ("the", "is")
- ✅ Highlights important words
- ✅ Better than raw counts for retrieval

**Cons:**
- ❌ Still ignores word order
- ❌ No semantic understanding
- ❌ Static vocabulary

**When to use:**
- Document retrieval and search
- Feature extraction for classical ML
- Keyword extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Machine learning is great",
    "Deep learning uses neural networks",
    "Machine learning uses algorithms"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())  # Higher values = more important words
```

### 3. Word2Vec

**Revolutionary idea:** Learn embeddings by predicting context

**Two Architectures:**

#### A. CBOW (Continuous Bag of Words)
**Goal:** Predict center word from context

```
Sentence: "The cat sat on the mat"
Window size = 2

Training examples:
Context: [The, cat, on, the] → Target: sat
Context: [cat, sat, the, mat] → Target: on
```

**Architecture:**
```
Input: One-hot vectors for context words
    ↓
Average / Sum
    ↓
Hidden Layer (embedding)
    ↓
Output: Softmax over vocabulary
```

#### B. Skip-Gram
**Goal:** Predict context from center word (opposite of CBOW)

```
Training examples:
Input: sat → Targets: [The, cat, on, the]
Input: on → Targets: [cat, sat, the, mat]
```

**Skip-gram works better for small datasets and rare words**
**CBOW is faster and works better for frequent words**

**Training Objective (Skip-gram):**
```
Maximize: P(context | word)

For word "sat" with context "cat":
Maximize: P("cat" | "sat")
```

**Key Innovation - Negative Sampling:**
- Instead of updating all words (expensive), update:
  - Target word (positive sample)
  - Few random words (negative samples)
- Makes training feasible

**What Word2Vec Captures:**

```python
# Semantic relationships
king - man + woman ≈ queen
paris - france + italy ≈ rome

# Syntactic patterns
walking - walk + swim ≈ swimming

# Analogies
good:better :: bad:worse
```

**Pros:**
- ✅ Captures semantic relationships
- ✅ Fixed-size dense vectors
- ✅ Efficient training
- ✅ Pre-trained models available

**Cons:**
- ❌ **Static embeddings** - "bank" (river) and "bank" (financial) have same embedding
- ❌ Out-of-vocabulary words get no embedding
- ❌ Doesn't leverage sub-word information

**When to use:**
- Need semantic similarity (document clustering, recommendation)
- Limited training data (use pre-trained)
- Don't need context-specific meanings

```python
from gensim.models import Word2Vec

# Training
sentences = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "uses", "algorithms"]
]

model = Word2Vec(
    sentences,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window
    min_count=1,      # Min word frequency
    sg=1              # 1=Skip-gram, 0=CBOW
)

# Get embedding
vector = model.wv['machine']  # 100-dimensional vector

# Find similar words
similar = model.wv.most_similar('machine', topn=5)
# [('learning', 0.89), ('algorithms', 0.76), ...]

# Analogies
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)  # Should give 'queen'
```

### 4. GloVe (Global Vectors)

**Concept:** Combine global matrix factorization with local context

**Key Idea:**
- Word2Vec uses local context windows
- GloVe uses **global co-occurrence statistics**

**Training:**
1. Build word-word co-occurrence matrix across entire corpus
2. Factorize matrix to get dense vectors

**Formula:**
```
Minimize: Σ f(X_ij) × (w_i^T × w_j + b_i + b_j - log(X_ij))²

X_ij = number of times word j appears in context of word i
w_i, w_j = word vectors
f(X_ij) = weighting function (gives less weight to rare co-occurrences)
```

**Pros:**
- ✅ Leverages global statistics
- ✅ Faster training than Word2Vec
- ✅ Often better performance on analogy tasks

**Cons:**
- ❌ Still static embeddings
- ❌ Requires large corpus for good co-occurrence stats

**When to use:**
- Similar to Word2Vec
- When you have large corpus and computational resources
- Analogy tasks

```python
# Using pre-trained GloVe
import numpy as np

# Download GloVe from Stanford NLP
# Load embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Get embedding
vector = embeddings_index['machine']  # 100-dimensional vector
```

### 5. FastText

**Key Innovation:** Uses **sub-word information** (character n-grams)

**Problem with Word2Vec/GloVe:**
```
"teaching" and "teacher" get completely different embeddings
No embedding for "teachable" if not in training vocab
```

**FastText Solution:**
```
Word "teaching" is represented by:
- The word itself: <teaching>
- Character n-grams: <te, tea, eac, ach, chi, hin, ing, ng>

Embedding = average of all these components
```

**Advantages:**
- ✅ Handles out-of-vocabulary words (use character n-grams)
- ✅ Works well for morphologically rich languages (German, Turkish)
- ✅ Shares information between related words
- ✅ Better for rare words

**Cons:**
- ❌ Larger model size (more subword units)
- ❌ Slower training and inference

**When to use:**
- Morphologically rich languages
- Small vocabularies or many rare words
- Need to handle typos or out-of-vocabulary words

```python
from gensim.models import FastText

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1  # Skip-gram
)

# Get embedding for in-vocabulary word
vector = model.wv['machine']

# Get embedding for out-of-vocabulary word (uses character n-grams)
vector_oov = model.wv['machinelearning']  # Even if not in training!
```

---

## Summary: Classical Embeddings Comparison

| Method | Captures Semantics | Handles OOV | Context-Aware | Training Speed | Use Case |
|--------|-------------------|-------------|---------------|----------------|----------|
| **BoW** | ❌ | ❌ | ❌ | ⚡⚡⚡ | Simple classification |
| **TF-IDF** | ❌ | ❌ | ❌ | ⚡⚡⚡ | Document retrieval |
| **Word2Vec** | ✅ | ❌ | ❌ | ⚡⚡ | Semantic similarity |
| **GloVe** | ✅ | ❌ | ❌ | ⚡⚡⚡ | Analogy tasks |
| **FastText** | ✅ | ✅ | ❌ | ⚡ | Rare words, morphology |

**Key Limitation of ALL Classical Embeddings:**
- **Static**: "bank" always gets the same embedding regardless of context
  - "river bank" → same embedding as "bank account"
  - Need contextual embeddings to solve this!

---

## Contextual Embeddings (The Modern Era)

### The Static Embedding Problem

```
Sentence 1: "I went to the bank to deposit money"
Sentence 2: "I sat on the river bank"

Word2Vec/GloVe: "bank" gets SAME embedding in both
❌ Misses that "bank" means different things!
```

### Evolution to Contextual Embeddings

#### ELMo (Embeddings from Language Models)

**Key Innovation:** Embeddings depend on context

**How it works:**
- Train bidirectional LSTM language model
- Embedding = function of entire sentence

```
"I went to the bank"
↓
[LSTM forward] → → → →
[LSTM backward] ← ← ← ←
↓
Context-aware embedding for "bank" (financial)

"I sat on the river bank"
↓
[LSTM forward] → → → →
[LSTM backward] ← ← ← ←
↓
Different embedding for "bank" (river)
```

**Advantage:** Same word gets different embeddings based on context
**Limitation:** LSTMs are slow, don't parallelize well

---

(Continued in next part due to length...)

**Next Section:** [BERT Embeddings and Transformers](#bert-embeddings) | **Back:** [README](../03_Modern_NLP_and_Transformers/README.md)
