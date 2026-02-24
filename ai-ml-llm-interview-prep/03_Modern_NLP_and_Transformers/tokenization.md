# Tokenization Strategies for Transformers

## Why tokenization matters
Tokenization controls vocabulary, sequence length, inference cost, and what patterns the model can represent.

## Levels
- Character-level: no OOV, long sequences.
- Word-level: short sequences, severe OOV.
- Subword-level: practical balance.

## BPE (Byte Pair Encoding)
Algorithm:
1. Start from basic symbols.
2. Count frequent adjacent pairs.
3. Merge highest-frequency pair.
4. Repeat until vocabulary size reached.

GPT families typically use byte-level BPE variants.

## WordPiece
Used by BERT-like models. Merge criterion based on likelihood improvement.

## SentencePiece
Language-agnostic; trains directly on raw text (no whitespace assumptions). Common in T5/LLaMA-like ecosystems.

## Unigram LM tokenizer
Starts with large vocab and prunes tokens by likelihood.

## Special tokens
- `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`
- `<bos>`, `<eos>`, `<unk>`

## Tokenization and cost
Inference cost often scales with tokens, not characters or words.
Prompt compression via better formatting can reduce cost significantly.

## Common failure modes
- Numeric reasoning fails due to token boundaries and weak arithmetic training.
- Multilingual mismatch from vocabulary bias.
- Domain-specific jargon split into many subwords.

## Interview questions
1. Why does BPE help with OOV words?
2. Why can tokenization hurt arithmetic?
3. Difference between WordPiece and BPE?

## Simple BPE demo (toy)
```python
from collections import Counter

def most_frequent_pair(tokens):
    pairs = Counter()
    for seq in tokens:
        for i in range(len(seq)-1):
            pairs[(seq[i], seq[i+1])] += 1
    return pairs.most_common(1)[0][0]

corpus = [list("low"), list("lower"), list("newest"), list("widest")]
pair = most_frequent_pair(corpus)
print("Most frequent pair:", pair)
```
