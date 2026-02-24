# 09 Cheat Sheets — ML/AI Interview Prep

This directory contains dense, last-minute revision cheat sheets for senior ML/AI engineering interviews. Each file is designed to be scanned in 5-10 minutes before an interview round. They are not tutorials — they assume you already know the concepts and need rapid recall.

---

## When to Use Each Cheat Sheet

| Situation | File to Open |
|-----------|-------------|
| "Design an ML system for X" (system design round) | `ml_algorithm_selection_guide.md` → `rag_design_checklist.md` → `mlops_checklist.md` |
| LLM / NLP system design or architecture questions | `transformer_and_llm_cheat_sheet.md` |
| Building a RAG pipeline from scratch | `rag_design_checklist.md` |
| MLOps, deployment, monitoring questions | `mlops_checklist.md` |
| "You're in a behavioral/coding trap" | `common_interview_traps.md` |
| Probability, statistics, loss functions | `statistics_and_math_quick_reference.md` |
| Whiteboard: pick the right algorithm | `ml_algorithm_selection_guide.md` |

---

## Quick-Access Links

1. **[ML Algorithm Selection Guide](ml_algorithm_selection_guide.md)**
   - Decision flowchart (Mermaid)
   - Algorithm comparison table (20+ algorithms)
   - Tabular / NLP / time series sub-guides
   - Common algorithm traps

2. **[Transformer & LLM Cheat Sheet](transformer_and_llm_cheat_sheet.md)**
   - Architecture formulas and parameter counts
   - Positional encoding comparison
   - Tokenization quick reference
   - Model family comparison table (BERT → LLaMA 3 → Claude)
   - Fine-tuning decision matrix (LoRA, QLoRA, full FT)
   - Quantization formats
   - Decoding strategies
   - Evaluation benchmark quick reference

3. **[RAG Design Checklist](rag_design_checklist.md)**
   - RAG vs fine-tuning vs ICL decision matrix
   - Document processing, chunking, embedding, vector DB selection
   - Retrieval tuning, generation, evaluation, and production checklists
   - Embedding model comparison table
   - Vector DB comparison table

4. **[MLOps Checklist](mlops_checklist.md)**
   - Complete data pipeline, experiment management, packaging, deployment, monitoring checklists
   - LLM-specific MLOps section
   - Drift detection methods
   - Rollout strategy guide

5. **[Common Interview Traps](common_interview_traps.md)**
   - 20 traps with: the trap, why people fall for it, correct nuanced answer, follow-up questions
   - Covers: data leakage, class imbalance, normalization myths, embedding mistakes, LLM misconceptions

6. **[Statistics & Math Quick Reference](statistics_and_math_quick_reference.md)**
   - Probability, distributions, MLE/MAP
   - Information theory formulas
   - Linear algebra essentials
   - Optimization (gradient descent, Adam full formulas)
   - Metric formulas (Precision, Recall, F1, AUC, NDCG, BLEU, Perplexity)
   - Attention formula derivation

---

## How to Use This Section

### 30 Minutes Before an Interview
1. Read `common_interview_traps.md` — know what NOT to say.
2. Skim the relevant cheat sheet for the expected topic.

### During a System Design Question
1. Open `ml_algorithm_selection_guide.md` — work top-down through the decision tree.
2. If LLMs are involved, reference `transformer_and_llm_cheat_sheet.md` for sizes and tradeoffs.
3. Reference `rag_design_checklist.md` if retrieval is part of the design.
4. Close with `mlops_checklist.md` to address productionization.

### For Math/Stat Questions
- Open `statistics_and_math_quick_reference.md` and scan the relevant section.

---

## Formatting Conventions Used

- `[ ]` = checklist item to verify before/during implementation
- Tables: columns sorted by importance for interview context
- Code blocks: Python pseudocode or exact formulas
- Mermaid diagrams: copy-paste ready for Excalidraw or Notion

---

## Maintenance Notes

- Last reviewed: February 2026
- Model family table covers models through LLaMA 3, Mistral, Claude 3.5, Gemini 1.5
- Benchmark scores are approximate; always verify against current leaderboards
- Vector DB comparison reflects state of Pinecone, Weaviate, Qdrant, Chroma, pgvector as of early 2026
