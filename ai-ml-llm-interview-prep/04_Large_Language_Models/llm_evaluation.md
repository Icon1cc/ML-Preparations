# LLM Evaluation Frameworks

## Why LLM eval is hard
Outputs are open-ended; lexical overlap may not reflect quality, factuality, or usefulness.

## Multi-layer evaluation stack
1. Offline benchmark scores.
2. Task-specific automated checks.
3. LLM-as-judge with calibration.
4. Human preference review.
5. Online business KPIs.

```mermaid
flowchart LR
    A[Offline benchmark] --> B[Task eval harness]
    B --> C[Human/LLM judge]
    C --> D[Online A/B]
    D --> E[Production monitoring]
```

## Benchmark quick map
- MMLU: broad knowledge.
- HellaSwag: commonsense completion.
- GSM8K: arithmetic reasoning.
- HumanEval: coding.
- TruthfulQA: hallucination resistance.
- HELM: multi-axis holistic evaluation.

## LLM-as-judge
Pros:
- scalable
- cheap relative to full human labeling

Cons:
- position bias, verbosity bias, model-family favoritism
- requires calibration against human labels

## RAG-specific metrics
- Retrieval recall@k
- context precision
- faithfulness/groundedness
- answer relevance

## Enterprise eval pipeline
- Build gold dataset from support tickets and expert answers.
- Include adversarial/security cases.
- Evaluate per slice: language, region, policy category.

## Tools
- Ragas
- DeepEval
- Promptfoo
- LangSmith eval traces

## Interview questions
1. How build eval pipeline for enterprise LLM?
2. What are limitations of LLM-as-judge?
3. How correlate offline and online metrics?

## Example eval record schema
```json
{
  "query": "Where is my shipment?",
  "expected_facts": ["tracking API status", "ETA window"],
  "response": "...",
  "retrieved_docs": ["doc1", "doc2"],
  "faithfulness": 0.88,
  "helpfulness": 4,
  "reviewer": "human"
}
```
