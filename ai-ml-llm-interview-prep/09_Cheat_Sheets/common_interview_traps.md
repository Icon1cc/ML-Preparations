# Common ML Interview Traps

## Trap 1: "More data always helps"
Correct: more high-quality, relevant, well-labeled data helps. Noisy data can hurt.

## Trap 2: "Deep learning always beats XGBoost"
Correct: for many tabular tasks, boosted trees are stronger and cheaper.

## Trap 3: "High accuracy means good model"
Correct: use task-appropriate metrics (PR-AUC, cost curves, calibration).

## Trap 4: "Cross-entropy is enough for all classification"
Correct: include class imbalance and decision-threshold business cost.

## Trap 5: "More features always better"
Correct: extra noisy/correlated features can reduce generalization.

## Trap 6: "99% test accuracy proves success"
Correct: suspect leakage, split issues, and dataset mismatch.

## Trap 7: "Always normalize data"
Correct: tree models typically do not require scaling.

## Trap 8: "Dropout always helps"
Correct: can hurt if model underfits.

## Trap 9: "Adam always best"
Correct: SGD+momentum may generalize better in some regimes.

## Trap 10: "Use vanilla BERT embeddings for retrieval"
Correct: use sentence-transformers/embedding models + reranking.

## Trap 11: "RAG eliminates hallucinations"
Correct: retrieval itself can fail; groundedness checks still needed.

## Trap 12: "Fine-tuning always beats prompting"
Correct: prompting + retrieval often sufficient and cheaper.

## Trap 13: "Cosine similarity always correct"
Correct: normalization and metric/data geometry matter.

## Trap 14: "Deeper model always better"
Correct: optimization stability and data regime matter.

## Trap 15: "Bigger batch always faster and better"
Correct: may hurt generalization and increase memory pressure.

## Trap 16: "L2 regularization equals weight decay in Adam"
Correct: use AdamW for decoupled weight decay.

## Trap 17: "SMOTE is always needed"
Correct: thresholding and class-weighting may outperform synthetic sampling.

## Trap 18: "t-SNE for production features"
Correct: non-parametric visualization tool, not production transform.

## Trap 19: "A/B test every change the same way"
Correct: watch for network effects, novelty effects, interference.

## Trap 20: "Works offline so it works in prod"
Correct: training-serving skew and drift can break production.
