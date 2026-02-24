# Fine-Tuning Transformers in Practice

## Strategies
1. Feature extraction: freeze backbone, train task head.
2. Full fine-tuning: update all parameters.
3. Parameter-efficient methods (LoRA/QLoRA, see LLM section).

## When to fine-tune
- Domain shift is high.
- You need strict task behavior.
- Enough labeled data and compute available.

## When not to fine-tune
- Small dataset and strong base model already performs well.
- Requirements are mostly prompt-level formatting.
- Knowledge freshness needed -> use RAG.

## Optimization heuristics
- Small LR (`1e-5` to `5e-5`) for encoder models.
- Warmup steps (5-10%).
- Weight decay for regularization.
- Layer-wise LR decay for stability.

## Catastrophic forgetting mitigation
- Mix domain and generic data.
- Use lower LR.
- Freeze lower layers initially.
- Regularize with KL to base model outputs.

## Task heads
- Classification: linear head on `[CLS]` or pooled representation.
- NER: token-level classifier.
- QA: start/end span heads.

## Interview questions
1. Fine-tune vs in-context learning?
2. Why low LR in transformer fine-tuning?
3. How prevent catastrophic forgetting?

## Hugging Face Trainer example
```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = 'distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

data = Dataset.from_dict({
    'text': ['package delayed', 'delivery successful'] * 200,
    'label': [1, 0] * 200
})

def preprocess(batch):
    return tok(batch['text'], truncation=True, padding='max_length', max_length=64)

ds = data.map(preprocess, batched=True)
ds = ds.train_test_split(test_size=0.2, seed=42)

args = TrainingArguments(
    output_dir='tmp_ft',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=20
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['test']
)

# trainer.train()
```
