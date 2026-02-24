# Full Fine-Tuning vs PEFT (LoRA, QLoRA, Prefix, Prompt Tuning)

## Full fine-tuning
Update all model parameters.

Pros:
- maximum adaptation capacity
- best for major distribution/task shifts when compute is available

Cons:
- expensive memory + compute
- larger risk of catastrophic forgetting
- difficult multi-tenant model management

## LoRA
Low-rank adapters added to weight update:
`W' = W + DeltaW`, `DeltaW = A B`, rank `r << min(d,k)`

Benefits:
- train small number of parameters
- reduced memory
- can merge adapters for deployment

Key knobs:
- target modules (`q_proj`, `v_proj`, etc.)
- rank `r`
- alpha scaling
- dropout

## QLoRA
Combine 4-bit quantized base model (NF4) + LoRA adapters.
Enables fine-tuning large models on limited hardware.

## Other PEFT methods
- Prefix tuning: learn virtual tokens per layer.
- Prompt tuning: learn soft prompt embeddings only.
- IA3/adapters: lightweight module-based adaptation.

## Comparison

| Method | Trainable params | Memory | Quality | Best use |
|---|---:|---|---|---|
| Full FT | 100% | highest | highest (usually) | mission-critical strong shift |
| LoRA | low | low | high | common enterprise adaptation |
| QLoRA | very low | very low | high-medium | limited GPU budget |
| Prefix/Prompt tuning | minimal | minimal | medium | fast lightweight specialization |

## Fine-tuning vs RAG
Fine-tune for style/behavior and domain instruction following.
Use RAG for fresh, citeable knowledge.
Use both for enterprise assistants requiring both style and factual grounding.

## Interview questions
1. Explain LoRA mathematically.
2. When use QLoRA over LoRA?
3. How pick LoRA rank?

## PEFT code example
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)
model = get_peft_model(base, config)
model.print_trainable_parameters()
```
