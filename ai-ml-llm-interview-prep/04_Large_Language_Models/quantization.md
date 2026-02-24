# Quantization for LLM Training and Inference

## Why quantization
Reduce memory footprint and improve throughput with acceptable quality loss.

## Precision formats
- FP32: baseline accuracy, high memory.
- FP16: compact but narrower dynamic range.
- BF16: FP16 memory + wider range; strong for training stability.
- INT8/INT4/NF4: inference-efficient low-bit representations.

## PTQ vs QAT
- Post-training quantization: no retraining or small calibration pass.
- Quantization-aware training: model learns to be robust to quantization noise.

## Methods
- GPTQ: Hessian-aware post-training quantization.
- AWQ: activation-aware weight quantization preserving important channels.
- bitsandbytes: practical 8-bit/4-bit loading in HF ecosystem.
- GGUF/GGML: CPU-oriented deployment format for llama.cpp.

## Tradeoff table

| Method | Quality | Speed | Memory | Typical use |
|---|---|---|---|---|
| FP16/BF16 | best | medium | medium | training/high-end serving |
| INT8 | high | high | low | production inference |
| INT4/NF4 | medium-high | very high | very low | constrained GPU/edge |
| GGUF | variable | CPU-friendly | very low | local/edge deployments |

## Practical guidance
- Training: BF16 + optimizer states in mixed precision.
- Inference: INT8 first; INT4 for tight memory budgets.
- Validate by perplexity + downstream task metrics.

## Interview questions
1. BF16 vs FP16 for training?
2. GPTQ vs AWQ difference?
3. How evaluate quantization quality drop?

## HF bitsandbytes snippet
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', quantization_config=bnb_cfg)
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
```
