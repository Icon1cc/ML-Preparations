# A/B Testing and Experimentation

## Why experiments
A/B tests estimate causal impact in production under uncertainty.

## Core statistics
- Null/alternative hypothesis
- Type I error alpha
- Type II error beta and power
- p-value interpretation
- confidence intervals and effect size

## Sample size planning
Depends on baseline rate, MDE, alpha, and power.
Smaller MDE requires larger sample.

## Experiment design
- randomization unit (user/session/device)
- stratified allocation
- guardrail metrics
- primary and secondary metrics

## Pitfalls
- peeking/p-hacking
- multiple comparisons
- sample ratio mismatch (SRM)
- network effects and interference

## Sequential and Bayesian testing
- sequential alpha spending
- Bayesian posterior probability decisioning

## ML-specific practice
- offline metrics first, online A/B for final validation
- shadow testing before full traffic

## Interview questions
1. How compute sample size?
2. What is peeking problem?
3. If p=0.08 with positive lift, what do you do?

## Python snippet
```python
from statsmodels.stats.proportion import proportions_ztest
count = [520, 560]
nobs = [10000, 10000]
stat, pval = proportions_ztest(count, nobs)
print(stat, pval)
```
