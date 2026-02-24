# Statistics and Math Quick Reference

## Probability
- Bayes: `P(A|B)=P(B|A)P(A)/P(B)`
- Chain rule: `P(x1..xn)=Π P(xi|x1..x(i-1))`
- Expectation: `E[X]=Σ x p(x)`
- Variance: `Var(X)=E[(X-E[X])^2]`

## Core distributions
- Bernoulli(p), Binomial(n,p), Normal(mu,sigma^2), Poisson(lambda), Beta(alpha,beta), Dirichlet(alpha)

## Information theory
- Entropy: `H(X)=-Σ p(x)log p(x)`
- Cross-entropy: `H(p,q)=-Σ p(x)log q(x)`
- KL: `KL(p||q)=Σ p(x)log(p(x)/q(x))`

## Linear algebra
- Matrix multiply: `(m×n)(n×p)=(m×p)`
- Eigendecomposition: `A=QΛQ^-1`
- SVD: `A=UΣV^T`
- Norms: `||x||1`, `||x||2`, `||A||F`

## Optimization
- GD: `theta <- theta - lr * grad`
- Adam:
  - `m_t = beta1 m_{t-1} + (1-beta1) g_t`
  - `v_t = beta2 v_{t-1} + (1-beta2) g_t^2`
  - `theta_t = theta_{t-1} - lr * mhat / (sqrt(vhat)+eps)`

## Metrics
- Precision = `TP/(TP+FP)`
- Recall = `TP/(TP+FN)`
- F1 = `2PR/(P+R)`
- RMSE = `sqrt(mean((y-yhat)^2))`
- NDCG = `DCG/IDCG`
- Perplexity = `exp(-mean(log p(token)))`

## Attention
- `Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`
- `MultiHead = Concat(head_i)W^O`
