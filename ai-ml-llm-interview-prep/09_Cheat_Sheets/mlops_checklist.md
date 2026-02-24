# MLOps Checklist

## Data pipeline
- [ ] Schema validation
- [ ] Null/range/distribution checks
- [ ] Point-in-time correctness
- [ ] Data versioning and lineage

## Experiment management
- [ ] Track params/metrics/artifacts
- [ ] Reproducible environment + seed
- [ ] Compare against baseline

## Model packaging
- [ ] Versioned artifact
- [ ] Dockerized serving runtime
- [ ] Input/output contract
- [ ] Model card and risk notes

## Deployment
- [ ] Unit + integration tests
- [ ] Quality gate before release
- [ ] Shadow/canary rollout
- [ ] Rollback procedure documented

## Monitoring
- [ ] Data drift alerts
- [ ] Prediction drift alerts
- [ ] Performance tracking with delayed labels
- [ ] Latency/error SLO alerts

## LLM-specific
- [ ] Prompt version control
- [ ] Token cost monitoring
- [ ] Guardrail pass-rate tracking
- [ ] RAG faithfulness monitoring
