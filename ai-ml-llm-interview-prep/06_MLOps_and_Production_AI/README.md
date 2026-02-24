# 06 — MLOps and Production AI

> Senior-level interview preparation for machine learning engineering, platform engineering, and AI systems roles. Every section is written at the depth expected in FAANG, tier-1 logistics, and enterprise AI platform interviews.

---

## Why MLOps Exists

Academic ML ends when a model achieves a benchmark score. Production ML begins there. The gap between a Jupyter notebook with 92% accuracy and a reliable, cost-effective, observable system serving real traffic is where most ML projects fail — and where senior engineers earn their keep.

MLOps (Machine Learning Operations) is the discipline of applying software engineering rigor — DevOps, SRE, data engineering — to the full ML lifecycle. It emerged because ML systems have properties that make standard software engineering approaches insufficient:

- **Non-determinism**: Training a model twice on the same data may yield different weights depending on random seeds, hardware, and framework versions.
- **Data dependency**: Unlike traditional software, ML system behavior is determined as much by data as by code.
- **Gradual degradation**: ML systems fail silently. A production API returning HTTP 200 may simultaneously be returning increasingly wrong predictions.
- **Feedback loops**: A deployed model influences user behavior, which changes future training data, which changes the model — a control-theory problem rarely addressed in software engineering education.
- **Experimentation culture**: Unlike most software (where one version is correct), ML teams run many parallel experiments. Managing this requires infrastructure traditional DevOps never needed.

---

## Section Map

| File | Topic | Key Interview Themes |
|------|--------|----------------------|
| `ml_lifecycle.md` | Full ML lifecycle and hidden technical debt | Project failure modes, team structure, Sculley paper |
| `experiment_tracking.md` | MLflow, W&B, DVC, reproducibility | How to reproduce experiments, model registry |
| `data_and_model_versioning.md` | DVC, Delta Lake, Iceberg, model registry | Why data versioning is hard, rollback |
| `cicd_for_ml.md` | ML pipelines, automated retraining, canary deploys | Google MLOps maturity levels, pipeline tools |
| `monitoring_and_drift.md` | Drift detection, PSI, alerting strategies | KS test, concept drift vs data drift, delayed labels |
| `feature_stores.md` | Training-serving skew, online/offline stores | Point-in-time retrieval, Feast, Tecton |
| `batch_vs_realtime_inference.md` | Serving patterns, latency SLAs, autoscaling | When to batch vs real-time, GPU vs CPU |
| `cloud_and_containerization.md` | Docker, K8s, AWS/GCP/Azure ML | SageMaker vs Vertex, spot instances, multi-cloud |
| `testing_ml_systems.md` | ML testing pyramid, behavioral tests | Great Expectations, invariance tests, regression tests |

---

## The Four Pillars

### 1. Reproducibility

Every experiment must be reproducible: same code + same data + same environment = same model. This requires:

- **Code versioning**: Git (standard)
- **Data versioning**: DVC, Delta Lake, Iceberg
- **Environment versioning**: Docker, conda lock files
- **Experiment tracking**: MLflow, W&B
- **Seed management**: NumPy, PyTorch, random seeds fixed and logged

Reproducibility is not just academic integrity — it is a prerequisite for debugging production incidents. When a model starts misbehaving, you need to be able to reproduce the exact state that existed when it was trained.

### 2. Scalability

ML workloads have extreme compute heterogeneity:

- **Training**: burst compute, GPU-heavy, hours to days per job, embarrassingly parallel across hyperparameter search
- **Inference (batch)**: high throughput, CPU or GPU, latency-tolerant, hours to process
- **Inference (real-time)**: latency-critical, sub-100ms SLAs, horizontal scaling required
- **Feature computation**: streaming or batch, variable data volumes

The infrastructure must scale independently across each axis. Kubernetes, autoscaling groups, and managed ML platforms (SageMaker, Vertex AI) all address different parts of this problem.

### 3. Observability

You cannot manage what you cannot measure. ML observability goes beyond traditional APM:

- **Infrastructure metrics**: CPU, GPU utilization, memory, latency, error rates (standard)
- **Data metrics**: input distribution, feature statistics, null rates, cardinality
- **Model metrics**: prediction distribution, confidence calibration, output variance
- **Business metrics**: downstream KPIs that the model influences
- **Ground truth metrics**: model accuracy, precision, recall — when labels eventually arrive

The delayed labels problem makes ML observability especially hard: you often know what the model predicted months before you know if the prediction was correct.

### 4. Reliability

Production ML systems must be reliable even when their components are not:

- **Graceful degradation**: fall back to a simpler rule-based system, a previous model version, or a cached response when the primary model fails
- **Circuit breakers**: stop sending traffic to a failing model endpoint
- **Canary deploys**: route small fractions of traffic to new model versions before full rollout
- **Automated rollback**: trigger rollback when error rate or drift score exceeds threshold

Reliability for ML is harder than for traditional APIs because model failures are rarely binary (HTTP 500) — they are often silent quality degradation.

---

## Common Interview Frameworks

### The "Three Environments" Model

```
Development Environment     → Staging Environment     → Production Environment
(Jupyter, local GPU)           (Kubeflow/Vertex)          (Serving cluster)
  ↓ code, config               ↓ full pipeline test        ↓ traffic split
  ↓ experiment tracking        ↓ shadow deployment         ↓ monitoring
  ↓ DVC data pull              ↓ integration tests         ↓ alerting
```

### The "Five Questions" for Any ML System

When designing or reviewing an ML system at an interview, always answer:

1. **Where does the data come from?** (freshness, quality, volume, lineage)
2. **What is the feature pipeline?** (batch vs streaming, training-serving skew)
3. **How is the model trained and validated?** (experiment tracking, data splits, evaluation)
4. **How is the model served?** (batch vs real-time, latency SLA, scale)
5. **How is the system monitored?** (drift, business metrics, alerting, feedback loop)

### The Google MLOps Maturity Model

| Level | Name | Characteristics |
|-------|------|-----------------|
| 0 | Manual | Notebooks, manual steps, no reproducibility |
| 1 | ML pipeline automation | Automated training, experiment tracking, model registry |
| 2 | CI/CD pipeline automation | Automated retraining triggers, automated testing, automated deployment |

Most enterprise teams operate between Level 0 and Level 1. Level 2 is aspirational but achievable for high-value models.

---

## Logistics Industry Context

Throughout this section, examples are grounded in logistics and supply chain:

- **Delivery ETA prediction**: real-time inference, sub-100ms SLA, 10M+ requests/day
- **Demand forecasting**: batch inference, overnight, weekly model retraining
- **Route optimization**: combinatorial optimization + ML, batch with real-time augmentation
- **Fraud/anomaly detection**: real-time, streaming features, concept drift common
- **Document processing**: OCR + NLP, batch pipeline, high throughput

These examples are chosen because they represent the full spectrum of ML deployment patterns and appear frequently in senior interview discussions.

---

## Key Papers and References

- **Sculley et al. (2015)**: "Hidden Technical Debt in Machine Learning Systems" — NIPS paper that coined the framework. Must-read for understanding ML technical debt.
- **Google MLOps Whitepaper (2021)**: Defines the three levels of MLOps maturity.
- **Breck et al. (2017)**: "The ML Test Score" — systematic rubric for ML production readiness.
- **Zhang et al. (2020)**: "Systems for Machine Learning" — systems perspective on ML infrastructure.
- **Klaise et al. (2020)**: Alibi Detect paper — reference implementation for drift detection.

---

## Quick Reference: Tool Ecosystem

```
Experiment Tracking:    MLflow | Weights & Biases | Comet ML | Neptune
Data Versioning:        DVC | Delta Lake | Apache Iceberg | Apache Hudi
Feature Stores:         Feast | Tecton | Vertex AI FS | SageMaker FS | Databricks FS
Orchestration:          Kubeflow | Prefect | Airflow | Dagster | Vertex Pipelines
Serving:                TorchServe | Triton | BentoML | Seldon | Ray Serve
Monitoring:             Evidently AI | WhyLogs | Arize | Fiddler | Arthur
Testing:                Great Expectations | Pandera | pytest | deepchecks
Cloud ML Platforms:     AWS SageMaker | GCP Vertex AI | Azure ML | Databricks
```

---

## How to Use This Section

1. Read `ml_lifecycle.md` first — it establishes vocabulary and mental models.
2. Study `monitoring_and_drift.md` and `feature_stores.md` deeply — these generate the most senior interview questions.
3. Use `cicd_for_ml.md` for system design discussions about pipeline automation.
4. Refer to `testing_ml_systems.md` when asked "how do you know your ML system is correct?"
5. Cross-reference with Section 07 (System Design) for full end-to-end design problems.

---

*Section 06 — MLOps and Production AI | AI/ML/LLM Interview Prep Repository*
