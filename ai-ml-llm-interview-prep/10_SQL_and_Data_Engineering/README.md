# 10. SQL and Data Engineering for Senior ML Roles

> **Core message**: At the senior level, you are not just a modeler. You own the data. You build the pipelines. You write the SQL that creates the features your models consume. Interviewers at FAANG, logistics companies, and financial institutions consistently report that SQL and data engineering gaps are the #1 reason otherwise strong ML candidates fail to clear the senior bar.

---

## Why Data Engineering Skills Are Non-Negotiable for Senior ML

A junior ML engineer can be handed a clean DataFrame and asked to train a model. A **senior ML engineer** is expected to:

1. **Design the schema** that stores raw events from operational systems
2. **Write the SQL** that transforms raw events into model-ready features
3. **Build and own the pipeline** that runs that SQL on a schedule, handles failures, and alerts on data quality issues
4. **Reason about time** — lag features, rolling windows, train/test leakage, temporal ordering
5. **Run rigorous experiments** — A/B tests, not just offline metric comparisons
6. **Apply causal thinking** — understanding that correlation does not imply the intervention will work
7. **Work with graph-structured data** — supply chains, recommendation systems, fraud networks

If you cannot do these things, you will hit a ceiling regardless of how sophisticated your modeling skills are. The data engineering layer is where most ML projects succeed or fail in production.

---

## Section Contents

| File | Topic | Priority |
|------|-------|----------|
| `sql_for_data_scientists.md` | Complete SQL reference: joins, window functions, CTEs, ML-specific patterns, query optimization | CRITICAL |
| `data_pipelines_and_etl.md` | ETL vs ELT, Airflow, Kafka, dbt, medallion architecture, data quality | CRITICAL |
| `time_series_forecasting_deep_dive.md` | ARIMA, Prophet, LightGBM for time series, walk-forward validation, hierarchical forecasting | HIGH |
| `causal_inference_basics.md` | Rubin framework, DAGs, DiD, propensity scores, uplift modeling | HIGH |
| `ab_testing_and_experimentation.md` | Statistical testing, sample size, sequential testing, MAB, ML-specific experimentation | HIGH |
| `graph_ml_basics.md` | GNNs, GCN, GAT, GraphSAGE, knowledge graphs, logistics applications | MEDIUM |

---

## Recommended Study Order

### Week 1: SQL Mastery (Foundation)
Start here. If you cannot fluently write window functions, CTEs, and self-joins, everything else is blocked. Senior ML roles at companies like Amazon Logistics, DoorDash, Uber Freight, and FedEx literally begin interviews with SQL screens.

```
Day 1-2: sql_for_data_scientists.md — Sections 1-3 (Fundamentals, JOINs, Aggregations)
Day 3-4: sql_for_data_scientists.md — Sections 4-5 (Window Functions, CTEs)
Day 5-7: sql_for_data_scientists.md — Sections 6-8 (Advanced patterns, ML-specific SQL, Interview Qs)
```

### Week 2: Data Pipelines
```
Day 1-3: data_pipelines_and_etl.md — ETL fundamentals, Airflow, data quality
Day 4-5: data_pipelines_and_etl.md — Streaming, Kafka, modern data stack
Day 6-7: data_pipelines_and_etl.md — ML pipeline patterns, interview prep
```

### Week 3: Experimentation and Causal Reasoning
```
Day 1-3: ab_testing_and_experimentation.md — Statistical foundations, experiment design
Day 4-5: ab_testing_and_experimentation.md — Sequential testing, MAB, ML-specific
Day 6-7: causal_inference_basics.md — Full file
```

### Week 4: Advanced Topics
```
Day 1-3: time_series_forecasting_deep_dive.md
Day 4-5: graph_ml_basics.md
Day 6-7: Review and practice problems
```

---

## How Senior ML Interviews Actually Test These Skills

### Type 1: SQL Screen (30-45 minutes)
You will be given a schema and asked to write queries. Common patterns:
- Window functions to compute rolling metrics
- Self-joins to find sequential events
- CTEs to compute multi-step calculations
- The interviewer is watching whether you write clean, correct SQL quickly

### Type 2: System Design (60 minutes)
"Design an ML system for ETA prediction at scale." You will be expected to describe:
- What data you ingest and from where
- How the feature pipeline works
- What the training pipeline looks like
- How you deploy, monitor, and retrain

Without data engineering knowledge, you cannot pass this round.

### Type 3: Case Study / Analytics (45-60 minutes)
"Our delivery success rate dropped 3% last week. How do you investigate?" This requires:
- SQL to segment and diagnose
- Understanding of pipeline failures
- Causal reasoning about what changed

### Type 4: ML Design (60 minutes)
"How would you build a model to predict package delay?" This requires:
- Feature engineering design (lag features, rolling aggregates — implemented in SQL or a pipeline)
- Validation strategy (walk-forward for time series)
- Experiment design (how do you A/B test the new model?)

---

## Key Themes Across All Files

### 1. Temporal Correctness
The most common production bug in ML systems is **temporal leakage**: using future information to predict the past. This shows up in:
- SQL: joining on dates without strict `<` conditions
- Features: using a rolling average that includes the target date
- Validation: random train/test split on time series data

Every file in this section touches on this theme. Internalize it.

### 2. Idempotency
Production pipelines **must** be safe to re-run. If a pipeline fails at step 3 and you re-run it, you should not get duplicate rows in your feature table. This is achieved through:
- `INSERT OVERWRITE` instead of `INSERT INTO`
- Upsert patterns (MERGE)
- Partitioning by date so reruns only affect the relevant partition

### 3. At What Granularity Are You Modeling?
Every ML problem has a **unit of analysis**. Is it:
- A package (track by shipment ID)?
- A customer (track by customer ID)?
- A route (track by origin-destination pair)?
- A day-by-carrier combination?

Getting the granularity wrong — either in SQL feature queries or in model design — produces silently wrong results. This is tested heavily.

### 4. The Metric Hierarchy
Not all metrics are equal. Know the difference between:
- **Guardrail metrics**: must not regress (latency, error rates)
- **Primary metrics**: what you are optimizing
- **Secondary metrics**: supporting signals
- **Proxy metrics**: stand-ins for what you truly care about but cannot measure directly

---

## Quick Reference: Vocabulary You Must Know Cold

| Term | Definition |
|------|------------|
| Idempotency | Re-running a pipeline produces the same result |
| Exactly-once semantics | Each event is processed exactly one time |
| Late-arriving data | Events that arrive after their logical processing window |
| Temporal leakage | Using future data as input to a model |
| Walk-forward validation | Time-respecting cross-validation for time series |
| Confounder | Variable that affects both treatment and outcome |
| ATE | Average Treatment Effect — the estimand of causal inference |
| SRM | Sample Ratio Mismatch — sign of a broken A/B test |
| Over-smoothing | Deep GNNs making all nodes have similar representations |
| Medallion architecture | Bronze/Silver/Gold data lake layers |
| dbt | Data Build Tool — SQL transformations with testing and lineage |
| Reconciliation | Making hierarchical forecasts sum to their aggregates |

---

## Logistics Domain Context

Throughout this section, examples are grounded in a **logistics / last-mile delivery** context because:
1. Logistics companies (Amazon, FedEx, UPS, DHL, DoorDash) are large employers of senior ML engineers
2. Logistics data has rich temporal structure (ideal for time series examples)
3. Logistics has clear causal questions (did this routing change reduce delays?)
4. Graph structure is natural (packages flow through depot networks)

The core SQL and engineering concepts apply universally. The logistics framing makes the examples concrete and memorable.

---

## Related Sections in This Repository

- `../05_ML_System_Design/` — How pipelines fit into end-to-end ML system design
- `../06_Deep_Learning/` — Deep learning for time series (LSTMs, Transformers)
- `../09_LLMs_and_GenAI/` — LLM pipelines and RAG system engineering
- `../04_Model_Evaluation/` — Offline evaluation metrics (complements A/B testing)

---

*This section was designed for engineers targeting senior/staff ML roles at data-intensive companies. The depth here reflects what it takes to clear the bar, not just pass a screen.*
