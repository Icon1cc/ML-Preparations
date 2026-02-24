# Case Studies: Real-World ML Systems in Production

## Overview

This section contains detailed technical case studies drawn from real-world enterprise and logistics AI/ML deployments. Each case study is structured as a technical design document — the kind you would produce in a system design interview or as part of an architecture review board submission.

These are not toy examples. Each case study covers a system operating at significant scale: millions of packages, terabytes of data, global infrastructure. The goal is to prepare you to discuss practical ML system design with depth, precision, and production credibility.

---

## How to Use These Case Studies for Interview Preparation

### For System Design Interviews

System design interviews at ML-focused companies (Uber, DoorDash, Instacart, Amazon, FedEx, DHL, Shopify, etc.) often take one of three formats:

1. **Open-ended design**: "Design a delivery ETA prediction system." You are expected to drive the conversation, ask clarifying questions, and propose an architecture.
2. **Critique an existing system**: "Here is how we built X. What would you change?" You need to identify weaknesses and propose improvements.
3. **Deep dive on a specific component**: "Tell me about a time you built a forecasting system." You need to go deep on a specific technical decision.

For each case study in this directory, practice all three formats.

### For Behavioral/Experience Questions

Even if you haven't worked on DHL-scale logistics, reading and understanding these case studies lets you discuss tradeoffs intelligently. Interviewers are evaluating whether you can reason about:

- Data quality and availability
- Model selection and tradeoffs
- Evaluation methodology
- Production concerns (latency, throughput, monitoring)
- Business impact

### Recommended Study Approach

1. **First pass**: Read the full case study end-to-end without stopping. Build a mental model of the system.
2. **Second pass**: For each section, ask yourself: "What would I have done differently? What are the failure modes?"
3. **Practice questions**: Use the interview questions at the end of each file. Answer them out loud, as if in a real interview.
4. **Draw the architecture**: Without looking at the Mermaid diagram, try to draw the system architecture from memory.
5. **Explain to a peer**: The best test of understanding is teaching. Explain the system to a colleague or rubber duck.

---

## Case Studies in This Directory

### 1. Demand Forecasting for a Global Logistics Network
**File**: `demand_forecasting_logistics.md`

**Scale**: DHL-level, 10M+ parcels/day, 220+ countries

**Core Topics**:
- Hierarchical time series forecasting (global → country → depot → route)
- Feature engineering for temporal data: lag features, rolling statistics, calendar features
- Model progression: seasonal naive → ARIMA/Prophet → LightGBM → TFT (Temporal Fusion Transformer)
- Hierarchical reconciliation methods (bottom-up, top-down, MinT)
- Handling demand shocks (COVID-19 example)
- Walk-forward backtesting methodology
- Cold start for new routes/depots

**Key Interview Topics**:
- Hierarchical forecasting and reconciliation
- MAPE vs SMAPE and when each breaks
- How to handle structural breaks in time series
- Walk-forward validation vs train/test split

**Best For**: ML Engineer, Data Scientist, Applied Scientist roles at logistics, e-commerce, retail companies.

---

### 2. Real-Time Delivery ETA Prediction
**File**: `delivery_time_prediction.md`

**Scale**: Hundreds of millions of shipments, real-time predictions at scan events

**Core Topics**:
- Dynamic ETA update: prediction refreshed at every package scan event
- Quantile regression for uncertainty quantification (P10, P50, P90)
- Conformal prediction introduction
- Feature engineering: zone pairs, historical OD (origin-destination) statistics, network load factors
- Event-driven architecture: Kafka → prediction service → customer API
- Distribution shift handling: seasonality, new routes, disruptions
- Survival analysis formulation for censored delivery times

**Key Interview Topics**:
- How to handle uncertainty in ML predictions for customer-facing applications
- Difference between point estimates and prediction intervals
- How to update ML predictions in real time using streaming events
- A/B testing ML models in production

**Best For**: ML Engineer, Senior MLE, Staff Engineer at e-commerce, logistics, delivery platforms.

---

### 3. Automated Document Processing with LLMs
**File**: `document_processing_llm.md`

**Scale**: 500K+ documents/day (invoices, customs forms, bills of lading)

**Core Topics**:
- Architecture comparison: OCR+Rules vs OCR+ML NER vs Multimodal LLM vs Fine-tuned Document AI
- Hybrid routing: cheap fine-tuned model for standard templates, GPT-4V for complex documents
- Structured output design: JSON schema, Pydantic validation, OpenAI function calling
- Confidence scoring for human-in-the-loop routing
- Cost optimization: batching, caching, model routing
- GDPR compliance: PII handling in document processing pipelines
- Field-level vs document-level evaluation metrics

**Key Interview Topics**:
- How to validate LLM-extracted structured data
- Multi-language document handling
- Cost vs accuracy tradeoffs in LLM applications
- Building human-in-the-loop ML pipelines

**Best For**: MLE/Applied Scientist at document AI, fintech, legal tech, logistics companies. Excellent for LLM product engineering roles.

---

### 4. RAG-Powered Customer Support Chatbot
**File**: `customer_support_chatbot_rag.md`

**Scale**: 2M customer contacts/month, 15 languages

**Core Topics**:
- Full RAG pipeline: multilingual embeddings (BGE-M3), hybrid search (BM25 + dense), cross-encoder reranking
- Intent classification and entity extraction (tracking number, issue type)
- Routing logic: tracking API vs RAG vs human escalation
- Conversation state management: multi-turn context, entity tracking
- Safety and guardrails: off-topic detection, PII masking, confidence thresholding
- Evaluation: RAGAS framework, containment rate, CSAT, escalation rate
- Knowledge base management: chunking strategy, incremental updates

**Key Interview Topics**:
- When to escalate to a human agent (confidence thresholding, topic detection)
- Hybrid search vs pure vector search tradeoffs
- Chunking strategies for RAG
- Evaluating chatbot quality without labeled data

**Best For**: MLE/Engineer at any customer-facing AI product. Essential for LLM/RAG system design interviews.

---

### 5. Anomaly Detection in Supply Chain Operations
**File**: `anomaly_detection_supply_chain.md`

**Scale**: Real-time monitoring of global logistics network

**Core Topics**:
- Anomaly types: volume spikes, delivery failures, fraud, equipment failure
- Unsupervised vs semi-supervised approaches
- Statistical methods: Z-score, IQR, CUSUM
- ML methods: Isolation Forest, LOF, One-Class SVM
- Deep learning: LSTM Autoencoder, Variational Autoencoder
- Time series specific: STL decomposition, Prophet residuals
- Ensemble with meta-model for reduced false positives
- Real-time (Kafka + Flink) vs batch detection
- Alert fatigue: severity scoring, runbooks, escalation matrix
- Evaluation without ground truth labels

**Key Interview Topics**:
- Evaluating anomaly detection without labels (precision@K, user feedback loops)
- Alert fatigue and how to address it
- STL decomposition for separating trend, seasonality, residual
- Real-time streaming anomaly detection architecture

**Best For**: MLE/Data Scientist at companies with operational monitoring requirements (fintech, logistics, manufacturing, cloud infrastructure).

---

### 6. ML-Augmented Route Optimization for Last-Mile Delivery
**File**: `route_optimization_ml.md`

**Scale**: 1000+ delivery routes per city, daily optimization

**Core Topics**:
- Classical OR: Vehicle Routing Problem (VRP), Clarke-Wright savings algorithm
- ML augmentation: traffic prediction, delivery success prediction, demand forecasting
- Graph Neural Networks for routing: nodes as locations, edges as road segments
- Pointer Networks and attention for sequence optimization
- Reinforcement Learning formulation: state/action/reward definition, simulated training environment
- Production integration: OR solver + ML features pipeline
- Driver acceptance rate as a key business metric
- Real-time disruption handling: failed deliveries, traffic rerouting

**Key Interview Topics**:
- Combining ML with classical optimization (ML for features, OR for optimization)
- Vehicle Routing Problem fundamentals
- When to use Reinforcement Learning vs supervised learning
- How to handle real-world constraints in optimization problems

**Best For**: MLE/Research Scientist at delivery platforms, ride-hailing, logistics, supply chain companies.

---

### 7. LLM-Powered Internal Knowledge Assistant
**File**: `llm_internal_knowledge_assistant.md`

**Scale**: 50K employees, multi-source knowledge base (Confluence, SharePoint, Slack, Jira)

**Core Topics**:
- Document-level access control in RAG systems
- Multi-source retrieval architecture
- Incremental knowledge base updates: delta sync, real-time for critical documents
- Source attribution and citation generation
- Freshness scoring: prefer recent documents in ranking
- Multi-modal support: text, PDFs, presentations, tables
- Evaluation: RAGAS on curated QA pairs, business metrics (time-to-answer)
- Handling conflicting information from multiple sources
- ROI measurement for internal knowledge tools

**Key Interview Topics**:
- How to enforce access control in a RAG system (filter before retrieval or after?)
- Handling conflicting information across sources
- Keeping knowledge base current (staleness problem)
- Measuring ROI of an internal AI tool

**Best For**: MLE/Engineer building internal enterprise AI tools. Strong signal for Senior/Staff MLE roles.

---

## Key Themes Across All Case Studies

### Theme 1: The ML Hierarchy of Needs
Before modeling, you need:
1. Reliable data pipelines
2. A clear problem formulation
3. A sensible baseline
4. Correct evaluation methodology

Too many candidates jump to "I'd use a Transformer" without establishing the baseline or evaluation strategy. These case studies always start with a baseline and business context.

### Theme 2: The Baseline is Sacred
Every case study establishes a strong baseline before introducing ML. Common baselines:
- **Forecasting**: same-week-last-year (seasonal naive)
- **ETA**: carrier-stated delivery window
- **Document processing**: manual data entry (cost and time)
- **Chatbot**: FAQ keyword search
- **Anomaly detection**: static threshold alerts
- **Route optimization**: greedy nearest-neighbor

You must always be able to answer: "What's the baseline, and by how much does your ML model beat it?"

### Theme 3: Evaluation Must Match Business Goals
RMSE alone is not a business metric. Each case study shows:
- Technical metrics (RMSE, MAPE, F1, AUC)
- Business metrics (SLA adherence, contact center volume, cost per delivery)
- How to connect the two

### Theme 4: Production is Where Systems Fail
Every case study has a "Production Concerns" section covering:
- Data pipeline reliability
- Model serving latency and throughput
- Monitoring and drift detection
- Retraining triggers
- Failure modes and fallbacks

### Theme 5: Cost vs Performance Tradeoffs
In enterprise settings, the best model is not the most accurate model — it is the one that achieves sufficient accuracy at acceptable cost. This appears in:
- LLM routing (cheap model for easy docs, expensive for hard docs)
- Anomaly detection (false positive cost vs missed anomaly cost)
- Route optimization (compute time vs solution quality)

### Theme 6: Human-in-the-Loop is Not Failure
A system that routes 15% of cases to human review is a successful ML system — it automated 85% of work that was previously 100% manual. Candidates who treat human review as a failure mode misunderstand production ML.

---

## Recommended Study Order

**If you have 1 week**:
Day 1: Demand Forecasting (foundational time series concepts)
Day 2: Delivery ETA Prediction (real-time ML, uncertainty)
Day 3: Customer Support Chatbot RAG (RAG fundamentals)
Day 4: Document Processing LLM (LLM applications, cost tradeoffs)
Day 5: Anomaly Detection (unsupervised ML, monitoring)
Day 6: Route Optimization (OR + ML, graph models)
Day 7: Internal Knowledge Assistant (enterprise RAG, access control)

**If you have 2 days**:
Day 1: Demand Forecasting + Customer Support Chatbot RAG
Day 2: Delivery ETA Prediction + Document Processing LLM

**If you have 2 hours**:
Read Delivery ETA Prediction (covers the most universal concepts: real-time ML, uncertainty, evaluation, production concerns)

---

## How to Approach a System Design Interview

### The First 5 Minutes: Requirements Clarification
Before drawing any architecture, ask:
- What is the scale? (users, data volume, query rate)
- What are the latency requirements? (real-time vs batch)
- What are the evaluation criteria? (business metric, not just ML metric)
- What data is available?
- What is the current baseline or existing solution?

### The Architecture Discussion
Structure your answer in layers:
1. Data layer: sources, ingestion, storage
2. Feature layer: feature engineering, feature store
3. Modeling layer: model selection, training pipeline
4. Serving layer: inference, caching, APIs
5. Monitoring layer: drift detection, alerting, retraining

### The Deep Dive
Be ready to go deep on any component. Common deep dives:
- "Walk me through your feature engineering approach"
- "How would you handle class imbalance?"
- "How do you detect model drift?"
- "How would you scale this to 10x the load?"

### Closing: Business Impact
Always tie back to business impact. "This model reduces forecast error by 15%, which translates to $X million in reduced over-staffing costs per year."

---

## Glossary of Terms Used Across Case Studies

| Term | Definition |
|------|-----------|
| MAPE | Mean Absolute Percentage Error. Fails when actuals are near zero. |
| SMAPE | Symmetric Mean Absolute Percentage Error. More robust than MAPE for near-zero values. |
| Walk-forward validation | Time series cross-validation that respects temporal order. No data leakage. |
| Hierarchical forecasting | Forecasting at multiple levels of aggregation (global → country → depot) and reconciling. |
| MinT reconciliation | Minimum Trace reconciliation. Optimal linear combination of base forecasts for hierarchical consistency. |
| Quantile regression | Predicts a specific quantile of the target distribution, not just the mean. Used for prediction intervals. |
| Conformal prediction | Distribution-free framework for generating calibrated prediction intervals. |
| Isolation Forest | Tree-based anomaly detection. Isolates anomalies by randomly partitioning feature space. |
| CUSUM | Cumulative Sum control chart. Detects small shifts in the mean of a time series. |
| STL decomposition | Seasonal-Trend decomposition using Loess. Separates time series into trend, seasonal, residual. |
| LSTM Autoencoder | Autoencoder using LSTM layers. Trained on normal data; anomalies have high reconstruction error. |
| VRP | Vehicle Routing Problem. NP-hard combinatorial optimization problem for routing a fleet of vehicles. |
| RAG | Retrieval-Augmented Generation. LLM answers questions grounded in retrieved documents. |
| RAGAS | RAG Assessment framework. Automated evaluation of faithfulness, relevance, groundedness. |
| Hybrid search | Combining dense vector search (semantic) with sparse BM25 search (keyword). |
| Cross-encoder reranker | Model that scores query-document pairs jointly. More accurate than bi-encoder but slower. |
| Pointer Network | Neural network that outputs a sequence by pointing to positions in its input. Used for TSP/routing. |
| GNN | Graph Neural Network. Operates on graph-structured data (nodes and edges). |
| Conformal prediction | Post-hoc calibration method that provides distribution-free coverage guarantees. |
| LayoutLM | Microsoft's document AI model. Pre-trained on document layout, text, and image features jointly. |
| BGE-M3 | BAAI General Embedding model. State-of-the-art multilingual embedding model. |
| TFT | Temporal Fusion Transformer. Attention-based model for multi-horizon time series forecasting. |
| N-BEATS | Neural Basis Expansion Analysis for Time Series. Pure deep learning forecasting model. |
| MinT | Minimum Trace. Optimal hierarchical reconciliation method. |

---

## Contributing New Case Studies

When adding a new case study to this directory, follow this template structure:

```
1. Business Problem (who, what, why, scale, impact of failure)
2. Data Description (sources, volume, quality issues)
3. Problem Formulation (ML task type, inputs, outputs, constraints)
4. Modeling Approach (baseline → classical → ML → DL → ensemble)
5. Feature Engineering (specific feature definitions with pseudocode)
6. Architecture Diagram (Mermaid flowchart)
7. Evaluation (technical metrics + business metrics + validation strategy)
8. Production Concerns (serving, monitoring, retraining, failure modes)
9. Failure Modes and Mitigations
10. Interview Questions with Model Answers
```

Minimum length: 500 lines. Each section must be substantive, not a placeholder.

---

*Last updated: February 2026*
*Repository: AI/ML/LLM Interview Preparation*
