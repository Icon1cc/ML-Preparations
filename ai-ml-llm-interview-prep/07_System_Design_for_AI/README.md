# 07 System Design for AI

## Overview

**System design interviews test your ability to build scalable, reliable ML systems.** This goes beyond model training - you need to think about data pipelines, infrastructure, monitoring, cost, and user experience.

## 📚 Contents

1. [End-to-End ML System Design](./end_to_end_ml_system_design.md) ⭐⭐⭐
2. [LLM Application Design](./llm_application_design.md) ⭐⭐⭐
3. [Interview Walkthroughs](./interview_walkthroughs.md) ⭐⭐⭐
4. [High-Throughput Inference](./high_throughput_inference.md)
5. [Streaming Pipelines](./streaming_pipelines.md)
6. [Data Architecture Patterns](./data_architecture_patterns.md)
7. [Cost vs Latency Tradeoffs](./cost_vs_latency_tradeoffs.md)
8. [Reliability and Availability](./reliability_and_availability.md)
9. [Multi-Tenant AI Systems](./multi_tenant_ai_systems.md)

## 🎯 Interview Format

**Typical structure (45-60 min):**

1. **Requirements (10 min):**
   - Clarify functional and non-functional requirements
   - Understand scale, latency, budget

2. **High-level design (15 min):**
   - Draw boxes and arrows
   - Data flow
   - Major components

3. **Deep dive (15 min):**
   - Pick 2-3 components to detail
   - Discuss tradeoffs
   - Database choices, caching, etc.

4. **Discussion (10 min):**
   - Failure modes
   - Monitoring
   - Scaling
   - Cost estimation

## 🎓 Key Framework: RADIO

Use this framework for every system design question:

### **R - Requirements**

**Functional:**
- What should the system do?
- Inputs and outputs?
- Use cases?

**Non-functional:**
- **Scale:** QPS? Users? Data volume?
- **Latency:** <100ms? <1s? Batch OK?
- **Availability:** 99.9%? 99.99%?
- **Consistency:** Real-time or eventual?
- **Budget:** Cost constraints?

### **A - Architecture**

**High-level components:**
- Data sources
- Data pipeline
- Feature store (optional)
- Model training
- Model serving
- Monitoring

### **D - Deep Dive**

**Pick 2-3 components:**
- Database choice
- Caching strategy
- Model serving (batch vs real-time)
- Feature engineering

### **I - Issues**

**What could go wrong?**
- Model failure
- Data pipeline failure
- Infrastructure failure
- Data drift

**Mitigation:**
- Fallbacks
- Circuit breakers
- Monitoring
- Alerts

### **O - Optimization**

**How to improve:**
- Caching
- Load balancing
- Horizontal scaling
- Cost optimization

## Common System Design Questions

### 1. Design a Recommendation System
**E.g., Netflix movie recommendations**

**Key considerations:**
- Collaborative filtering vs content-based
- Cold-start problem (new users/items)
- Real-time vs batch
- Personalization
- Diversity and exploration

### 2. Design a Search Ranking System
**E.g., Google search, e-commerce search**

**Key considerations:**
- Query understanding
- Retrieval (candidate generation)
- Ranking (ML model)
- Latency constraints (<100ms)
- Relevance vs business metrics

### 3. Design an Ad Serving System
**E.g., Facebook ads**

**Key considerations:**
- Real-time bidding
- Click-through rate prediction
- Budget pacing
- Exploration-exploitation
- Auction mechanism

### 4. Design a Fraud Detection System
**E.g., Credit card fraud**

**Key considerations:**
- Real-time scoring (<100ms)
- Imbalanced data (fraud is rare)
- False positive cost
- Feature engineering (velocity, location)
- Rule-based + ML

### 5. Design a Feed Ranking System
**E.g., Facebook news feed, Twitter timeline**

**Key considerations:**
- Personalization
- Engagement prediction
- Freshness vs relevance
- Diversity
- Real-time updates

### 6. Design an LLM Application
**E.g., ChatGPT for customer support**

**Key considerations:**
- RAG vs fine-tuning
- Prompt engineering
- Context management
- Cost (LLM calls expensive)
- Latency
- Hallucination mitigation

## Example Walkthrough: Design YouTube Video Recommendations

### 1. Requirements

**Functional:**
- Recommend videos to users based on watch history
- Personalized for each user
- Update recommendations frequently

**Non-functional:**
- **Scale:** 2B users, 500M daily active
- **Latency:** <200ms for API
- **Availability:** 99.95%
- **Budget:** Optimize inference cost

**Clarifying questions:**
- What's the goal? (Watch time? Click rate?)
- Real-time personalization needed?
- Cold-start for new users?

### 2. High-Level Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│          API Gateway                │
│     (Load Balancer, Auth)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Recommendation Service             │
│   - Fetch user profile               │
│   - Get candidates                   │
│   - Rank candidates                  │
│   - Apply business logic             │
└──────────────┬───────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐  ┌──────────────┐
│ Candidate    │  │  Ranking     │
│ Generation   │  │  Model       │
│ Service      │  │  Service     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ User Profile │  │  Model       │
│ Store        │  │  Store       │
│ (Redis)      │  │  (S3)        │
└──────────────┘  └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│    Data Pipeline (Offline)           │
│    - Watch history processing        │
│    - Feature engineering             │
│    - Model training                  │
└──────────────────────────────────────┘
```

### 3. Component Deep Dive

**Candidate Generation:**
- **Purpose:** Filter 1B videos → 1K candidates
- **Methods:**
  - Collaborative filtering (users who watched A also watched B)
  - Content-based (similar to recently watched)
  - Trending/popular
- **Storage:** Pre-compute daily, store in Redis
- **Latency:** <50ms (cached)

**Ranking Model:**
- **Purpose:** Rank 1K candidates → top 50
- **Features:**
  - User: age, location, watch history
  - Video: title, category, upload date, engagement
  - Context: time of day, device
- **Model:** XGBoost or neural network
- **Serving:** TensorFlow Serving, <100ms
- **Update:** Retrain daily

**User Profile Store:**
- **Storage:** Redis (fast reads)
- **Data:** Recent 100 videos, preferences, demographics
- **Update:** Real-time (Kafka stream)

### 4. Scaling

**For 500M DAU:**
- **QPS:** ~50K requests/sec (peak)
- **Candidate generation:** 100 instances (500 QPS each)
- **Ranking:** 200 instances (250 QPS each)
- **Redis:** Sharded across 50 nodes

**Caching:**
- Cache recommendations for 5 minutes
- Cache hit rate ~60% → effective QPS = 20K

### 5. Monitoring

**Metrics to track:**
- **Business:** Click-through rate, watch time
- **ML:** Model prediction distribution, feature drift
- **System:** Latency (p50, p95, p99), error rate, throughput

**Alerts:**
- Latency > 500ms
- Error rate > 1%
- Click-through rate drops >10%

### 6. Failure Modes & Mitigation

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Ranking model down | Can't personalize | Fallback to popular/trending |
| Redis down | Can't fetch profiles | Use default profile, degraded experience |
| Candidate service slow | Timeout | Cache candidates, reduce candidate set |
| Model accuracy drops | Poor recommendations | Automated retraining, A/B testing |

---

## Key Principles

### 1. Start Simple, Then Scale

**Don't over-engineer day 1:**
- Start with batch predictions
- Add real-time later if needed
- Use managed services (SageMaker, BigQuery)

**Scale when you have to:**
- Premature optimization is root of all evil
- Measure first, optimize later

### 2. Separate Concerns

**Clean separation:**
- Data pipeline (Airflow)
- Training (SageMaker)
- Serving (REST API)
- Monitoring (separate service)

**Benefits:**
- Independent scaling
- Easier debugging
- Team can own components

### 3. Think About Cost

**ML can be expensive:**
- **Training:** GPUs cost $1-10/hour
- **Inference:** 1M predictions/day × $0.0001 = $100/day
- **Storage:** Features, models, logs

**Optimize:**
- Batch where possible
- Cache aggressively
- Use smaller models if acceptable
- Auto-scaling

### 4. Plan for Failure

**Everything fails:**
- Models serve stale predictions
- APIs timeout
- Databases go down

**Design for resilience:**
- Fallback mechanisms
- Circuit breakers
- Graceful degradation
- Monitoring and alerts

### 5. Iterate and Measure

**Don't guess:**
- A/B test changes
- Measure impact on business metrics
- Iterate based on data

---

## Interview Tips

### Do's ✅

1. **Ask clarifying questions** - Requirements are intentionally vague
2. **Think out loud** - Show your thought process
3. **Draw diagrams** - Visual aids help
4. **Discuss tradeoffs** - No perfect solution
5. **Consider scale** - Numbers matter
6. **Talk about monitoring** - Production is important
7. **Mention cost** - Shows business awareness

### Don'ts ❌

1. **Don't jump to solution** - Understand problem first
2. **Don't ignore scale** - 100 users ≠ 100M users
3. **Don't forget failures** - Things break
4. **Don't over-engineer** - Start simple
5. **Don't use buzzwords** - Explain clearly
6. **Don't ignore latency** - Real-time matters

---

## Practice Questions

1. Design a real-time fraud detection system
2. Design a personalized email recommendation system
3. Design a content moderation system for user-generated content
4. Design a dynamic pricing system for ride-sharing
5. Design a chatbot with RAG for customer support
6. Design a demand forecasting system for e-commerce
7. Design a real-time bidding system for ads
8. Design a music recommendation system (Spotify)
9. Design a job recommendation system (LinkedIn)
10. Design a translation service (Google Translate)

---

**Most Important:** [End-to-End ML System Design](./end_to_end_ml_system_design.md)

**Most Practical:** [Interview Walkthroughs](./interview_walkthroughs.md)

**Next:** Practice with [Case Studies](../08_Case_Studies/README.md)
