# Common Interview Traps

> **Learn from others' mistakes** - Avoid these common pitfalls that candidates fall into

## Overview

This cheat sheet covers the **most common mistakes** candidates make in AI/ML/Data Science interviews and how to avoid them. These are patterns observed across hundreds of technical interviews.

---

## 🚫 Part 1: Technical Answer Traps

### Trap 1: "XGBoost is Always Best"

**❌ What candidates say:**
"I'll use XGBoost because it's the best algorithm."

**Why it's wrong:**
- No algorithm is universally best
- Ignores problem constraints (latency, interpretability, data size)
- Shows lack of nuance

**✅ Better answer:**
"For tabular data, XGBoost often performs well, but let me consider the constraints:
- **Interpretability**: If we need explainability, logistic regression or decision trees are better
- **Inference latency**: If we need <10ms, simpler models like logistic regression are better
- **Dataset size**: For <1K samples, simpler models with regularization might generalize better

For this specific problem with [details], I'd choose [algorithm] because [reasons]. I'd validate this choice by comparing against [baseline] using [metric] via cross-validation."

---

### Trap 2: Forgetting to Ask Clarifying Questions

**❌ What candidates do:**
Jump straight into solution without understanding the problem

**Why it's wrong:**
- Real-world problems are ambiguous
- Shows lack of practical experience
- Miss critical constraints

**✅ Better approach:**

**Always ask:**
1. **Data questions:**
   - "How much data do we have? What's the quality?"
   - "What features are available?"
   - "Is data labeled?"

2. **Business questions:**
   - "What's the business objective?"
   - "What's more costly: false positives or false negatives?"
   - "What's the current baseline/solution?"

3. **Constraint questions:**
   - "What's the latency requirement?"
   - "Any interpretability requirements?"
   - "What's the budget/resources available?"
   - "Timeline to deployment?"

**Example:**
"Before I propose a solution, let me clarify a few things:
- How much historical data do we have?
- What's more critical: precision or recall?
- Do we need real-time predictions or batch?
- Any regulatory requirements for interpretability?"

---

### Trap 3: Not Discussing Tradeoffs

**❌ What candidates say:**
"We'll use a deep neural network to get maximum accuracy."

**Why it's wrong:**
- Every decision has tradeoffs
- Shows lack of production experience
- Interviewers want to see decision-making process

**✅ Better answer:**
"Let me think through the tradeoffs:

**Option 1: Deep Neural Network**
- ✅ Pros: Can capture complex patterns, potential for highest accuracy
- ❌ Cons: Needs large data, long training, hard to interpret, slow inference

**Option 2: Gradient Boosting (XGBoost)**
- ✅ Pros: Great for tabular data, faster training, built-in feature importance
- ❌ Cons: Slower than linear models, more hyperparameters to tune

**Option 3: Logistic Regression**
- ✅ Pros: Fast, interpretable, works with less data
- ❌ Cons: Assumes linear boundaries, may underfit

Given [problem constraints], I'd recommend [choice] because [reasoning]."

---

### Trap 4: Ignoring Data Leakage

**❌ What candidates do:**
- Scale data before splitting train/test
- Use features that wouldn't be available at prediction time
- Don't mention data leakage at all

**Why it's wrong:**
- Data leakage is a top cause of model failure in production
- Shows lack of attention to detail
- Critical for real-world ML

**✅ Better approach:**

**Proactively mention:**
"I want to be careful about data leakage. I'll:
1. Split data BEFORE any preprocessing
2. Fit scalers/encoders on training data only
3. Check each feature: 'Would this be available at prediction time?'
4. Use time-based splits for time series data
5. Watch for target leakage in feature engineering"

**Example:**
"For this customer churn problem, I'd avoid features like 'cancellation_email_sent' even if highly predictive, because it's only available AFTER the decision to churn. Instead, I'd use historical behavior patterns."

---

### Trap 5: Not Starting with a Baseline

**❌ What candidates do:**
Jump straight to complex models

**Why it's wrong:**
- Need a performance floor for comparison
- Simple baselines can be surprisingly good
- Shows lack of systematic approach

**✅ Better approach:**
"I'd start with a simple baseline to establish a performance floor:

**Baseline options:**
- **Regression**: Predict mean/median (gives RMSE baseline)
- **Classification**: Majority class prediction (gives accuracy baseline)
- **Time series**: Naive forecast (yesterday's value, seasonal naive)

Then I'd build increasingly complex models:
1. **Baseline**: Logistic regression / Linear regression
2. **Intermediate**: Random Forest
3. **Advanced**: XGBoost / Neural Network

This lets me understand:
- Is the problem learnable?
- Do I need complexity?
- What's the ROI of each increment in complexity?"

---

### Trap 6: Choosing Accuracy for Imbalanced Data

**❌ What candidates say:**
"I'll optimize for accuracy."

**Why it's wrong:**
- Accuracy is misleading with imbalance (99% accuracy by always predicting majority class)
- Shows fundamental misunderstanding
- Very common trap!

**✅ Better answer:**
"For this [fraud detection / rare disease / anomaly detection] problem with class imbalance, accuracy is inappropriate. Instead:

**Better metrics:**
- **Precision** if false positives are costly
- **Recall** if false negatives are costly
- **F1-Score** for balanced view
- **PR-AUC** (better than ROC-AUC for imbalanced data)

For fraud detection specifically, I'd optimize for **high recall** (catch most fraud) while maintaining acceptable **precision** (don't overwhelm investigators with false alarms). I'd use **PR-AUC** as the primary evaluation metric."

---

### Trap 7: Forgetting About Production

**❌ What candidates say:**
"Once we train the model, we're done."

**Why it's wrong:**
- Training is only the beginning
- Production is where value is created
- Shows lack of end-to-end thinking

**✅ Better answer:**
"Beyond training, I'd consider:

**Deployment:**
- How do we serve predictions? (REST API, batch, streaming)
- What's the infrastructure? (cloud, on-prem)
- Latency requirements? (real-time vs batch)

**Monitoring:**
- Track model performance (accuracy, latency)
- Detect data drift (feature distributions changing)
- Monitor for concept drift (relationship between features and target changing)
- Business metrics (did we actually reduce costs? improve conversions?)

**Maintenance:**
- How often to retrain?
- A/B testing new models?
- Rollback strategy if new model underperforms?

**Cost:**
- Inference cost per prediction
- Storage costs (features, models, logs)
- Retraining costs"

---

## 🚫 Part 2: LLM/Modern AI Traps

### Trap 8: "RAG is Always Better than Fine-Tuning"

**❌ What candidates say:**
"We should use RAG because it's better than fine-tuning."

**Why it's wrong:**
- Each approach has use cases
- Often best solution is hybrid
- Shows lack of nuance

**✅ Better answer:**
"Let me compare RAG vs fine-tuning for this use case:

**RAG:**
- ✅ Up-to-date information (just update document store)
- ✅ Interpretable (can show sources)
- ✅ Cost-effective
- ❌ Retrieval quality dependency
- ❌ Added latency (retrieval + generation)

**Fine-tuning:**
- ✅ Knowledge baked into model
- ✅ No retrieval latency
- ✅ Can learn style and tone
- ❌ Static knowledge (needs retraining for updates)
- ❌ Expensive
- ❌ Less interpretable

**For this problem:**
[Analyze requirements and recommend approach]

Often, the best approach is **hybrid**: Fine-tune for style/tone/domain expertise, use RAG for factual, up-to-date information."

---

### Trap 9: Not Discussing Hallucinations

**❌ What candidates do:**
Design LLM systems without mentioning hallucinations

**Why it's wrong:**
- Hallucinations are THE critical issue with LLMs
- Shows lack of real-world LLM experience
- Risk mitigation is essential

**✅ Better approach:**
"A key risk with LLMs is hallucination—confidently generating false information. I'd mitigate this by:

**Detection:**
- Low probability tokens suggest hallucination
- Fact-checking against knowledge base
- Prompt model to express uncertainty

**Prevention:**
- Use RAG with citations (lets users verify)
- Lower temperature for factual tasks
- Clear system prompts: 'Only use provided context'
- Fact-checking layer for critical applications

**Communication:**
- Show sources/citations
- Confidence scores
- Clear disclaimers about potential errors
- Human-in-the-loop for high-stakes decisions"

---

### Trap 10: Treating LLMs as Magic

**❌ What candidates say:**
"We'll just use GPT-4 and it will solve everything."

**Why it's wrong:**
- LLMs are tools, not magic
- Need proper engineering and constraints
- Shows lack of critical thinking

**✅ Better answer:**
"LLMs are powerful but have limitations:

**Strengths:**
- Strong language understanding
- Few-shot learning
- Flexible for diverse tasks

**Limitations:**
- Hallucinations (make up information)
- No real-time knowledge (knowledge cutoff)
- Context window limits
- Can be expensive
- Latency (especially for large models)

**For this problem, I'd:**
1. Use LLM for [specific capability]
2. Combine with [structured systems] for [specific needs]
3. Add [safeguards] for [risks]
4. Monitor [metrics] to ensure quality

This is a **systems engineering** problem, not just 'plug in GPT-4.'"

---

## 🚫 Part 3: Communication Traps

### Trap 11: Too Much Jargon

**❌ What candidates say:**
"We'll use a bidirectional encoder with multi-head self-attention and layer normalization, training with AdamW optimizer using linear warmup schedule and polynomial decay..."

**Why it's wrong:**
- Not all interviewers are deeply technical
- Shows poor communication skills
- Can't explain to stakeholders

**✅ Better answer:**
Start with intuition, add technical depth if needed:

"I'd use a transformer-based model—think of it like a sophisticated pattern matching system that can understand context bidirectionally. [Pause to gauge understanding]

If you'd like, I can dive into specifics like the attention mechanism... [Wait for confirmation]"

**Rule: Explain like you're talking to:**
- A technical PM for basic explanation
- A fellow engineer for detailed discussion
- An executive for business impact

---

### Trap 12: Not Quantifying Impact

**❌ What candidates say:**
"This model will improve predictions."

**Why it's wrong:**
- Vague, no business value
- Can't justify investment
- Shows lack of business thinking

**✅ Better answer:**
"Let me estimate the business impact:

**Current state:**
- Baseline accuracy: 75%
- False positives: 1,000/day at $10 each = $10K/day
- False negatives: 100/day at $500 each = $50K/day
- Total cost: $60K/day

**With proposed model:**
- Improved accuracy: 85%
- False positives: 600/day = $6K/day (-40%)
- False negatives: 60/day = $30K/day (-40%)
- Total cost: $36K/day

**Annual savings: $8.8M**

**Investment:**
- Development: 2 engineers × 3 months = ~$75K
- Infrastructure: ~$5K/month = $60K/year
- Total: ~$135K

**ROI: 65x in first year**"

---

### Trap 13: Not Admitting Uncertainty

**❌ What candidates do:**
- Make up answers they don't know
- Pretend to be certain when unsure
- Avoid saying "I don't know"

**Why it's wrong:**
- Interviewers can tell
- Integrity matters
- Better to acknowledge gaps

**✅ Better approach:**
"I'm not deeply familiar with [specific topic], but here's my understanding... [share what you know]

If this were a real project, I'd:
1. Research [specific resources]
2. Consult with [experts]
3. Run experiments to validate

Can we perhaps discuss [related topic you know well] instead?"

**Or:**
"That's a great question. I haven't worked with [X] directly, but I understand it's similar to [Y], which I have experience with. [Show transferable knowledge]"

---

## 🚫 Part 4: System Design Traps

### Trap 14: Skipping Requirements

**❌ What candidates do:**
Jump straight to architecture diagram

**Why it's wrong:**
- No system is designed in vacuum
- Requirements drive decisions
- Shows lack of systematic thinking

**✅ Better approach:**
"Let me start by clarifying requirements:

**Functional Requirements:**
- What should the system do?
- What are the inputs/outputs?
- What's the expected behavior?

**Non-Functional Requirements:**
- **Scale**: QPS? Users? Data volume?
- **Latency**: What's acceptable response time?
- **Availability**: 99.9%? 99.99%?
- **Consistency**: Real-time or eventual consistency OK?

**Constraints:**
- **Budget**: Cost limits?
- **Team**: Team size and skills?
- **Timeline**: When does this need to ship?

[Only THEN draw architecture]"

---

### Trap 15: Overengineering

**❌ What candidates do:**
Design systems with Kubernetes, microservices, distributed databases, message queues... for a simple problem

**Why it's wrong:**
- Premature optimization
- Adds complexity without benefit
- Shows lack of pragmatism

**✅ Better approach:**
"Let me think about scale:

**Current needs:**
- 100 requests/day
- <5s latency acceptable
- Small dataset (10K rows)

**Solution:**
- Simple Flask API + SQLite
- Deploy on single EC2 instance
- Total cost: $20/month

**If we scale to 10K requests/day:**
- Upgrade to PostgreSQL
- Add caching layer
- Multiple API instances behind load balancer

**If we scale to 1M requests/day:**
- Then we'd consider:
  - Distributed database
  - Microservices
  - Message queues
  - Kubernetes

Start simple, scale when needed. Don't prematurely optimize."

---

### Trap 16: Ignoring Failure Modes

**❌ What candidates do:**
Design happy-path only

**Why it's wrong:**
- Systems fail in production
- Reliability is critical
- Shows lack of production experience

**✅ Better approach:**
"Let me think about failure modes:

**What could go wrong?**
1. **Model failure**: Model returns errors/garbage
   - Mitigation: Fallback to rule-based system, circuit breaker pattern

2. **Data pipeline failure**: Stale features
   - Mitigation: Data validation checks, alerts, graceful degradation

3. **Infrastructure failure**: Server down
   - Mitigation: Multi-AZ deployment, health checks, auto-scaling

4. **Upstream dependency failure**: External API down
   - Mitigation: Caching, timeouts, retries with exponential backoff

5. **Data drift**: Model accuracy degrades
   - Mitigation: Monitoring, automated retraining, A/B testing

**Monitoring & Alerts:**
- SLOs: 99.9% uptime, <2s p99 latency
- Alerts: Error rate >1%, latency >5s
- Dashboards: Real-time metrics"

---

## 💡 General Interview Strategies

### Do's ✅

1. **Think out loud** - Show your reasoning process
2. **Ask clarifying questions** - Ambiguity is intentional
3. **Start simple** - Baseline before complexity
4. **Discuss tradeoffs** - Every decision has pros/cons
5. **Consider production** - Training is just the start
6. **Quantify impact** - Connect to business value
7. **Admit gaps** - "I don't know" is better than making up
8. **Structure your answer** - Clear framework (requirements → approach → tradeoffs → evaluation)

### Don'ts ❌

1. **Don't jump to solutions** - Understand problem first
2. **Don't memorize answers** - Adapt to specific problem
3. **Don't ignore constraints** - Budget, latency, interpretability matter
4. **Don't use buzzwords** - Explain clearly
5. **Don't design for scale prematurely** - Start simple
6. **Don't forget data** - Garbage in, garbage out
7. **Don't treat models as magic** - They're tools with limitations
8. **Don't design happy-path only** - Plan for failures

---

## 📝 Answer Framework (Use This Structure)

### For Algorithm Selection Questions:

```
1. Clarify the problem
   - "Let me understand: we're predicting [X] using [Y] data..."

2. Ask about constraints
   - Data size? Latency? Interpretability? Budget?

3. Consider options
   - "I'm considering [A], [B], and [C]..."

4. Compare tradeoffs
   - "Option A is [pros] but [cons]..."

5. Make recommendation
   - "Given [constraints], I'd choose [X] because [reasons]"

6. Validation plan
   - "I'd validate using [CV strategy] with [metrics]"

7. Production considerations
   - "For deployment, I'd consider [monitoring/scaling/cost]..."
```

### For System Design Questions:

```
1. Clarify requirements
   - Functional: What should it do?
   - Non-functional: Scale, latency, availability

2. High-level architecture
   - Draw boxes: data sources → processing → model → serving

3. Drill down on critical components
   - "Let's talk about the serving layer..."

4. Discuss tradeoffs
   - "We could use [A] or [B], here's the tradeoff..."

5. Failure modes
   - "What could go wrong? [list] → mitigation [list]"

6. Monitoring
   - "I'd track [metrics], alert on [conditions]"

7. Scaling
   - "For 10x traffic, we'd need [changes]"
```

---

## 🎯 Red Flags to Avoid

**Interviewer thinks:**
- ❌ "This person has only read blogs, no real experience"
  - Fix: Discuss specific experiences, acknowledge limitations

- ❌ "They're following a memorized script"
  - Fix: Adapt to the specific problem, ask questions

- ❌ "They don't think about production"
  - Fix: Mention deployment, monitoring, costs unprompted

- ❌ "They can't communicate clearly"
  - Fix: Start with intuition, check understanding, avoid jargon

- ❌ "They're overconfident about things they don't understand"
  - Fix: Be honest about knowledge gaps

---

## Key Takeaways

1. **No silver bullets** - Every approach has tradeoffs
2. **Context matters** - Always clarify requirements and constraints
3. **Start simple** - Baseline → Intermediate → Advanced
4. **Think production** - Training is 10%, production is 90%
5. **Quantify impact** - Connect to business value
6. **Discuss failures** - What could go wrong? How to mitigate?
7. **Communication > Technical depth** - Explain clearly at appropriate level
8. **Admit uncertainty** - "I don't know" is OK

**Remember:** Interviewers are evaluating:
- **Technical knowledge** (Do you know your stuff?)
- **Problem-solving** (Can you think through problems?)
- **Communication** (Can you explain clearly?)
- **Judgment** (Do you make good decisions?)
- **Pragmatism** (Are you practical or dogmatic?)

**Best preparation:** Practice explaining your past projects using the frameworks above!

---

**Next:** Review [ML Algorithm Selection Guide](./ml_algorithm_selection_guide.md) for decision frameworks

**Back:** [09 Cheat Sheets README](./README.md)
