# Business Case Study: ML Product Prioritization & Resource Allocation

> **Domain:** Product Management / Business Strategy
> **Problem Type:** Decision Framework / Prioritization
> **Difficulty:** ⭐⭐ (Mid-Senior Level)
> **Interview Duration:** 45 minutes
> **Common in:** Product-focused ML roles, ML Product Manager interviews

## Table of Contents
1. [Business Context](#business-context)
2. [Problem Statement](#problem-statement)
3. [Prioritization Framework](#prioritization-framework)
4. [Evaluation Criteria](#evaluation-criteria)
5. [Opportunity Analysis](#opportunity-analysis)
6. [Resource Allocation](#resource-allocation)
7. [Recommendation & Rationale](#recommendation--rationale)
8. [Implementation Plan](#implementation-plan)
9. [Success Metrics](#success-metrics)
10. [Interview Approach](#interview-approach)

---

## Business Context

**Company:** E-commerce marketplace (similar to Amazon/eBay)
- 100M active users
- 2M sellers
- $20B GMV (Gross Merchandise Value) annually
- 500 engineers, 50 on ML team
- Mature ML infrastructure

**ML Team Structure:**
- 50 ML engineers (data scientists + ML engineers)
- 10 ML product managers
- Shared ML platform team
- Quarterly planning cycle

**Current Quarter Results:**
- Search relevance: Improved 5% (YoY)
- Fraud detection: Caught 15% more fraud
- Recommendation CTR: Up 8%

**Challenge:**
ML team has proposals for **12 new projects** for next quarter, but only has capacity for **4 projects**.

**Your Role:** Senior ML Product Manager tasked with prioritizing

---

## Problem Statement

**Task:** Select 4 out of 12 proposed ML projects to execute next quarter

**Constraints:**
- Budget: $2M for Q4 (external costs like labeling, compute)
- Team capacity: 40 engineer-quarters (10 engineers × 4 quarters, accounting for 20% on-call/maintenance)
- Timeline: Projects must show results within 6-12 months
- Company goals: Increase GMV 20%, reduce operating costs 10%

**Deliverable:**
1. Prioritized list of 4 projects with rationale
2. Resource allocation plan
3. Success metrics for each
4. What to *not* do and why

---

## The 12 Proposed Projects

### Category A: Revenue Growth

**Project 1: Visual Search**
- Let users search by uploading photos
- **Impact:** +5% conversion on fashion/home categories (15% of GMV = $3B)
- **Effort:** 12 engineer-quarters (large CV team needed)
- **Cost:** $400K (GPU infrastructure)
- **Timeline:** 9 months to launch
- **Risk:** Medium (unproven in our market)

**Project 2: Personalized Home Page**
- Replace static homepage with personalized product feed
- **Impact:** +3% overall conversion (+$600M GMV)
- **Effort:** 8 engineer-quarters
- **Cost:** $200K (A/B testing infrastructure)
- **Timeline:** 5 months to launch
- **Risk:** Low (proven by competitors)

**Project 3: Dynamic Pricing for Sellers**
- ML-powered price recommendations for sellers
- **Impact:** +2% GMV through better pricing (+$400M)
- **Effort:** 6 engineer-quarters
- **Cost:** $100K
- **Timeline:** 4 months to MVP
- **Risk:** Low (simple models work)

**Project 4: Voice Search**
- Alexa/Google Home integration for voice shopping
- **Impact:** +0.5% GMV (+$100M), mostly mobile users
- **Effort:** 10 engineer-quarters
- **Cost:** $300K (speech recognition licensing)
- **Timeline:** 8 months
- **Risk:** High (low adoption rates seen elsewhere)

---

### Category B: Cost Reduction / Efficiency

**Project 5: Automated Customer Service (Chatbot)**
- AI chatbot for tier-1 support queries
- **Impact:** $10M annual savings (reduce 500 support agents)
- **Effort:** 10 engineer-quarters
- **Cost:** $150K
- **Timeline:** 6 months to launch
- **Risk:** Medium (must maintain quality)

**Project 6: Fraud Detection V2 (Upgrade)**
- Improve existing fraud model with deep learning
- **Impact:** $15M annual fraud prevention (catching 20% more)
- **Effort:** 4 engineer-quarters (upgrade existing system)
- **Cost:** $50K
- **Timeline:** 3 months
- **Risk:** Low (proven tech)

**Project 7: Warehouse Demand Forecasting**
- Better inventory predictions to reduce overstock
- **Impact:** $5M annual savings (less warehousing cost)
- **Effort:** 5 engineer-quarters
- **Cost:** $75K
- **Timeline:** 4 months
- **Risk:** Low (internal pilot succeeded)

---

### Category C: User Experience

**Project 8: Real-Time Translation (Global Expansion)**
- Auto-translate product listings for international users
- **Impact:** +$200M GMV from non-English speaking countries
- **Effort:** 7 engineer-quarters
- **Cost:** $250K (translation API costs)
- **Timeline:** 5 months
- **Risk:** Medium (quality concerns)

**Project 9: AR Try-On (Augmented Reality)**
- Let users virtually try on clothes/furniture
- **Impact:** +10% conversion in fashion (+$300M GMV)
- **Effort:** 15 engineer-quarters (very complex)
- **Cost:** $500K (AR technology)
- **Timeline:** 12 months
- **Risk:** High (cutting-edge tech, unproven)

**Project 10: Review Quality Scoring**
- ML to detect fake/low-quality reviews
- **Impact:** Improve trust, reduce return rate by 2% ($50M savings)
- **Effort:** 4 engineer-quarters
- **Cost:** $30K
- **Timeline:** 3 months
- **Risk:** Low

---

### Category D: Platform / Infrastructure

**Project 11: Real-Time Feature Store**
- Build real-time feature platform for all ML models
- **Impact:** Enables future projects, improves model latency
- **Effort:** 12 engineer-quarters
- **Cost:** $400K (infrastructure)
- **Timeline:** 8 months (no direct business impact initially)
- **Risk:** Low (proven technology)

**Project 12: ML Model Monitoring Dashboard**
- Centralized monitoring for model drift/performance
- **Impact:** Prevent outages, faster debugging
- **Effort:** 3 engineer-quarters
- **Cost:** $50K
- **Timeline:** 2 months
- **Risk:** Low

---

## Prioritization Framework

### Step 1: Calculate Business Impact Score

**Formula:**
```
Business Impact = (Revenue Impact + Cost Savings) / Company Revenue
```

**For each project:**

| Project | Revenue Impact | Cost Savings | Total Annual Impact | Impact Score |
|---------|----------------|--------------|---------------------|--------------|
| 1. Visual Search | $150M | - | $150M | 0.75% |
| 2. Personalized Home | $600M | - | $600M | 3.00% |
| 3. Dynamic Pricing | $400M | - | $400M | 2.00% |
| 4. Voice Search | $100M | - | $100M | 0.50% |
| 5. Chatbot | - | $10M | $10M | 0.05% |
| 6. Fraud V2 | - | $15M | $15M | 0.08% |
| 7. Forecasting | - | $5M | $5M | 0.03% |
| 8. Translation | $200M | - | $200M | 1.00% |
| 9. AR Try-On | $300M | - | $300M | 1.50% |
| 10. Review Quality | - | $50M | $50M | 0.25% |
| 11. Feature Store | - | - | Enabler | N/A |
| 12. Monitoring | - | - | Risk reduction | N/A |

---

### Step 2: Calculate Effort Score

**Effort = (Engineer-Quarters × $50K) + External Costs**

| Project | Engineer-Quarters | Labor Cost | External Cost | Total Effort |
|---------|-------------------|------------|---------------|--------------|
| 1. Visual Search | 12 | $600K | $400K | $1,000K |
| 2. Personalized Home | 8 | $400K | $200K | $600K |
| 3. Dynamic Pricing | 6 | $300K | $100K | $400K |
| 4. Voice Search | 10 | $500K | $300K | $800K |
| 5. Chatbot | 10 | $500K | $150K | $650K |
| 6. Fraud V2 | 4 | $200K | $50K | $250K |
| 7. Forecasting | 5 | $250K | $75K | $325K |
| 8. Translation | 7 | $350K | $250K | $600K |
| 9. AR Try-On | 15 | $750K | $500K | $1,250K |
| 10. Review Quality | 4 | $200K | $30K | $230K |
| 11. Feature Store | 12 | $600K | $400K | $1,000K |
| 12. Monitoring | 3 | $150K | $50K | $200K |

---

### Step 3: Calculate ROI

**1-Year ROI = (Annual Impact / Total Effort) × 100**

| Project | Annual Impact | Total Effort | 1-Year ROI |
|---------|---------------|--------------|------------|
| 1. Visual Search | $150M | $1,000K | 15,000% |
| 2. Personalized Home | $600M | $600K | **100,000%** ⭐ |
| 3. Dynamic Pricing | $400M | $400K | **100,000%** ⭐ |
| 4. Voice Search | $100M | $800K | 12,500% |
| 5. Chatbot | $10M | $650K | 1,538% |
| 6. Fraud V2 | $15M | $250K | **6,000%** ⭐ |
| 7. Forecasting | $5M | $325K | 1,538% |
| 8. Translation | $200M | $600K | 33,333% |
| 9. AR Try-On | $300M | $1,250K | 24,000% |
| 10. Review Quality | $50M | $230K | **21,739%** ⭐ |
| 11. Feature Store | Enabler | $1,000K | N/A |
| 12. Monitoring | Risk | $200K | N/A |

---

### Step 4: Apply Weighted Scoring Model

**Criteria and Weights:**
- **Business Impact (40%):** Revenue + cost savings
- **Feasibility (25%):** Technical risk, timeline
- **Strategic Fit (20%):** Aligns with company goals
- **Resource Efficiency (15%):** ROI, team capacity

**Scoring Each Project (1-10 scale):**

| Project | Impact (40%) | Feasibility (25%) | Strategy (20%) | Efficiency (15%) | **Weighted Score** |
|---------|-------------|------------------|---------------|-----------------|-------------------|
| 1. Visual Search | 6 | 5 | 7 | 7 | **6.15** |
| 2. Personalized Home | 10 | 9 | 10 | 10 | **9.65** ⭐ |
| 3. Dynamic Pricing | 8 | 9 | 9 | 10 | **8.90** ⭐ |
| 4. Voice Search | 4 | 4 | 5 | 6 | **4.50** |
| 5. Chatbot | 5 | 6 | 8 | 3 | **5.45** |
| 6. Fraud V2 | 6 | 10 | 9 | 10 | **8.50** ⭐ |
| 7. Forecasting | 4 | 9 | 7 | 5 | **6.10** |
| 8. Translation | 7 | 7 | 8 | 8 | **7.40** |
| 9. AR Try-On | 8 | 3 | 6 | 5 | **5.95** |
| 10. Review Quality | 5 | 10 | 7 | 10 | **7.65** ⭐ |
| 11. Feature Store | Enabler | 8 | 9 | N/A | **7.00** |
| 12. Monitoring | Risk | 10 | 9 | 9 | **8.00** |

---

## Recommendation & Rationale

### Selected Projects (Top 4)

#### **✅ Project 2: Personalized Home Page** (Score: 9.65)
**Why:**
- **Highest ROI:** 100,000% first-year return
- **Huge impact:** +$600M GMV (3% conversion lift)
- **Proven technology:** Competitors already doing this successfully
- **Relatively quick:** 5 months to launch
- **Low risk:** A/B testing mitigates risk

**Resource Allocation:** 8 engineer-quarters, $200K

---

#### **✅ Project 3: Dynamic Pricing** (Score: 8.90)
**Why:**
- **Equal ROI to #2:** 100,000% return
- **Empowers sellers:** Helps small sellers compete
- **Quick win:** 4 months to MVP
- **Low risk:** Simple regression models are sufficient
- **Strategic:** Differentiates our platform

**Resource Allocation:** 6 engineer-quarters, $100K

---

#### **✅ Project 6: Fraud Detection V2** (Score: 8.50)
**Why:**
- **Excellent ROI:** 6,000% (saves $15M annually)
- **Builds on existing system:** Upgrade, not greenfield
- **Fast:** 3 months (quick win for the quarter)
- **Low risk:** Proven deep learning techniques
- **Critical:** Fraud is existential risk

**Resource Allocation:** 4 engineer-quarters, $50K

---

#### **✅ Project 10: Review Quality Scoring** (Score: 7.65)
**Why:**
- **Great ROI:** 21,739% ($50M impact)
- **Trust and safety:** Core to marketplace integrity
- **Fast:** 3 months
- **Low risk:** NLP is mature
- **Reduces returns:** Improves customer satisfaction

**Resource Allocation:** 4 engineer-quarters, $30K

---

### Total Resource Usage

**Engineering Capacity:**
- Project 2: 8 engineer-quarters
- Project 3: 6 engineer-quarters
- Project 6: 4 engineer-quarters
- Project 10: 4 engineer-quarters
- **Total: 22 engineer-quarters** (out of 40 available)

**Budget:**
- Project 2: $200K
- Project 3: $100K
- Project 6: $50K
- Project 10: $30K
- **Total: $380K** (out of $2M available)

**Remaining Capacity:**
- 18 engineer-quarters
- $1.62M budget

**Use remaining capacity for:**
- Ongoing maintenance (20% of team = 10 engineer-quarters)
- Emergency bug fixes / on-call
- Small experiments (2-week sprints)
- Start planning Q1 projects

---

## Why NOT the Others?

### Project 1: Visual Search (Score: 6.15)
**❌ Deferred to Q1**
- **Reason:** High effort (12 EQ), uncertain product-market fit
- **Decision:** Run user research first (Q4), build in Q1 if validated
- **Impact:** Waiting costs us ~$37M in Q4, but reduces risk of $1M wasted investment

### Project 4: Voice Search (Score: 4.50)
**❌ Not doing**
- **Reason:** Low impact ($100M), high effort (10 EQ), high risk
- **ROI too low:** Many e-commerce companies tried and failed
- **Alternative:** Monitor industry trends, revisit in 1-2 years

### Project 5: Chatbot (Score: 5.45)
**❌ Deferred to Q1**
- **Reason:** Good cost savings, but lower priority than revenue
- **Decision:** Deprioritize since company goal is GMV growth (revenue > cost cuts this quarter)
- **Note:** Reconsider in Q1 when focus shifts to margins

### Project 7: Warehouse Forecasting (Score: 6.10)
**❌ Deferred**
- **Reason:** Lower ROI than selected projects
- **Alternative:** Existing system is "good enough" for now

### Project 8: Translation (Score: 7.40)
**❌ Close call, deferred to Q1**
- **Reason:** Good score (7.40), but requires product/legal prep
- **Decision:** Spending Q4 on market research, legal review, then build in Q1
- **Strategic:** International expansion is 2025 priority

### Project 9: AR Try-On (Score: 5.95)
**❌ Not doing (yet)**
- **Reason:** Very high risk, long timeline (12 months), cutting-edge tech
- **Decision:** Too speculative for this quarter
- **Alternative:** Pilot with 1 engineer in background, re-evaluate in 6 months

### Project 11: Feature Store (Score: 7.00)
**❌ Deferred (but important)**
- **Reason:** Pure infrastructure, no immediate business impact
- **Decision:** We can deliver selected projects without it
- **Commitment:** Prioritize in Q1 2025 to enable future velocity

### Project 12: Monitoring Dashboard (Score: 8.00)
**❌ Partial implementation**
- **Reason:** Low effort (3 EQ), high strategic value
- **Decision:** Use remaining capacity (2 engineers) to build MVP in background
- **Justification:** Risk mitigation for new models we're launching

---

## Implementation Plan

### Month 1 (October)

**All Teams:**
- Kickoff meetings, finalize requirements
- Data pipeline setup
- Initial exploratory analysis

**Personalized Home Page:**
- User segmentation analysis
- Recommendation algorithm design
- A/B test framework setup

**Dynamic Pricing:**
- Analyze historical pricing data
- Build baseline models
- Seller survey (what features do they want?)

**Fraud V2:**
- Feature engineering (new behavioral signals)
- Model architecture selection
- Training pipeline setup

**Review Quality:**
- Label 10K reviews (human annotators)
- NLP model selection
- Baseline rule-based system

---

### Month 2 (November)

**Personalized Home Page:**
- Model training (collaborative filtering + content-based)
- Offline evaluation
- UI/UX design for personalized feed

**Dynamic Pricing:**
- Price elasticity modeling
- Competitive pricing analysis
- Seller dashboard mockups

**Fraud V2:**
- Model training (deep learning on transaction graphs)
- Offline evaluation vs. V1
- Integration planning

**Review Quality:**
- Model training (BERT-based classifier)
- Threshold tuning (precision vs. recall tradeoff)
- Define action: flag for manual review vs. auto-hide

---

### Month 3 (December)

**Personalized Home Page:**
- Online A/B test (1% traffic)
- Monitor metrics: CTR, conversion, engagement
- Iterate on ranking algorithm

**Dynamic Pricing:**
- Beta test with 100 top sellers
- Collect feedback
- Refine recommendations

**Fraud V2:**
- Shadow mode (run in parallel with V1, don't block transactions)
- Compare performance
- Oncall training for ops team

**Review Quality:**
- Production deployment (start with low-confidence reviews)
- Measure false positive rate
- Seller communication plan

---

### Month 4 (January - Early Q1)

**Personalized Home Page:**
- Ramp to 100% traffic (if A/B test succeeds)
- Launch blog post / press release
- Continuous monitoring

**Dynamic Pricing:**
- Full launch to all sellers (with opt-in option)
- Educational content for sellers
- Measure adoption rate

**Fraud V2:**
- Fully replace V1
- Celebrate $15M annual savings 🎉

**Review Quality:**
- Expand to medium-confidence reviews
- Public transparency report

---

## Success Metrics

### Project 2: Personalized Home Page

**Primary Metrics:**
- **Conversion rate:** +3% (baseline: 2.5% → target: 2.575%)
- **GMV per visitor:** +$6 (from $100 → $106)
- **Revenue impact:** $600M annually

**Secondary Metrics:**
- Click-through rate on recommendations: >15%
- Engagement time: +2 minutes per session
- Return visit rate: +5%

**Guardrail Metrics:**
- Page load time: No degradation (remain <2 sec)
- Diversity of recommendations: No filter bubbles
- Seller distribution: No bias toward large sellers

**A/B Test Plan:**
- **Control:** Static homepage (current)
- **Treatment:** Personalized feed
- **Traffic split:** 50/50
- **Duration:** 2 weeks
- **Sample size:** 10M users (sufficient for 95% confidence)
- **Decision criteria:** Conversion +2% AND no drop in guardrails → full launch

---

### Project 3: Dynamic Pricing

**Primary Metrics:**
- **GMV lift:** +$400M annually (+2%)
- **Seller adoption:** 30% of active sellers using it within 3 months
- **Pricing accuracy:** Recommended price within 10% of optimal

**Secondary Metrics:**
- Conversion rate for sellers using tool: +10%
- Average price adjustment: ~5% (slight increase)
- Seller satisfaction (NPS): +15 points

**Guardrail Metrics:**
- No extreme price changes (>50% in 24 hours)
- No price wars (detect collusion patterns)
- Maintain buyer trust (no price gouging)

**Evaluation:**
- Cohort analysis: Sellers using tool vs. not using
- Time-series comparison: Before/after adoption

---

### Project 6: Fraud Detection V2

**Primary Metrics:**
- **Fraud detection rate:** 80% → 95%
- **False positive rate:** <5% (currently 8%)
- **Annual savings:** $15M (catching $15M more fraud)

**Secondary Metrics:**
- Detection latency: <100ms per transaction
- Model accuracy: 98% (on labeled test set)
- Coverage: 100% of transactions scored

**Guardrail Metrics:**
- No increase in false positives (blocks legitimate users)
- Explainability: Can justify decision to customer support
- Compliance: Meets regulatory requirements

**Monitoring:**
- Daily fraud rate dashboard
- Weekly model performance review
- Monthly cost savings report

---

### Project 10: Review Quality Scoring

**Primary Metrics:**
- **Fake review detection:** Flag 80% of suspicious reviews
- **Return rate reduction:** 2% (from 8% → 7.84%)
- **Annual savings:** $50M (fewer returns/refunds)

**Secondary Metrics:**
- Precision: >90% (flagged reviews are actually problematic)
- Recall: >80% (catch most fake reviews)
- Seller appeals: <1% of flagged reviews

**Guardrail Metrics:**
- No bias against certain product categories
- No bias against non-native English speakers
- Maintain review volume (don't over-filter)

**A/B Test:**
- **Control:** No filtering
- **Treatment:** Hide low-quality reviews
- **Metric:** Return rate, conversion rate, trust score

---

## Interview Approach

### How to Structure Your Answer (45 min interview)

**1. Clarify the Problem (5 min)**

Ask:
- "What are the company's top goals this quarter?"
- "Do we have constraints on budget or team size?"
- "Are there any strategic priorities (e.g., international expansion)?"
- "What's the risk tolerance?"

**2. Present Framework (5 min)**

"I'll use a weighted scoring model with four criteria:
- **Business impact** (40%): Revenue + cost savings
- **Feasibility** (25%): Technical risk, timeline
- **Strategic fit** (20%): Alignment with goals
- **Resource efficiency** (15%): ROI, team capacity

I'll score each project 1-10, then multiply by weights."

**3. High-Level Analysis (10 min)**

Quickly categorize:
- "Quick wins" (high impact, low effort)
- "Strategic bets" (high impact, high effort)
- "Efficiency plays" (cost savings)
- "Long-term" (infrastructure)

Shortlist top 6-7 based on first-pass intuition.

**4. Detailed Scoring (15 min)**

For shortlisted projects:
- Calculate ROI
- Assess risk
- Check resource fit
- Apply weighted scores

Show your math (interviewers love this!).

**5. Final Recommendation (5 min)**

Present top 4 with rationale:
- Clear ranking
- Resource allocation
- Why NOT the others
- Success metrics

**6. Address Tradeoffs (5 min)**

"I'm deferring Translation and Feature Store, which are strategic long-term, to prioritize immediate GMV impact. This means we might have some tech debt, but we can address it in Q1."

---

### Common Follow-Up Questions

**Q: "What if the CEO insists on AR Try-On because he saw it at a conference?"**

A: "I'd acknowledge the vision, but explain the risk:
- 15 engineer-quarters = 38% of our capacity
- 12 months with no guarantee of success
- Opportunity cost: We'd have to cut 2-3 proven projects

I'd propose:
- Allocate 2 engineers for a 6-month proof-of-concept
- Measure user engagement in pilot
- If successful, scale in 2025 with dedicated budget

This lets us test the idea without betting the farm."

---

**Q: "Halfway through the quarter, Personalized Home Page is delayed. What do you do?"**

A: "First, diagnose why:
- Technical blocker? Allocate more engineers.
- Data quality issue? Pause and fix data pipeline.
- Model performance? Simplify model or lower bar.

Then, manage expectations:
- Update stakeholders immediately (transparency)
- Revise timeline: 5 months → 7 months
- Decide: Can we still launch in Q4, or move to Q1?

If moving to Q1, backfill with a deferred project (e.g., Chatbot) to show progress."

---

**Q: "How do you handle if Fraud V2 underperforms in production?"**

A: "Risk mitigation plan:
- **Shadow mode first:** Run V2 alongside V1, compare results
- **Gradual rollout:** Start with 10% of traffic
- **Rollback plan:** Can revert to V1 in 5 minutes
- **Post-mortem:** If it fails, analyze why (data drift? Implementation bug?)

I'd also set a decision point:
- If V2 doesn't outperform V1 after 4 weeks, pause and investigate
- Don't force a launch just because we built it"

---

### Key Takeaways

1. **Use a structured framework** (don't just go with gut feel)
2. **Show your math** (ROI calculations impress interviewers)
3. **Balance short-term wins and long-term strategy**
4. **Explain tradeoffs explicitly** (what you're NOT doing and why)
5. **Think about risk and contingency plans**
6. **Stakeholder management matters** (how to say "no" to CEO)

---

**Next:** [AI Ethics & Governance Case Study](./ai_ethics_governance.md)
