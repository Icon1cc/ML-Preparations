# Business Case Study: AI Ethics, Fairness & Governance

> **Domain:** Ethics / Compliance / Responsible AI
> **Problem Type:** Risk Management / Governance Framework
> **Difficulty:** ⭐⭐⭐ (Senior/Principal Level)
> **Interview Duration:** 60 minutes
> **Common in:** Senior ML roles, AI Ethics positions, Regulated industries

## Table of Contents
1. [Business Context](#business-context)
2. [Problem Statement](#problem-statement)
3. [Ethical Risk Assessment](#ethical-risk-assessment)
4. [Fairness Analysis Framework](#fairness-analysis-framework)
5. [Technical Solutions](#technical-solutions)
6. [Governance Framework](#governance-framework)
7. [Stakeholder Management](#stakeholder-management)
8. [Monitoring & Auditing](#monitoring--auditing)
9. [Crisis Response Plan](#crisis-response-plan)
10. [Interview Approach](#interview-approach)

---

## Business Context

**Company:** National mortgage lender
- Processes 500,000 loan applications annually
- $50B in loan originations
- Operates in all 50 US states
- Subject to Fair Lending Laws (ECOA, Fair Housing Act)

**Current Situation:**
- Manual underwriting: 3-5 days, high cost
- Approval rate: 65%
- Default rate: 2.5%
- Customer satisfaction: declining due to slow process

**Opportunity:**
- Implement ML-based automated loan decisioning
- Reduce approval time to <1 hour
- Maintain or improve default rate
- Scale to 1M applications/year

**The Challenge:**
Last year, a competitor faced a $10M discrimination lawsuit when investigative journalists discovered their ML model denied loans to minority applicants at 2x the rate of white applicants, even after controlling for credit scores.

**Your Role:** Head of AI Governance, tasked with ensuring the new system is fair, compliant, and defensible.

---

## Problem Statement

**Task:** Design a comprehensive AI governance framework for automated loan decisioning that:
1. Ensures fairness across protected groups
2. Maintains regulatory compliance
3. Provides explainability for decisions
4. Establishes monitoring and auditing processes
5. Prepares for worst-case scenarios

**Constraints:**
- **Regulatory:** Must comply with ECOA, Fair Housing Act, state laws
- **Business:** Cannot sacrifice too much accuracy (default rate must stay <3%)
- **Timeline:** System launches in 6 months
- **Stakeholders:** Engineering, Legal, Compliance, Business, Regulators

**Deliverable:** Present governance framework to Board of Directors

---

## Ethical Risk Assessment

### Protected Classes Under Federal Law

**ECOA (Equal Credit Opportunity Act) Protected Classes:**
- Race/Color
- Religion
- National origin
- Sex (including gender identity, sexual orientation)
- Marital status
- Age (must be 18+, but cannot discriminate against elderly)
- Receipt of public assistance

**State-Level Additions:**
- ZIP code (proxy for race in some states)
- Employment status
- Source of income

---

### Risk Identification Matrix

| Risk Category | Specific Risk | Probability | Impact | Mitigation Priority |
|--------------|--------------|-------------|--------|---------------------|
| **Bias** | Racial disparate impact | High | Critical | 1 |
| **Bias** | Gender discrimination | Medium | Critical | 1 |
| **Bias** | Age discrimination (elderly) | Medium | High | 2 |
| **Proxy discrimination** | ZIP code as race proxy | High | Critical | 1 |
| **Explainability** | Cannot justify denials | High | Critical | 1 |
| **Data quality** | Historical bias in training data | High | High | 2 |
| **Model drift** | Performance changes over time | Medium | High | 2 |
| **Security** | Model manipulation/gaming | Medium | Medium | 3 |
| **Reputational** | Media investigation | Medium | Critical | 1 |

**Top 3 Risks:**
1. Racial/ethnic disparate impact
2. Lack of explainability
3. Historical bias in training data

---

## Fairness Analysis Framework

### Step 1: Define Fairness Metric(s)

**The problem:** There are multiple definitions of fairness, and they can be mutually exclusive!

**Common Fairness Metrics:**

**1. Demographic Parity (Statistical Parity)**
```
P(Approved | Race=A) = P(Approved | Race=B)
```
- **Meaning:** Approval rates should be equal across groups
- **Pros:** Easy to understand, aligns with public perception of fairness
- **Cons:** Ignores differences in qualifications

**2. Equal Opportunity**
```
P(Approved | Qualified, Race=A) = P(Approved | Qualified, Race=B)
```
- **Meaning:** Among qualified applicants, approval rates should be equal
- **Pros:** Considers merit
- **Cons:** Requires defining "qualified"

**3. Predictive Parity**
```
P(Default | Approved, Race=A) = P(Default | Approved, Race=B)
```
- **Meaning:** Among approved applicants, default rates should be equal
- **Pros:** Business-aligned (risk is equal)
- **Cons:** Can perpetuate historical inequalities

**4. Calibration**
```
P(Default | Score=X, Race=A) = P(Default | Score=X, Race=B)
```
- **Meaning:** Risk scores mean the same thing across groups
- **Pros:** Predictive accuracy is equal
- **Cons:** May still have disparate impact

---

### Our Approach: Multi-Metric Fairness

**Primary Metric:** Equal Opportunity (business + fairness balance)
- Among creditworthy applicants (score >700), approval rates within 5% across racial groups

**Secondary Metrics:**
- Demographic parity: Approval rates within 10% (allows for qualification differences)
- Predictive parity: Default rates within 1% across groups

**Rationale:**
- Equal opportunity is legally defensible (merit-based)
- Secondary metrics provide additional safety nets
- We'll report all three to regulators

---

### Step 2: Data Audit

**Training Data Analysis:**

```python
# Historical loan data
loans = pd.read_csv('historical_loans.csv')

# Protected class distribution
print(loans.groupby('race')['approved'].mean())

# Output:
# White:     72%
# Black:     58%
# Hispanic:  61%
# Asian:     75%
```

**🚨 Red Flag:** 14% gap between White and Black approval rates

**Root Cause Analysis:**
1. **Historical bias:** Loan officers may have been biased
2. **Systemic factors:** Black applicants may have lower average incomes (legacy of discrimination)
3. **Data quality:** Missing data more common for minority applicants

**Decision:** We CANNOT train on raw historical data. It perpetuates bias.

---

### Step 3: Feature Audit

**Explicitly Prohibited Features (Direct):**
- Race, color, religion, national origin
- Sex, marital status
- Age (except to verify 18+)
- ZIP code (in isolation)

**Proxy Features (Indirect):**

| Feature | Proxy For | Correlation | Action |
|---------|----------|-------------|--------|
| **ZIP code** | Race | High (0.7+) | **Remove or use carefully** |
| **First name** | Race/ethnicity | Medium | **Remove** |
| **Employer name** | Race (if segregated industry) | Medium | **Audit** |
| **Bank account type** | Wealth (correlated with race) | Low | **Monitor** |
| **Income** | Protected class | Medium | **Keep but monitor** |

**Decision:**
- Remove: First name, standalone ZIP code
- Keep with monitoring: Income, debt-to-income ratio
- Derived feature: Instead of ZIP code, use "median home value in ZIP" (less direct proxy)

---

## Technical Solutions

### Solution 1: Fairness-Constrained Training

**Approach:** Add fairness constraints directly to the loss function

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# Base model
base_model = GradientBoostingClassifier()

# Fairness-constrained model
fairness_model = ExponentiatedGradient(
    estimator=base_model,
    constraints=EqualizedOdds(),  # Equal opportunity constraint
    sensitive_features=['race']  # Only for training, not at inference
)

# Train
fairness_model.fit(X_train, y_train, sensitive_features=race_train)

# At inference, sensitive features NOT used
predictions = fairness_model.predict(X_test)
```

**Result:**
- Approval rate gap: 14% → 5%
- Overall accuracy: 87% → 85% (small tradeoff)
- Default rate: 2.5% → 2.6% (acceptable)

---

### Solution 2: Post-Processing Threshold Adjustment

**Approach:** Use different decision thresholds for different groups to achieve parity

```python
def calibrated_thresholds(model, X, y, sensitive_feature, target_parity=0.05):
    """
    Find group-specific thresholds that achieve demographic parity
    """
    thresholds = {}

    for group in sensitive_feature.unique():
        mask = (sensitive_feature == group)
        X_group = X[mask]
        y_group = y[mask]

        # Find threshold that achieves target approval rate
        scores = model.predict_proba(X_group)[:, 1]
        threshold = find_threshold_for_target_rate(scores, y_group, target_rate)

        thresholds[group] = threshold

    return thresholds

# Apply
thresholds = calibrated_thresholds(model, X, y, race, target_parity=0.05)
```

**⚠️ Legal Risk:** Group-specific thresholds may be viewed as "quotas" (illegal under ECOA)

**Decision:** Use Solution 1 (fairness-constrained training) instead

---

### Solution 3: Explainability (SHAP for Loan Decisions)

**Requirement:** Applicant has a right to know why they were denied (ECOA Adverse Action Notice)

```python
import shap

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For a specific denial
applicant_id = 12345
explanation = shap_values[applicant_id]

# Top 3 reasons for denial
reasons = get_top_reasons(explanation, X_test[applicant_id], feature_names)
```

**Output:**
```
Your application was denied due to:
1. Debt-to-income ratio: 45% (guideline: <40%)
2. Credit score: 620 (guideline: 680+)
3. Employment history: 6 months (guideline: 12+ months)

To improve your application:
- Reduce your monthly debt payments
- Work on building your credit score
- Reapply after 6 more months of employment
```

**Compliance:** This meets ECOA Adverse Action Notice requirements

---

## Governance Framework

### Layer 1: Pre-Deployment Review

**AI Ethics Board:**
- **Composition:** Legal, Compliance, Engineering, Business, External Ethicist
- **Meeting:** Quarterly + ad-hoc for new models
- **Authority:** Veto power over launches

**Checklist for New Models:**
- [ ] Fairness audit passed (all 3 metrics within tolerance)
- [ ] Explainability system tested
- [ ] Legal review completed
- [ ] Documentation complete (model card)
- [ ] Regulator notification sent (if required)
- [ ] Red team testing passed (adversarial scenarios)

---

### Layer 2: Model Card (Documentation)

**Required for Every Model:**

```markdown
# Model Card: Loan Approval Model v2.1

## Model Details
- **Developer:** Acme Bank ML Team
- **Model date:** 2024-03-15
- **Model type:** XGBoost classifier
- **Model version:** 2.1

## Intended Use
- **Primary:** Automated loan decisioning for mortgage applications
- **Out of scope:** Commercial loans, refinancing
- **Users:** Loan officers, applicants (indirect)

## Training Data
- **Source:** 500K historical loans (2019-2023)
- **Preprocessing:** Removed ZIP code, first name
- **Protected class distribution:** [Table]

## Performance
- **Overall accuracy:** 85%
- **Default rate:** 2.6%
- **Approval rate:** 67%

## Fairness Metrics
- **Equal opportunity:** Approval rates within 5% for qualified applicants
- **Demographic parity:** Approval rates within 10%
- **Predictive parity:** Default rates within 1%

## Limitations
- **Geographic:** Trained on US data only
- **Economic:** Trained during low interest rate environment
- **Temporal:** Performance may degrade in recession

## Monitoring
- **Cadence:** Weekly fairness checks, monthly full audit
- **Alert thresholds:** Approval rate gap >7%
```

---

### Layer 3: Human-in-the-Loop

**Tiered Review Process:**

**Tier 1: Full Automation (70% of applications)**
- Clear approvals (score >0.8, no red flags)
- Clear denials (score <0.3, unambiguous)
- **Human review:** None (post-hoc auditing only)

**Tier 2: Semi-Automated (25% of applications)**
- Borderline cases (score 0.3-0.8)
- Any protected class flag
- **Human review:** Loan officer reviews recommendation + explanation

**Tier 3: Manual (5% of applications)**
- High loan amount (>$1M)
- Complex scenarios (self-employed, recent immigrant)
- **Human review:** Full underwriting

**Fallback:** If model API fails, route 100% to Tier 3 (manual)

---

## Monitoring & Auditing

### Real-Time Monitoring

**Dashboard Metrics (Updated Hourly):**

```python
# Fairness metrics
approval_rate_by_race = loans.groupby('race')['approved'].mean()
gap = approval_rate_by_race.max() - approval_rate_by_race.min()

if gap > 0.07:  # 7% threshold
    send_alert("Fairness threshold breached", severity="HIGH")

# Model performance
default_rate_week = loans[loans.approved==True]['defaulted'].mean()

if default_rate_week > 0.04:  # 4% threshold (normal is 2.5%)
    send_alert("Default rate spike", severity="MEDIUM")
```

**Alert Response:**
- **High severity:** Pause automated decisions, route to manual review
- **Medium severity:** Investigate within 24 hours
- **Low severity:** Log for weekly review

---

### Monthly Audit

**Conducted by:** Independent audit team (not ML team)

**Audit Checklist:**
1. **Fairness check:** Approval rates by protected class
2. **Outcome analysis:** Default rates by protected class
3. **Explanation quality:** Sample 100 denials, verify SHAP explanations
4. **Data drift:** Feature distributions compared to training data
5. **Adversarial testing:** Test with synthetic biased applicants

**Report to:** AI Ethics Board, Legal, Compliance

---

### Annual External Audit

**Conducted by:** Third-party auditing firm (e.g., NIST, academic researchers)

**Scope:**
- Full model audit
- Review of governance processes
- Recommendations for improvement

**Output:**
- Public-facing fairness report
- Certification (if passed)
- Shared with regulators

---

## Crisis Response Plan

### Scenario: Journalist Investigation Finds Bias

**Timeline:**

**Day 1 (Story Breaks):**
- 8:00 AM: Article published: "Bank's AI denies loans to Black applicants at 2x rate"
- 9:00 AM: Crisis team convenes (CEO, General Counsel, Head of AI, PR)
- 10:00 AM: Pull automated decisions (route all to manual)
- 11:00 AM: Notify regulators proactively
- 2:00 PM: Public statement: "We take this seriously, investigating immediately"

**Day 2-3 (Investigation):**
- Independent audit of model
- Review last 90 days of decisions
- Identify root cause

**Possible Findings:**

**Finding A: Claim is inaccurate**
- Approval rates actually equal after controlling for credit scores
- Data was misinterpreted by journalist
- **Response:** Publish detailed rebuttal with data, offer to share methodology with independent experts

**Finding B: Claim is accurate - model is biased**
- Model shows 8% approval gap
- **Root cause:** Proxy feature (employer name) inadvertently correlated with race
- **Response:**
  - Public apology
  - Immediately retrain model without biased feature
  - Offer to re-review all denied applications from last 90 days
  - $5M fund for financial literacy programs in affected communities
  - Commit to annual external audits

**Finding C: Claim is accurate - systemic issue**
- Model itself is fair, but outcomes reflect broader economic inequality
- **Response:**
  - Acknowledge the systemic issue
  - Clarify model is not the cause (but may perpetuate it)
  - Launch alternative lending program (lower rates for underserved communities)

---

## Stakeholder Management

### To Regulators (OCC, CFPB)

**Proactive Engagement:**
- **Before Launch:** Submit model documentation, request informal feedback
- **Quarterly:** Share fairness metrics
- **Annually:** Invite regulator to audit

**Message:**
"We're committed to fair lending. Here's our comprehensive framework. We welcome your feedback."

**Build Trust:**
- Transparency (share data)
- Humility (acknowledge challenges)
- Accountability (own mistakes)

---

### To Executives (CEO, Board)

**Frame as Risk Management:**
- "A discrimination lawsuit costs $10M+ in settlements plus reputational damage"
- "Our framework reduces this risk by 80%"
- "The 2% accuracy tradeoff is insurance"

**Show Business Case:**
- Still saves $15M/year vs. manual underwriting
- Increases approval volume 3x
- Protects company from existential risk

---

### To Engineering Team

**Frame as Technical Challenge:**
- "This is cutting-edge ML research (fairness-aware learning)"
- "We're building something that will be published/open-sourced"
- "This makes you a better engineer"

**Provide Tools:**
- Fairness libraries (Fairlearn, AI Fairness 360)
- Training on bias mitigation
- Dedicated time for ethics work (10% of sprint)

---

### To Loan Officers (Users)

**Address Job Security Fears:**
- "AI handles routine cases, you focus on complex scenarios"
- "Your expertise is needed for edge cases"
- "We're not eliminating jobs, we're enhancing them"

**Training:**
- How to interpret AI explanations
- When to override AI recommendations
- How to explain decisions to customers

---

## Interview Approach

### How to Structure Your Answer (60 min interview)

**1. Acknowledge Complexity (3 min)**
"This is a critical problem. Fairness in lending is not just an ethical imperative but a legal requirement. There's no perfect solution, but I'll walk through a comprehensive framework."

**2. Risk Assessment (10 min)**
- Identify top ethical risks (bias, explainability, compliance)
- Categorize by probability and impact
- Prioritize top 3

**3. Technical Solutions (15 min)**
- Fairness metrics (choose one primary, explain why)
- Bias mitigation techniques (constrained optimization, threshold tuning)
- Explainability approach (SHAP)
- Acknowledge tradeoffs (fairness vs. accuracy)

**4. Governance Framework (15 min)**
- Pre-deployment review (AI Ethics Board)
- Human-in-the-loop for edge cases
- Documentation (Model Card)
- Monitoring and auditing (real-time, monthly, annual)

**5. Stakeholder Strategy (10 min)**
- Regulators: Proactive transparency
- Executives: Risk management frame
- Engineers: Technical challenge frame
- Loan officers: Job enhancement, not replacement

**6. Crisis Plan (7 min)**
- Scenario: Bias discovered
- Response: Immediate pause, investigation, remediation
- Communication: Transparency, accountability, action

---

### Common Follow-Up Questions

**Q: "Can't we just remove race from the data and be done with it?"**

A: "Unfortunately, no. This is called 'fairness through unawareness,' and it doesn't work because:
1. **Proxy features:** Other features (ZIP code, name) can proxy for race
2. **Historical bias:** Training data itself contains biased decisions
3. **Systemic factors:** Correlated economic factors reflect historical discrimination

Paradoxically, we need to *measure* race during training and auditing to ensure fairness, even though we don't use it at inference."

---

**Q: "What if the fairness constraint makes the model too inaccurate?"**

A: "First, I'd quantify 'too inaccurate':
- If accuracy drops from 87% to 85%, that's acceptable (fairness is worth 2%)
- If it drops to 75%, we have a problem

If the tradeoff is unacceptable:
1. **Better features:** Collect more predictive, non-biased features
2. **Larger training set:** More data can improve both fairness and accuracy
3. **Different model:** Try different architectures
4. **Hybrid approach:** Human review for borderline cases

But I'd push back on the premise: studies show fairness and accuracy are *not* fundamentally in tension if the model is well-designed."

---

**Q: "How do you handle if a protected class actually has higher default rates?"**

A: "This is a critical question. Let's say data shows Group A defaults at 5%, Group B at 2%.

**We CANNOT:**
- Use race directly in the model
- Apply different thresholds based on race (that's illegal discrimination)

**We CAN:**
- Use legitimate risk factors that happen to correlate with default (e.g., credit score, debt-to-income)
- Ensure the model is calibrated (a score of 0.7 means 70% approval probability for *everyone*)

**The key distinction:**
- **Disparate treatment** (using race directly): ILLEGAL
- **Disparate impact** (outcomes differ): LEGAL *if* based on legitimate business necessity

If Group A has higher denials because of lower average credit scores, that's legal. But we must:
1. Verify credit score is a valid predictor (business necessity)
2. Ensure no less discriminatory alternative exists
3. Monitor for proxy discrimination"

---

### Red Flags to Avoid

❌ **"Fairness is simple, just treat everyone the same"**
- Shows lack of understanding of systemic bias

❌ **"We'll just remove race and be fine"**
- Ignores proxy discrimination

❌ **"Fairness and accuracy are mutually exclusive"**
- Shows defeatist mindset

❌ **"This is a technical problem only"**
- Ignores legal, ethical, social dimensions

❌ **"Regulators will never find out"**
- Massive ethical failure, career-ending

---

### Strong Answers Include

✅ **Acknowledge complexity and tradeoffs**
✅ **Multiple fairness metrics with justification**
✅ **Technical solutions (constrained optimization, SHAP)**
✅ **Comprehensive governance (pre-launch, monitoring, auditing)**
✅ **Stakeholder-specific communication strategies**
✅ **Crisis response plan**
✅ **Humility ("no perfect solution, but here's a robust framework")**

---

## Key Takeaways

1. **Fairness is multidimensional** - no single metric captures it
2. **Removing protected attributes is not enough** - proxy discrimination is real
3. **Fairness-accuracy tradeoff is small** with good engineering
4. **Governance is as important as technical solutions**
5. **Proactive transparency with regulators** reduces risk
6. **Human-in-the-loop for edge cases** is non-negotiable
7. **Crisis planning is essential** - assume you'll be investigated

---

**Next:** [Technical Case Study: ML System Design](../Technical_Case_Studies/README.md)
