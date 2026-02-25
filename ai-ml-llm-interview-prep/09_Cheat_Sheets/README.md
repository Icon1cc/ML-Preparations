# 09 Cheat Sheets

## Overview

**Quick-reference guides** for rapid interview preparation and on-the-job decision-making.

These cheat sheets distill complex topics into actionable decision frameworks. Perfect for:
- **Last-minute interview prep** (24 hours before interview)
- **Quick refreshers** before technical discussions
- **On-the-job reference** when making architecture decisions

---

## 📚 Contents

1. [**ML Algorithm Selection Guide**](./ml_algorithm_selection_guide.md) ⭐⭐⭐
   - Decision tree for choosing algorithms
   - Comparison matrices
   - Interview answer frameworks

2. [**Transformer Cheat Sheet**](./transformer_cheat_sheet.md) ⭐⭐⭐
   - Architecture overview
   - Attention mechanism quick reference
   - BERT vs GPT comparison

3. [**LLM Fine-Tuning Decision Tree**](./llm_finetuning_decision_tree.md) ⭐⭐⭐
   - When to fine-tune vs RAG vs prompting
   - LoRA vs QLoRA vs full fine-tuning
   - Cost-benefit analysis

4. [**RAG Design Checklist**](./rag_design_checklist.md) ⭐⭐⭐
   - Chunking strategies
   - Embedding choices
   - Retrieval optimization
   - Production considerations

5. [**MLOps Checklist**](./mlops_checklist.md) ⭐⭐
   - Production ML pipeline checklist
   - Monitoring essentials
   - Deployment patterns

6. [**Common Interview Traps**](./common_interview_traps.md) ⭐⭐⭐
   - Frequent mistakes candidates make
   - How to avoid them
   - Better ways to answer

7. [**Evaluation Metrics Quick Reference**](./evaluation_metrics_quick_reference.md) ⭐⭐
   - When to use which metric
   - One-page summary

8. [**System Design Patterns for AI**](./system_design_patterns.md) ⭐⭐
   - Common architecture patterns
   - When to use each
   - Tradeoffs

---

## 🎯 Usage Guide

### For Last-Minute Interview Prep (4-6 hours before)

**Read in this order:**

1. **Hour 1:** [ML Algorithm Selection Guide](./ml_algorithm_selection_guide.md)
   - Decision frameworks
   - How to answer "which algorithm?" questions

2. **Hour 2:** [Common Interview Traps](./common_interview_traps.md)
   - Avoid common mistakes
   - Learn better answer patterns

3. **Hour 3:** [Transformer Cheat Sheet](./transformer_cheat_sheet.md) + [LLM Fine-Tuning Decision Tree](./llm_finetuning_decision_tree.md)
   - LLM fundamentals
   - When to fine-tune vs use RAG

4. **Hour 4:** [RAG Design Checklist](./rag_design_checklist.md)
   - RAG architecture patterns
   - Common pitfalls

5. **Hours 5-6:** Review case studies with these cheat sheets as reference

### For On-the-Job Use

Keep these open when:
- Designing new ML systems
- Making architecture decisions
- Troubleshooting production issues
- Writing design documents

### For Deep Learning

Start with the full sections (00-08), then use cheat sheets for quick review and reinforcement.

---

## 🚀 Quick Wins

### Top 3 Must-Read (30 minutes total)

If you only have 30 minutes:

1. **[Common Interview Traps](./common_interview_traps.md)** - 10 min
2. **[ML Algorithm Selection Guide](./ml_algorithm_selection_guide.md)** - 15 min
3. **[RAG Design Checklist](./rag_design_checklist.md)** - 5 min (skim)

These three cover 80% of common interview stumbling blocks.

---

## 📋 Checklist Format

Each cheat sheet follows a consistent format:

✅ **Decision Framework** - How to make choices
✅ **Comparison Tables** - Visual summaries
✅ **Common Pitfalls** - What to avoid
✅ **Interview Tips** - How to communicate decisions
✅ **Code Snippets** - Quick implementations
✅ **Key Takeaways** - Remember these

---

## 💡 Pro Tips

### How to Use Cheat Sheets in Interviews

**Don't:**
- ❌ Memorize word-for-word
- ❌ Recite facts without context
- ❌ Ignore the specific problem

**Do:**
- ✅ Use as mental frameworks
- ✅ Adapt to the problem at hand
- ✅ Show your thought process
- ✅ Discuss tradeoffs

**Example:**

**Bad Answer:**
"I'll use XGBoost because it's the best algorithm."

**Good Answer (Using Cheat Sheet Framework):**
"Let me think through the algorithm choice:
- Data type: Tabular → Consider tree-based models
- Dataset size: 100K rows → Sufficient for XGBoost
- Interpretability: Not critical → Can use black-box
- Performance priority: High → XGBoost typically best for tabular

I'd start with XGBoost. However, if inference latency becomes an issue, I'd reconsider using a simpler model like Logistic Regression. Let me validate this with cross-validation and compare against a Random Forest baseline."

---

## 🔗 Connections to Main Sections

These cheat sheets **distill** content from:

- [00 Foundations](../00_Foundations/) → Metrics quick reference
- [01 Machine Learning](../01_Machine_Learning/) → Algorithm selection
- [03 Modern NLP and Transformers](../03_Modern_NLP_and_Transformers/) → Transformer cheat sheet
- [04 Large Language Models](../04_Large_Language_Models/) → LLM fine-tuning decisions
- [05 RAG and Agent Systems](../05_RAG_and_Agent_Systems/) → RAG design checklist
- [06 MLOps](../06_MLOps_and_Production_AI/) → MLOps checklist
- [07 System Design](../07_System_Design_for_AI/) → System design patterns

**Workflow:**
1. Learn deeply from main sections
2. Use cheat sheets for quick review
3. Apply in case studies and interviews

---

## 📈 Study Strategy

### Week 1-3: Deep Learning
Go through main sections (00-07) thoroughly. Build understanding.

### Week 4: Practice & Review
- Work through case studies (08)
- Use cheat sheets for quick refreshers
- Practice explaining concepts using cheat sheet frameworks

### Day Before Interview
- Review all cheat sheets (2-3 hours)
- Focus on [Common Interview Traps](./common_interview_traps.md)
- Practice talking through decision frameworks out loud

### Interview Day
- Quickly skim your weakest areas (30 min in morning)
- Remember: Frameworks, not facts

---

## 🎯 Interview Success Criteria

You're ready when you can:

- [ ] Explain why you chose an algorithm (without memorizing)
- [ ] Discuss tradeoffs confidently
- [ ] Avoid common pitfalls
- [ ] Use decision frameworks naturally
- [ ] Connect technical choices to business impact

**Remember:** Interviewers want to see **how you think**, not just what you know.

---

## 📝 Contributing

Have a great cheat sheet to add?

**Good cheat sheets:**
- ✅ Actionable (help make decisions)
- ✅ Visual (tables, flowcharts)
- ✅ Practical (real-world applicable)
- ✅ Concise (1-2 pages max)

**Format:**
- Decision frameworks or comparison tables
- "When to use X vs Y"
- Common pitfalls and how to avoid
- Interview tips

---

## Key Takeaways

1. **Cheat sheets are for review**, not initial learning
2. **Understand the frameworks**, don't memorize
3. **Use in practice** - case studies and mock interviews
4. **Adapt to context** - no one-size-fits-all answers
5. **Last 24 hours** - cheat sheets are your best friend

---

**Ready?** Start with → [ML Algorithm Selection Guide](./ml_algorithm_selection_guide.md)

**Next Section:** [08 Case Studies](../08_Case_Studies/) (Apply your knowledge)
