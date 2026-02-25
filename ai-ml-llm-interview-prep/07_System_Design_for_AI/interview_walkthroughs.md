# ML System Design: Step-by-Step Interview Walkthroughs

This file provides "cheat sheets" for three of the most common ML system design problems. Use these to practice the 7-step framework.

---

## 1. Recommendation System (e.g., Netflix/YouTube)

**Objective:** Increase user watch time.
**Target:** $P(click) 	imes P(watch\_time > 30s)$.

1.  **Candidate Generation (Retrieval):** 
    *   From millions of videos, find the top 500 relevant ones.
    *   *Approach:* Collaborative filtering, Two-Tower Neural Network (User Embedding dot Item Embedding).
2.  **Ranking (Scoring):**
    *   Rank the 500 candidates.
    *   *Approach:* Pointwise ranking using DeepFM or Wide & Deep model (captures both memorization and generalization).
    *   *Features:* User history, video genre, thumbnail quality, current trending score.
3.  **Re-Ranking (Post-Processing):**
    *   Apply business logic.
    *   *Filters:* Remove adult content, remove already seen videos, ensure **diversity** (don't show 10 cat videos in a row).
4.  **Offline Eval:** Precision@K, NDCG.
5.  **Online Eval:** CTR, Average Watch Time per User.

---

## 2. Ad Click Prediction (CTR Prediction)

**Objective:** Maximize revenue.
**Target:** Binary Classification ($0$ or $1$ click).

1.  **Architecture:** Logistic Regression (baseline) or Factorization Machines (handles high-cardinality categorical features like `ad_id` and `user_id` very well).
2.  **The "Cold Start" Problem:** How to handle new ads/users with no history?
    *   *Solution:* Use content-based features (ad category, image text) rather than ID-based features.
3.  **Calibration:** In ads, we don't just need the class ($0$ or $1$), we need the exact probability (e.g., $0.0352$) to calculate the expected value for bidding. Use **Platt Scaling** or **Isotonic Regression** to calibrate the model's raw scores into true probabilities.
4.  **Data Leakage:** Be careful! Never use features like "number of times user clicked in the future" or data from the same session as the target. Use a sliding time window for training.

---

## 3. Fraud Detection System (e.g., Credit Card/Transaction)

**Objective:** Minimize financial loss while minimizing user friction.
**Target:** Probability of fraud.

1.  **Classification Task:** High-imbalance problem (99.9% legit, 0.1% fraud).
2.  **Features:** 
    *   *Velocity:* Number of transactions in the last hour.
    *   *Geography:* Is the purchase 500 miles from the previous one?
    *   *Network:* Is the user's IP associated with other fraudulent accounts?
3.  **Model:** XGBoost or LightGBM (good for tabular data and handles missing values well).
4.  **Handling Imbalance:** 
    *   *Metrics:* Do NOT use Accuracy. Use **Precision-Recall AUC** or **F1-Score**.
    *   *Sampling:* Use SMOTE (Synthetic Minority Over-sampling Technique) or simply adjust the class weights in the loss function.
5.  **The Pipeline:**
    *   Needs to be **Synchronous/Online**. Decisions must be made in $<100ms$ to approve/deny the transaction at the point of sale. Use a Feature Store (Redis) for millisecond lookups of user historical features.

---

## 4. Key Takeaway: The "Two-Tower" Pattern
In interviews for Search, Recommendations, or Ad-Matching, always mention the **Two-Tower Architecture**.
*   **Tower 1 (User):** Deep net that learns a 128-dim vector for the user.
*   **Tower 2 (Item):** Deep net that learns a 128-dim vector for the item.
*   **The Magic:** You can pre-calculate and store all Item Embeddings in a Vector DB. At runtime, you only compute the User Embedding and do a fast dot-product search. This is the only way to scale these systems to millions of items.