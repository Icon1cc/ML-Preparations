# Case Study: Supply Chain Anomaly Detection

## 1. Business problem
Detect operational anomalies across parcel volume, route delays, and fraud-like patterns.

## 2. Anomaly categories
- sudden depot volume spikes
- abnormal scan sequence patterns
- unusual billing or claims behavior
- equipment throughput drops

## 3. Methods
- statistical baselines (z-score, STL residual)
- Isolation Forest / One-Class SVM
- sequence autoencoders for temporal anomalies

## 4. Ensemble design
Combine detector scores with weighted voting and calibrated thresholds by region/time.

## 5. Architecture
```mermaid
flowchart LR
    A[Event streams] --> B[Feature engineering]
    B --> C[Detector ensemble]
    C --> D[Severity scoring]
    D --> E[Alert manager]
    E --> F[Ops runbooks]
```

## 6. Alert operations
- severity levels
- suppression windows
- deduplication
- incident ownership matrix

## 7. Evaluation
- precision@k for analyst queue
- time-to-detection
- false positive rate by segment

## 8. Interview questions
1. How evaluate with sparse labels?
2. How reduce alert fatigue?
3. How separate seasonal peaks from anomalies?
