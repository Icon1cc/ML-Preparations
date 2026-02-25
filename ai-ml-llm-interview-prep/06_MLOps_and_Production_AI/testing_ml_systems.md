# Testing ML Systems

Testing machine learning code is notoriously difficult because ML models are non-deterministic. If a traditional software function `add(2, 2)` returns `5`, the test fails. If an ML model predicts house price `$400k` but the true price is `$410k`, is it broken, or is it just the expected margin of error?

A robust ML testing strategy requires three distinct layers of testing.

---

## 1. Code-Level Testing (Unit & Integration Tests)
This is testing the deterministic Python code that surrounds the model.

*   **Data Pipeline Tests:** Write assertions to ensure your custom `clean_data()` function correctly drops NaNs and correctly scales arrays.
*   **Shape & Dimensionality Tests:** Pass dummy tensors (`torch.randn(1, 3, 224, 224)`) through your PyTorch model to ensure the output tensor shape perfectly matches the expected classification layer. Most custom neural network bugs are dimension mismatches.
*   **The "Overfit a Single Batch" Test:** The ultimate integration test. In your CI/CD pipeline, train the model on just 5 rows of data for 100 epochs. Assert that the training loss drops to exactly `0.0`. If it doesn't, your core gradient calculation or loss function is fundamentally broken.

## 2. Data-Level Testing (Data Quality)
"Silent Data Failures" are the most common cause of production ML crashes. You must write tests for your data pipelines before training begins.

*   **Schema Validation:** Assert that the number of columns and their data types match expectations. (If the upstream database team changes the `zip_code` column from an Integer to a String, your ML pipeline should fail immediately, rather than crashing deep inside the training loop).
*   **Distribution Checks:** Assert that values fall within logical bounds. 
    *   `assert min(Age) >= 0 and max(Age) < 120`
    *   `assert percent_nulls(Delivery_Time) < 5%`
*   **Tools:** **Great Expectations** is the industry standard library for defining and running these data quality rules.

## 3. Model-Level Testing (Behavioral Testing)
Once the model is trained, you must test its behavior *before* deploying it, going beyond simple aggregate metrics like overall Accuracy.

### A. Invariance Tests (Perturbation Testing)
Testing if the model remains stable when given inputs that *should not* change the output.
*   *NLP Example:* If you change the name in the prompt from "John" to "Jamal", the sentiment prediction should remain identical. If it changes, the model has learned a racial bias.
*   *Vision Example:* If you rotate an image of a dog by 2 degrees, it should still be classified as a dog.

### B. Directional Expectation Tests
Testing if the model moves in the logical direction when a specific feature is changed.
*   *Logistics Example:* If you take a specific shipment profile and artificially increase the `Distance_in_Miles` feature by 500, the predicted `ETA` *must* increase. If the predicted ETA goes down, the model has learned an illogical, broken relationship.

### C. Slice Testing (Sub-population Analysis)
Aggregate accuracy hides localized failures. A model might have 95% accuracy overall, but completely fail on a specific minority group.
*   Test the model's accuracy on distinct slices of data: e.g., specifically testing accuracy on `Route_Type = Urban` vs `Route_Type = Rural`. The performance should remain relatively equitable across critical business segments.

## Interview Strategy
"Traditional software testing checks if code works. ML testing must verify that the code, the data, and the mathematical assumptions all work together. I would implement **Great Expectations** in the Airflow DAG to catch data anomalies before training. Post-training, I would automate **Slice Testing** to ensure the model doesn't degrade performance for specific critical logistics routes before allowing it to pass the CI/CD gate."