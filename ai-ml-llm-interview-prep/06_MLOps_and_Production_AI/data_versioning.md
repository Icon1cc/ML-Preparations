# Data Versioning

In machine learning, data is code. If you train a model on a dataset on Monday, and the dataset is modified on Tuesday, your Monday model is no longer reproducible. Data Versioning solves this by tracking changes to datasets over time, just like Git tracks changes to source code.

---

## 1. Why Git Fails for Data
Git is designed for small, line-by-line text files. 
*   If you commit a 10GB CSV file to Git, the repository will crash or become unusably slow.
*   Git cannot efficiently track binary files (like folders of millions of images for computer vision).

## 2. DVC (Data Version Control)
The industry standard open-source tool for data versioning. It integrates directly with Git.

### How DVC Works
1.  **Storage Decoupling:** DVC stores the actual heavy data files (CSV, images, model weights) in a remote cloud storage bucket (AWS S3, Google Cloud Storage, Azure Blob).
2.  **Pointer Files:** DVC creates a tiny text file (e.g., `dataset.csv.dvc`) in your local directory. This file contains an MD5 hash of the data and the remote storage path.
3.  **Git Integration:** You commit the tiny `.dvc` pointer file to Git, NOT the massive CSV. 
4.  **Time Travel:** When you `git checkout` an older branch from 6 months ago, the `.dvc` file changes. You then run `dvc pull`, and DVC fetches the exact 10GB dataset associated with that specific point in time from S3.

## 3. Data Lakehouse Architecture (Delta Lake / Iceberg)
For massive tabular data, modern enterprises use Lakehouse formats like Databricks Delta Lake or Apache Iceberg. These formats build versioning directly into the database layer.

### Features
*   **Time Travel:** You can write a SQL query like: 
    `SELECT * FROM logistics_data TIMESTAMP AS OF '2023-01-01'`
    The system automatically returns the exact state of the database on that day.
*   **ACID Transactions:** Allows safe, concurrent reads and writes to massive datasets without data corruption.

## 4. Why Data Versioning is Mandatory for MLOps

### A. Reproducibility
A regulator asks: "Why did your model deny this loan last year?" If you don't have the exact training data from last year, you cannot retrain or audit the model.

### B. Debugging Model Degradation
If your model's accuracy drops from 90% to 70% after retraining, how do you know if it's a code bug or a data bug?
With versioning, you can do an exact diff between Dataset v1 and Dataset v2. You might find that Dataset v2 accidentally dropped an entire column of data during a broken ETL pipeline job.

### C. Pipeline Automation
DVC isn't just for storage; it tracks dependencies. You can define a pipeline: `Raw Data -> Cleaning Script -> Clean Data -> Training Script -> Model`.
If the Raw Data changes, DVC knows the pipeline is "dirty" and automatically re-runs only the necessary scripts to produce the new model.

## Interview Strategy
"In a production system, data is a hyperparameter. I would implement **DVC** to version our image datasets, pushing the raw files to S3 and committing the DVC pointers to Git alongside the training code. This guarantees that every registered model in MLflow can be perfectly traced back to the exact byte-for-byte dataset it was trained on."