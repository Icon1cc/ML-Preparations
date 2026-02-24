# Data and Model Versioning

> Why versioning ML data is fundamentally harder than versioning code, DVC, Delta Lake, Iceberg, model versioning patterns, and rollback strategies.

---

## Why Data Versioning Is Hard

In software engineering, version control is solved. Git handles code versioning with elegant efficiency. For ML data, every property that makes Git work well for code works against it for data:

| Property | Code (Git works well) | ML Data (Git fails) |
|----------|----------------------|---------------------|
| Size | Kilobytes to megabytes | Gigabytes to terabytes |
| Mutability | Rarely mutated in place | Often updated incrementally |
| Structure | Text, line-based diff | Binary, columnar, nested |
| Semantics | Diff is meaningful | Row-level diff is noise |
| Frequency | Committed when changed | Changes continuously in streaming |
| Network | Reasonable to clone | Impractical to clone 10TB |

The result: teams that try to version data with Git either store nothing (losing reproducibility) or break Git with massive files.

### The Stakes

Without data versioning, you cannot answer:
- "Which version of the training data produced the model currently in production?"
- "We discovered a data quality bug last month — which models were affected?"
- "The model quality dropped — did the data change or the code change?"
- "Reproduce the exact model from 3 months ago for an audit."

---

## DVC — Data Version Control

DVC (dvc.org) is the most widely used solution for ML data versioning. Its design philosophy: use Git for metadata, use remote storage for data.

### Architecture

```
Git Repository                    Remote Storage (S3/GCS/Azure)
┌───────────────────────┐        ┌────────────────────────────┐
│  .dvc/cache/           │        │  /ml-data-bucket/          │
│  data/                │◄──────►│    files/md5/              │
│    training.csv.dvc   │  DVC   │      ab/cdef123...parquet  │
│    features.pkl.dvc   │  push/ │      12/3456abc...pkl      │
│  dvc.yaml             │  pull  │      ...                   │
│  dvc.lock             │        └────────────────────────────┘
│  .gitignore           │
└───────────────────────┘
Git tracks: .dvc files, dvc.yaml, dvc.lock
DVC tracks: actual large files
```

### DVC Deep Dive

```bash
# === SETUP ===
pip install dvc dvc-s3

git init my-ml-project && cd my-ml-project
dvc init
git add .dvc .dvcignore && git commit -m "Initialize DVC"

# Configure S3 remote
dvc remote add -d production s3://company-ml-data/dvc
dvc remote modify production region us-east-1
dvc remote modify production sse aws:kms  # Encryption
git add .dvc/config && git commit -m "Configure DVC remote"

# === TRACKING DATA ===
# Add a dataset
dvc add data/raw/deliveries_2024.parquet
# Creates: data/raw/deliveries_2024.parquet.dvc (tracked by git)
#          data/raw/.gitignore (contains deliveries_2024.parquet)
#          Moves file to .dvc/cache with content-addressed name

git add data/raw/deliveries_2024.parquet.dvc data/raw/.gitignore
git commit -m "feat: add delivery dataset 2024, 45M rows"
dvc push  # Upload to S3

# === VERSIONING A NEW DATA VERSION ===
# Replace with updated data
cp data/raw/deliveries_2024_updated.parquet data/raw/deliveries_2024.parquet
dvc add data/raw/deliveries_2024.parquet  # Updates .dvc file with new hash
git add data/raw/deliveries_2024.parquet.dvc
git commit -m "feat: update delivery dataset, add October 2024 data"
dvc push

# === SWITCHING BETWEEN VERSIONS ===
# Go back to previous data version
git checkout HEAD~1 -- data/raw/deliveries_2024.parquet.dvc
dvc checkout data/raw/deliveries_2024.parquet  # Fetches from cache or remote
# Now the file is restored to the previous version

# === COMPARING VERSIONS ===
git log --oneline data/raw/deliveries_2024.parquet.dvc
# a3f8b2c feat: update delivery dataset, add October 2024 data
# b1c2d3e feat: add delivery dataset 2024, 45M rows

# Show what changed in the data pointer
git diff HEAD~1 HEAD data/raw/deliveries_2024.parquet.dvc
```

### DVC Metrics and Parameters

```bash
# params.yaml
params:
  model:
    learning_rate: 0.05
    n_estimators: 500
    max_depth: 8
  data:
    training_window_days: 90
    label_threshold_hours: 1.0

# Track metrics output
dvc metrics show
# Path                    train_loss    val_auc    test_auc
# reports/metrics.json    0.2341        0.891      0.887

# Compare across Git commits
dvc metrics diff HEAD~2
# Path                    Metric      Old       New       Change
# reports/metrics.json    val_auc     0.863     0.891     +0.028
# reports/metrics.json    test_auc    0.859     0.887     +0.028

# Compare parameters between branches
dvc params diff main feature/weather-features
# Path          Param                     Old       New
# params.yaml   model.learning_rate       0.1       0.05
# params.yaml   data.training_window      60        90
```

---

## Delta Lake — Table-Level Versioning at Scale

DVC works well for file-level versioning. For large, frequently-updated tabular datasets, Delta Lake provides table-level ACID transactions with time travel.

### Delta Lake Architecture

```
Delta Table: s3://data-lake/deliveries/

├── _delta_log/                          # Transaction log
│   ├── 00000000000000000000.json        # Initial table creation
│   ├── 00000000000000000001.json        # First append
│   ├── 00000000000000000002.json        # Schema evolution
│   ├── 00000000000000000003.json        # Update (MERGE INTO)
│   └── 00000000000000000010.checkpoint.parquet  # Checkpoint every N commits
│
├── part-00000-abc123.snappy.parquet     # Data files
├── part-00001-def456.snappy.parquet
└── ...

The _delta_log records every operation: what files were added, removed, modified.
Time travel works by replaying the log to any point in history.
```

### Delta Lake Operations

```python
from delta import DeltaTable
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

DELTA_PATH = "s3://company-data-lake/deliveries"

# === WRITING (creates versioned table) ===
df = spark.read.parquet("s3://raw-data/deliveries_2024.parquet")
df.write.format("delta").mode("overwrite").save(DELTA_PATH)
# This is version 0

# Append new data (creates version 1)
new_data = spark.read.parquet("s3://raw-data/deliveries_oct_2024.parquet")
new_data.write.format("delta").mode("append").save(DELTA_PATH)

# === TIME TRAVEL ===
# Read table as of a specific version
df_v0 = spark.read.format("delta").option("versionAsOf", 0).load(DELTA_PATH)

# Read table as of a specific timestamp
df_sept = spark.read.format("delta") \
    .option("timestampAsOf", "2024-09-30 23:59:59") \
    .load(DELTA_PATH)

# === HISTORY ===
dt = DeltaTable.forPath(spark, DELTA_PATH)
history = dt.history()
history.select("version", "timestamp", "operation", "operationParameters", "userMetadata").show()

# === MERGE (UPSERT) — creates atomic transaction ===
updates = spark.read.parquet("s3://raw-data/delivery_status_updates.parquet")

dt.alias("target").merge(
    updates.alias("source"),
    "target.tracking_id = source.tracking_id"
).whenMatchedUpdate(set={
    "actual_delivery_ts": "source.actual_delivery_ts",
    "final_status": "source.final_status",
    "delay_minutes": "source.delay_minutes"
}).whenNotMatchedInsert(values={
    "tracking_id": "source.tracking_id",
    "actual_delivery_ts": "source.actual_delivery_ts",
    "final_status": "source.final_status",
    "delay_minutes": "source.delay_minutes"
}).execute()

# === SCHEMA EVOLUTION ===
df_with_new_col = new_data.withColumn("carrier_zone_code", F.lit(None).cast("string"))
df_with_new_col.write.format("delta") \
    .option("mergeSchema", "true") \  # Adds new columns, doesn't break existing readers
    .mode("append") \
    .save(DELTA_PATH)

# === OPTIMIZE AND VACUUM ===
dt.optimize().executeCompaction()  # Merge small files
spark.sql(f"VACUUM delta.`{DELTA_PATH}` RETAIN 168 HOURS")  # Remove files older than 7 days
```

### Using Delta for ML Training

```python
# Training pipeline: always load data at a specific version
# This makes the training job reproducible

TRAINING_VERSION = 47  # Logged to experiment tracker

train_df = spark.read.format("delta") \
    .option("versionAsOf", TRAINING_VERSION) \
    .load(DELTA_PATH) \
    .filter(F.col("event_date").between("2024-01-01", "2024-09-30"))

# Log to MLflow
mlflow.log_param("delta_table_version", TRAINING_VERSION)
mlflow.log_param("delta_table_path", DELTA_PATH)
mlflow.log_param("delta_table_timestamp",
    dt.history().filter(f"version = {TRAINING_VERSION}").select("timestamp").first()[0])
```

---

## Apache Iceberg — The Enterprise Alternative

Apache Iceberg competes with Delta Lake for table-level versioning. Broader support across query engines (Spark, Flink, Trino, Presto, Dremio, Snowflake external tables):

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|------------|----------------|-------------|
| **ACID transactions** | Yes | Yes | Yes |
| **Time travel** | Yes | Yes | Yes |
| **Schema evolution** | Yes | Yes | Yes |
| **Partition evolution** | Limited | Yes (major advantage) | Yes |
| **Engine support** | Spark, Databricks, Delta RS | Spark, Flink, Trino, Presto, Athena | Spark, Hive, Presto |
| **Row-level deletes** | Yes | Yes | Yes (core use case) |
| **Streaming writes** | Limited | Good | Excellent (core use case) |
| **Cloud native** | AWS, GCP, Azure | AWS (Glue), GCP, Azure | AWS (native), others |
| **Best for** | Databricks shops | Multi-engine, AWS-first | Streaming / CDC |

### Iceberg Partition Evolution

Iceberg's killer feature over Delta: you can change partition layout without rewriting data:

```sql
-- Original table, partitioned by month
CREATE TABLE deliveries (
    tracking_id STRING,
    event_date DATE,
    carrier STRING,
    origin_zone STRING
)
USING iceberg
PARTITIONED BY (months(event_date));

-- Later: add a new partition dimension without rewriting
ALTER TABLE deliveries
ADD PARTITION FIELD carrier;
-- Old data: partitioned by month only
-- New data: partitioned by month AND carrier
-- Query engine handles mixed layouts transparently
```

---

## Data Lineage Tracking

Data lineage answers: where did this data come from, and what transformations were applied?

```python
# OpenLineage — open standard for lineage tracking
# Integrates with Airflow, Spark, dbt, Flink

from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

client = OpenLineageClient.from_environment()

# Emit START event
client.emit(RunEvent(
    eventType=RunState.START,
    eventTime="2024-10-15T09:00:00Z",
    run=Run(runId="abc123"),
    job=Job(namespace="logistics-ml", name="prepare-training-data"),
    inputs=[
        Dataset(
            namespace="s3://company-data-lake",
            name="raw/deliveries_2024",
            facets={"dataQuality": {"rowCount": 45_000_000, "nullRate": 0.02}}
        )
    ],
    outputs=[
        Dataset(
            namespace="s3://company-data-lake",
            name="processed/training_data_v1.4.2",
            facets={"schema": {"fields": [...]}}
        )
    ]
))
```

Modern data catalogs (Apache Atlas, DataHub, Alation, Collibra) visualize lineage:

```
raw/deliveries_2024 ──────► prepare_data.py ──────► training_data_v1.4.2
raw/weather_2024    ──┘                              └──► train_model.py ──► model_v3.1
external/carrier_sla ─────────────────────────────────┘
```

---

## Model Versioning — Semantic Versioning for Models

Unlike data, models have code-like versioning semantics. Proposed model semver:

```
v{major}.{minor}.{patch}

MAJOR: breaking change to input/output API, incompatible model behavior
       Example: adding a required feature, changing output schema
       Consumers MUST update their integration

MINOR: backward-compatible model improvement
       Example: retraining with more data, hyperparameter tuning
       Consumers can update at their convenience

PATCH: infrastructure fix, no model change
       Example: updating serving container, fixing logging
       Transparent to consumers

Examples:
  v1.0.0 → initial production model
  v1.1.0 → retrained with October data, AUC improved from 0.887 to 0.892
  v1.1.1 → fixed memory leak in serving container
  v2.0.0 → new feature API added (weather features), breaking change
```

### Model Registry Patterns

```python
# MLflow Model Registry — lifecycle management
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"
model_name = "delivery-delay-classifier"

registered = mlflow.register_model(model_uri, model_name)
version = registered.version

# Set version-level metadata
client.update_model_version(
    name=model_name,
    version=version,
    description="""
        XGBoost classifier with weather features.
        Trained on delivery data v1.4.2 (Jan-Sep 2024).
        Test AUC: 0.892, F1: 0.814.
        Latency: 12ms p99.
        Approved for staging by: Jane Smith (ML Lead)
        Jira: ML-1234
    """
)

# Set custom tags on the version
client.set_model_version_tag(model_name, version, "feature_set", "v2.3-with-weather")
client.set_model_version_tag(model_name, version, "data_version", "v1.4.2")
client.set_model_version_tag(model_name, version, "training_date", "2024-10-15")
client.set_model_version_tag(model_name, version, "git_commit", "a3f8b2c1")

# Transition stages
client.transition_model_version_stage(model_name, version, "Staging")
# ... run validation tests ...
client.transition_model_version_stage(model_name, version, "Production",
                                      archive_existing_versions=True)

# Load specific version for serving
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
# OR
model = mlflow.pyfunc.load_model(f"models:/{model_name}/3")  # Specific version number
```

---

## Artifact Management: Linking the Full Provenance Chain

A complete ML artifact chain must link:

```
Data Version → Feature Set Version → Code Version → Model Version → Serving Version
v1.4.2       →  v2.3              →  git:a3f8b2c → v1.1.0        → container:2024.10.15
```

```python
# Artifact lineage record — stored in model registry metadata or a metadata DB

artifact_lineage = {
    "model_name": "delivery-delay-classifier",
    "model_version": "v1.1.0",
    "model_registry_version": 7,

    # Code provenance
    "training_code": {
        "repo": "git@github.com:company/ml-models",
        "commit": "a3f8b2c1d4e7f9a0b1c2d3e4f5a6b7c8",
        "branch": "main",
        "tag": "v1.1.0-training"
    },

    # Data provenance
    "training_data": {
        "source": "s3://company-data-lake/deliveries",
        "delta_version": 47,
        "dvc_hash": "md5:abc123...",
        "row_count": 3_619_113,
        "date_range": {"start": "2024-01-01", "end": "2024-09-30"}
    },

    # Feature provenance
    "feature_set": {
        "version": "v2.3",
        "feature_store": "feast",
        "feature_list_hash": "sha256:def456...",
        "features": ["carrier_id", "origin_zone", "dest_zone", "weight_kg",
                     "hour_of_day", "day_of_week", "weather_severity", ...]
    },

    # Environment provenance
    "environment": {
        "docker_image": "ml-training:2024.10.15",
        "python_version": "3.10.12",
        "xgboost_version": "1.7.6",
        "instance_type": "ml.m5.4xlarge"
    },

    # Experiment provenance
    "mlflow_run_id": "f4a8b2c1d4e7f9a0",
    "wandb_run_id": "org/project/abc123"
}
```

---

## Rollback Strategies

When a model deployment goes wrong, rollback must be fast and reliable.

### Pre-requisites for Rollback

1. Previous model version is retained in the registry (never delete without retention policy)
2. Previous model artifact is still available in storage (not vacuumed)
3. Rollback procedure is documented and tested before the deployment
4. Automated rollback triggers are configured in monitoring

### Automated Rollback

```python
# Monitoring system triggers rollback when thresholds exceeded

def check_and_rollback(current_version: str, previous_version: str,
                        monitoring_client, registry_client):
    """Called every 5 minutes by monitoring system."""

    metrics = monitoring_client.get_recent_metrics(window_minutes=30)

    rollback_conditions = [
        metrics["error_rate"] > 0.05,               # 5% error rate
        metrics["p99_latency_ms"] > 200,             # Latency SLA breach
        metrics["psi_score"] > 0.30,                 # Severe input drift
        metrics["prediction_null_rate"] > 0.01,      # Model returning nulls
        metrics["business_metric_decline"] > 0.15,   # 15% business metric drop
    ]

    if any(rollback_conditions):
        triggered_by = [name for name, condition in zip(
            ["error_rate", "latency", "psi", "null_rate", "business"],
            rollback_conditions
        ) if condition]

        print(f"ROLLBACK TRIGGERED by: {triggered_by}")

        # Atomically switch traffic to previous version
        registry_client.transition_model_version_stage(
            name="delivery-delay-classifier",
            version=previous_version,
            stage="Production",
            archive_existing_versions=True
        )

        # Alert on-call team
        alert_oncall(
            severity="P1",
            message=f"Model rolled back from v{current_version} to v{previous_version}",
            triggered_by=triggered_by
        )
```

### Rollback Types

| Type | Mechanism | Speed | Risk |
|------|-----------|-------|------|
| **Model rollback** | Promote previous registry version | Minutes | Low |
| **Feature rollback** | Revert feature pipeline code | 30-60 min | Medium |
| **Data rollback** | Restore previous Delta table version | Minutes | Low |
| **Infrastructure rollback** | Kubernetes rollout undo | Seconds | Very Low |
| **Full pipeline rollback** | Restore all components to tagged state | 1-2 hours | Medium |

---

## Interview Questions

### "Why is data versioning harder than code versioning?"

**Senior answer**:
"Four dimensions make data versioning fundamentally harder. First, size: training datasets are gigabytes to terabytes, making git commits infeasible — we need content-addressed storage in S3 or GCS with metadata pointers in git, which is DVC's approach. Second, mutability: production data tables are continuously appended and sometimes updated in place, so versioning requires transactional semantics — Delta Lake and Iceberg solve this with ACID transactions and a transaction log that enables time travel. Third, semantic meaning: unlike code where a line-level diff is meaningful, a row-level diff between two versions of a 100M-row table is noise. Useful data versioning tracks metadata (schema, statistics, row counts) rather than row-level changes. Fourth, access patterns: you don't want to clone the full dataset to get a previous version — you want zero-copy time travel that reads the correct version directly from the shared lake, which is what Delta's version markers provide."

### "How would you handle a situation where you discover a data quality bug that affected 3 months of training data?"

**Senior answer**:
"First, I scope the impact: which model versions were trained on affected data? The model registry with data version metadata answers this immediately. Second, I archive (not delete) all affected model versions from production and roll back to the last version trained on clean data — using the model registry rollback procedure we tested in advance. Third, I fix the data quality bug in the source pipeline and validate the fix on a sample. Fourth, if we're on Delta Lake, I use time travel to restore the table to the last clean state (the version before the bug was introduced), apply the fix forward, and rebuild. Fifth, I retrain all affected models with the corrected data, compare metrics to the archived buggy versions (they should be different in the affected feature distributions), and promote the clean models through the standard staging process. Throughout, I maintain an incident timeline document that traces every affected artifact for audit purposes."

---

*Next: See `cicd_for_ml.md` for automated training pipelines and deployment automation.*
