# Experiment Tracking — Reproducibility, Comparison, and Model Registry

> Why tracking is non-negotiable in production ML, what to track, the major tools, and how to design a reproducible ML experiment system.

---

## Why Experiment Tracking Is Critical

An ML team without experiment tracking is a team that cannot answer the following questions:

- "What was the model that was deployed 6 months ago — exact code version, data version, hyperparameters?"
- "We tried adding weather features last quarter — what was the result?"
- "Two data scientists both trained a fraud model this week — which one is better and why?"
- "The model deployed on January 15th has degraded — can we roll back to the January 1st version?"
- "Why did accuracy drop from 0.91 to 0.88 between experiment 47 and experiment 48?"

Without a tracking system, the answers to these questions require forensic investigation of Git history, filesystem timestamps, and Slack messages. With a tracking system, they are dashboard queries.

Experiment tracking is the foundation of **reproducibility** — the property that the same code, data, and environment always produce the same model with the same performance characteristics.

---

## What to Track

For every experiment, the tracking system must capture:

### 1. Parameters (Hyperparameters + Config)

```python
# All model hyperparameters
{
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}

# Preprocessing config
{
    "training_window_days": 90,
    "feature_set_version": "v2.3",
    "label_definition": "delayed_by_1hr_or_more",
    "train_cutoff_date": "2024-09-30",
    "test_start_date": "2024-10-14"
}

# Infrastructure config
{
    "instance_type": "ml.m5.4xlarge",
    "framework": "xgboost==1.7.6",
    "python_version": "3.10.12"
}
```

### 2. Metrics (Training and Evaluation)

```python
# During training (logged per epoch/iteration)
training_metrics = {
    "train_logloss": [per_iteration_values],
    "val_logloss": [per_iteration_values],
    "best_iteration": 423
}

# Final evaluation metrics
eval_metrics = {
    "test_auc_roc": 0.892,
    "test_f1": 0.814,
    "test_precision": 0.831,
    "test_recall": 0.797,
    "test_pr_auc": 0.856,
    "calibration_ece": 0.023,
    "inference_latency_p99_ms": 12.4
}
```

### 3. Artifacts

- Trained model file (`.pkl`, `.joblib`, `.pt`, ONNX)
- Preprocessing pipeline (fitted scalers, encoders)
- Feature importance plot
- Confusion matrix image
- ROC curve image
- SHAP summary plot
- Evaluation report (HTML or JSON)

### 4. Code Version

```python
# Git commit hash — links experiment to exact code
git_info = {
    "commit": "a3f8b2c1d4e7f9a0b1c2d3e4f5a6b7c8d9e0f1a2",
    "branch": "feature/add-weather-features",
    "dirty": False  # True if uncommitted changes exist
}
```

### 5. Data Version

```python
# DVC pointer or Delta table version
data_info = {
    "dataset_version": "v1.4.2",
    "dvc_hash": "md5:abc123...",
    "row_count": 4_523_891,
    "train_rows": 3_619_113,
    "test_rows": 904_778,
    "feature_set_hash": "sha256:def456..."
}
```

### 6. Environment

```python
env_info = {
    "docker_image": "ml-training:2024.10.15",
    "python": "3.10.12",
    "dependencies": "requirements-lock.txt",  # pinned versions
    "hardware": "NVIDIA A10G, 8 vCPUs, 32GB RAM"
}
```

---

## MLflow — The Open-Source Standard

MLflow is the most widely deployed open-source experiment tracking system. It has four components:

### MLflow Tracking

The core component. An API and UI for logging parameters, metrics, and artifacts.

```python
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

# Set tracking URI (local file, SQLite, or remote server)
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("delivery-delay-prediction-v2")

with mlflow.start_run(run_name="xgb-weather-features-lr005") as run:
    # Log all hyperparameters
    params = {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    mlflow.log_params(params)

    # Log data info as tags
    mlflow.set_tags({
        "dataset_version": "v1.4.2",
        "feature_set": "v2.3-with-weather",
        "git_commit": "a3f8b2c1",
        "team": "logistics-ml",
        "author": "jane.smith"
    })

    # Train model
    model = xgb.XGBClassifier(**params, use_label_encoder=False,
                               eval_metric='logloss', early_stopping_rounds=20)

    # Log metrics during training via callback
    evals_result = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        evals_result=evals_result,
        verbose=False
    )

    # Log training curve metrics
    for step, (train_loss, val_loss) in enumerate(
        zip(evals_result['validation_0']['logloss'],
            evals_result['validation_1']['logloss'])
    ):
        mlflow.log_metric("train_logloss", train_loss, step=step)
        mlflow.log_metric("val_logloss", val_loss, step=step)

    # Evaluate and log final metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    mlflow.log_metrics({
        "test_auc_roc": roc_auc_score(y_test, y_pred_proba),
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
    })

    # Log model with signature
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    # Log artifacts
    mlflow.log_artifact("reports/evaluation_report.html")
    mlflow.log_artifact("plots/shap_summary.png")
    mlflow.log_artifact("plots/confusion_matrix.png")

    print(f"Run ID: {run.info.run_id}")
```

### MLflow Model Registry

Promotes models through stages: None → Staging → Production → Archived.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow-server:5000")

# Register model from a run
model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="delivery-delay-classifier"
)

# Add model version details
client.update_model_version(
    name="delivery-delay-classifier",
    version=registered_model.version,
    description="XGBoost with weather features. AUC=0.892. Trained on v1.4.2 dataset."
)

# Transition to staging
client.transition_model_version_stage(
    name="delivery-delay-classifier",
    version=registered_model.version,
    stage="Staging",
    archive_existing_versions=False
)

# After validation, promote to production
client.transition_model_version_stage(
    name="delivery-delay-classifier",
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True  # Archives current production version
)

# Load production model anywhere
model = mlflow.pyfunc.load_model(
    model_uri="models:/delivery-delay-classifier/Production"
)
```

### MLflow Projects

Defines an ML project as a directory with an MLproject file — enables reproducible runs:

```yaml
# MLproject file
name: delivery-delay-prediction

conda_env: conda.yaml

entry_points:
  train:
    parameters:
      learning_rate: {type: float, default: 0.05}
      n_estimators: {type: int, default: 500}
      max_depth: {type: int, default: 8}
      data_version: {type: str, default: "v1.4.2"}
    command: "python train.py --lr {learning_rate} --n-estimators {n_estimators}
              --max-depth {max_depth} --data-version {data_version}"

  evaluate:
    parameters:
      model_uri: str
      test_data_path: str
    command: "python evaluate.py --model-uri {model_uri} --test-data {test_data_path}"
```

Run from any environment:
```bash
mlflow run . -P learning_rate=0.01 -P n_estimators=300
mlflow run git@github.com:org/project.git -P learning_rate=0.01
```

---

## Weights & Biases (W&B)

W&B is MLflow's main competitor, with richer visualization, better team collaboration features, and tighter integration with PyTorch and deep learning workflows.

```python
import wandb
import torch
import torch.nn as nn

wandb.init(
    project="logistics-demand-forecast",
    name="lstm-7day-forecast-run-42",
    config={
        "architecture": "LSTM",
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "sequence_length": 30,
        "forecast_horizon": 7,
        "dataset_version": "v3.1",
    },
    tags=["lstm", "demand-forecast", "weekly"],
    notes="Added zone-level embedding for geographic demand patterns"
)

model = LSTMForecast(
    input_size=wandb.config.hidden_size,
    num_layers=wandb.config.num_layers,
    dropout=wandb.config.dropout
)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_mae, val_mape = evaluate(model, val_loader)

    # Log metrics — W&B automatically tracks step
    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mae": val_mae,
        "val/mape": val_mape,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    # Log model checkpoint as artifact
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "model_checkpoint.pt")
        artifact = wandb.Artifact(
            name="demand-forecast-model",
            type="model",
            description=f"LSTM forecast model, epoch {epoch}, val_mae={val_mae:.4f}"
        )
        artifact.add_file("model_checkpoint.pt")
        wandb.log_artifact(artifact)

# Log final evaluation plots
wandb.log({
    "test/forecast_plot": wandb.Image("plots/forecast_comparison.png"),
    "test/error_distribution": wandb.Image("plots/error_dist.png"),
    "test/final_mae": test_mae,
    "test/final_mape": test_mape
})

wandb.finish()
```

### W&B Sweeps — Hyperparameter Optimization

W&B Sweeps provides a managed HPO service with Bayesian optimization, grid search, and random search:

```python
# sweep_config.yaml
sweep_config = {
    "name": "xgboost-delay-prediction-sweep",
    "method": "bayes",  # or "grid", "random"
    "metric": {
        "name": "val/auc_roc",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3
        },
        "max_depth": {
            "values": [4, 6, 8, 10, 12]
        },
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 100,
            "max": 1000
        },
        "subsample": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.0
        },
        "colsample_bytree": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.0
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5
    }
}

def train_sweep():
    with wandb.init() as run:
        config = wandb.config
        model = train_model(config)
        val_auc = evaluate(model, val_data)
        wandb.log({"val/auc_roc": val_auc})

sweep_id = wandb.sweep(sweep_config, project="delivery-delay-prediction")
wandb.agent(sweep_id, function=train_sweep, count=50)  # Run 50 trials
```

---

## DVC — Data Version Control

DVC extends Git to handle large files (data, models) that don't belong in Git itself.

### Core DVC Concepts

```bash
# Initialize DVC in a Git repo
git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Configure remote storage (S3, GCS, Azure, local)
dvc remote add -d s3remote s3://ml-team-data/dvc-store
dvc remote modify s3remote region us-east-1
git add .dvc/config
git commit -m "Configure S3 DVC remote"

# Track a large dataset
dvc add data/raw/delivery_records_2024.parquet

# This creates data/raw/delivery_records_2024.parquet.dvc
# The .dvc file is a small pointer file tracked in Git
git add data/raw/.gitignore data/raw/delivery_records_2024.parquet.dvc
git commit -m "Add delivery records dataset v1.0"

# Push data to remote storage
dvc push

# On another machine: pull data
git pull
dvc pull
```

The `.dvc` pointer file looks like:
```yaml
outs:
- md5: a4f8b2c1d4e7f9a0b1c2d3e4f5a6b7c8
  size: 14823847392
  path: delivery_records_2024.parquet
```

### DVC Pipelines

DVC can define reproducible ML pipelines:

```yaml
# dvc.yaml
stages:
  prepare_data:
    cmd: python src/prepare_data.py --input data/raw/delivery_records.parquet
                                    --output data/processed/training_data.parquet
                                    --config config/data_config.yaml
    deps:
      - src/prepare_data.py
      - data/raw/delivery_records.parquet
      - config/data_config.yaml
    outs:
      - data/processed/training_data.parquet

  train:
    cmd: python src/train.py --data data/processed/training_data.parquet
                             --config config/model_config.yaml
                             --output models/xgboost_model.pkl
    deps:
      - src/train.py
      - data/processed/training_data.parquet
      - config/model_config.yaml
    outs:
      - models/xgboost_model.pkl
    metrics:
      - reports/metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py --model models/xgboost_model.pkl
                                --data data/processed/test_data.parquet
                                --output reports/evaluation.json
    deps:
      - src/evaluate.py
      - models/xgboost_model.pkl
      - data/processed/test_data.parquet
    metrics:
      - reports/evaluation.json:
          cache: false
    plots:
      - reports/roc_curve.json
```

Run the pipeline:
```bash
dvc repro          # Run only changed stages
dvc dag            # Show pipeline DAG
dvc metrics show   # Compare metrics across versions
dvc params diff    # Show parameter changes between commits
```

---

## Tool Comparison: MLflow vs W&B vs Neptune

| Feature | MLflow | Weights & Biases | Neptune |
|---------|--------|------------------|---------|
| **Deployment** | Self-hosted or Databricks | Cloud SaaS or self-hosted | Cloud SaaS or self-hosted |
| **Cost** | Free (self-hosted) | Free tier, then $50+/user/month | Free tier, then usage-based |
| **Experiment UI** | Functional, basic | Rich, best-in-class | Rich, customizable |
| **Model Registry** | Yes, native | Yes (W&B Artifacts) | Yes |
| **HPO Sweeps** | Via MLflow (basic) | W&B Sweeps (Bayesian) | Integration with Optuna |
| **Data Versioning** | Via artifacts (limited) | W&B Artifacts | Neptune Files |
| **Deep Learning** | Good (PyTorch, TF integrations) | Excellent (native PyTorch) | Good |
| **Team Collaboration** | Basic (project sharing) | Excellent (reports, workspaces) | Good (team workspaces) |
| **Framework Support** | 20+ integrations | 50+ integrations | 40+ integrations |
| **Pipeline Integration** | Native (MLflow Pipelines) | Via API | Via API |
| **Governance/Audit** | Limited | Limited | Better enterprise features |
| **Best For** | On-prem, Databricks shops | Research teams, deep learning | Enterprise, large teams |

### When to Choose Each

**MLflow**: When you need self-hosted, open-source, or are already on Databricks. Good for teams that want full control and don't want vendor lock-in. The model registry integrates well with SageMaker and Vertex AI.

**W&B**: When you have a deep learning team that benefits from rich visualization, collaborative reports, and Sweeps for HPO. Best-in-class UI. Good for research-forward organizations.

**Neptune**: When you need strong governance, audit trails, and enterprise SSO. Good for regulated industries (finance, healthcare logistics).

---

## Experiment Naming Conventions

Consistent naming is underrated. A naming system prevents confusion in large teams:

### Naming Hierarchy

```
Project (experiment group)
└── Experiment (model version / hypothesis)
    └── Run (single training execution)
```

### Naming Convention

```
# Project naming
{team}-{use-case}
examples: logistics-delay-prediction, fraud-detection, demand-forecast

# Experiment naming (what hypothesis are we testing?)
{model-type}-{key-change}-{date}
examples:
  xgboost-baseline-20241015
  xgboost-weather-features-20241022
  lightgbm-weather-features-20241022
  xgboost-weather-features-zone-embeddings-20241029

# Run naming (for sweeps, include key param value)
{experiment-name}-lr{lr}-depth{depth}
examples:
  xgboost-weather-features-lr005-depth8
  xgboost-weather-features-lr001-depth6
```

### Tagging Strategy

```python
mlflow.set_tags({
    # Mandatory tags
    "team": "logistics-ml",
    "use_case": "delivery-delay-prediction",
    "hypothesis": "Adding weather features improves AUC by 2pp",
    "data_version": "v1.4.2",
    "feature_set": "v2.3-with-weather",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),

    # Optional but useful
    "author": "jane.smith@company.com",
    "jira_ticket": "ML-1234",
    "status": "experimental",  # experimental | candidate | champion
    "training_cost_usd": "12.40",
})
```

---

## Model Registry: Staging to Production

The model registry is a centralized catalog of model versions with lifecycle states.

### Promotion Criteria

```
STAGING criteria (minimum bar for real-traffic testing):
  ✓ AUC_ROC >= 0.88 (baseline: 0.85)
  ✓ F1 >= 0.80
  ✓ Calibration ECE <= 0.05
  ✓ Inference latency p99 <= 50ms
  ✓ All unit tests pass
  ✓ Integration test passes
  ✓ No data leakage detected

PRODUCTION criteria (champion model):
  ✓ All staging criteria met
  ✓ Shadow mode run on 10% of traffic for 7 days with no issues
  ✓ Business metric impact estimated and approved by product
  ✓ Canary deployment passed (5% traffic, 48 hours, no alerts)
  ✓ Rollback plan documented and tested
  ✓ Monitoring dashboard confirmed live
```

### Automated Promotion Pipeline

```python
def evaluate_model_for_promotion(model_version: str, experiment_id: str) -> bool:
    """Run automated quality gate checks. Returns True if model passes."""
    client = MlflowClient()

    # Get evaluation metrics from the run
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # Quality gate thresholds
    gates = {
        "test_auc_roc": (metrics.get("test_auc_roc", 0), 0.88, ">="),
        "test_f1": (metrics.get("test_f1", 0), 0.80, ">="),
        "calibration_ece": (metrics.get("calibration_ece", 1), 0.05, "<="),
        "inference_latency_p99_ms": (metrics.get("inference_latency_p99_ms", 999), 50, "<="),
    }

    all_passed = True
    gate_results = {}

    for gate_name, (value, threshold, operator) in gates.items():
        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold

        gate_results[gate_name] = {"value": value, "threshold": threshold, "passed": passed}

        if not passed:
            all_passed = False
            print(f"GATE FAILED: {gate_name} = {value:.4f}, required {operator} {threshold}")

    # Compare against current production model
    production_models = client.get_latest_versions(
        name="delivery-delay-classifier", stages=["Production"]
    )

    if production_models:
        prod_run = client.get_run(production_models[0].run_id)
        prod_auc = prod_run.data.metrics.get("test_auc_roc", 0)
        new_auc = metrics.get("test_auc_roc", 0)

        if new_auc < prod_auc - 0.005:  # Must not regress by more than 0.5pp
            print(f"REGRESSION DETECTED: new AUC {new_auc:.4f} < prod AUC {prod_auc:.4f}")
            all_passed = False

    return all_passed, gate_results
```

---

## Interview Questions and Senior Answers

### "How do you ensure reproducibility in ML experiments?"

**Senior answer**:
"Reproducibility in ML has four dimensions that must all be addressed. First, code reproducibility — every experiment is linked to a specific Git commit hash, which I log to the experiment tracker. Uncommitted changes are flagged or forbidden in production training runs. Second, data reproducibility — I version data using DVC or Delta Lake, and every experiment records the exact dataset version hash. This means I can reconstruct any historical training dataset. Third, environment reproducibility — training happens in Docker containers with pinned dependency versions. The Docker image tag is logged alongside the experiment. Fourth, algorithm reproducibility — I fix all random seeds (NumPy, Python, framework-specific) and log them. For distributed training, I also fix the data shard order.

In practice, perfect reproducibility across hardware and framework versions is sometimes impossible — different GPU architectures produce slightly different floating-point results, and some operations are non-deterministic even with fixed seeds. For this reason, I distinguish between bit-for-bit reproducibility (often unnecessary) and statistical reproducibility (same model quality within confidence intervals), and I document which level of reproducibility my system guarantees."

### "What should be stored in a model registry vs an experiment tracker?"

**Senior answer**:
"The experiment tracker is a log of all experiments — including failed ones, exploratory ones, and one-off tests. It's append-only and captures the full history of the team's exploration. The model registry is curated: it contains only the models that have passed a quality gate and are candidates for production. Every entry in the model registry links back to a specific experiment tracker run, so you can always trace a registered model back to its full provenance. The model registry also manages the deployment lifecycle (staging/production/archived), which the experiment tracker doesn't. I think of the experiment tracker as the science journal and the model registry as the product catalog."

---

*Next: See `data_and_model_versioning.md` for DVC deep dive and Delta Lake table versioning.*
