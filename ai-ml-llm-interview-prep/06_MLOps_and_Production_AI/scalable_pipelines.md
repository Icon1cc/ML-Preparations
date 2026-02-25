# Scalable ML Pipelines (Airflow & Kubeflow)

A production ML system is not a Jupyter Notebook. It is a Directed Acyclic Graph (DAG) of distinct tasks that must execute reliably, concurrently, and securely. To manage this complexity, engineers use Orchestration frameworks.

---

## 1. What is an ML Pipeline?
A pipeline breaks down the ML lifecycle into modular, executable steps.
Example Steps: `Extract Data` $ightarrow$ `Validate Schema` $ightarrow$ `Pre-process` $ightarrow$ `Train Model` $ightarrow$ `Evaluate` $ightarrow$ `Deploy to Staging`.

### Why modularize?
*   **Caching:** If the `Train Model` step fails due to an Out-Of-Memory error, you don't want to wait 4 hours for the `Extract Data` step to run again. A pipeline framework caches the outputs of successful steps and resumes from the exact point of failure.
*   **Scalability:** The `Extract Data` step might require 50 CPU cores (Spark), while the `Train Model` step requires 4 A100 GPUs. Pipelines allow different steps to request completely different hardware environments dynamically.

## 2. Apache Airflow
The undisputed industry standard for general data orchestration.
*   **Concept:** Workflows are defined as DAGs written in pure Python.
*   **Components:** 
    *   *Scheduler:* Triggers workflows based on time (cron) or external events.
    *   *Workers:* Execute the actual tasks.
    *   *Operators:* Airflow has hundreds of built-in plugins (Operators) to talk to AWS, Snowflake, Kubernetes, and Databricks seamlessly.
*   **Best Practice:** Airflow is an *orchestrator*, not an execution engine. You should NOT process 10GB of Pandas data inside an Airflow PythonOperator. Airflow should act as the "Manager", sending a signal to Spark or Snowflake to do the heavy lifting, and waiting for the "Success" signal.

## 3. Kubeflow Pipelines (KFP)
Designed specifically for Kubernetes and Machine Learning workflows.
*   **Concept:** Every single step in the pipeline is compiled into a standalone **Docker Container**. 
*   **Why it's powerful:** 
    *   *Dependency Isolation:* The Data Preprocessing step can use Python 3.8 and Pandas 1.0, while the Training step uses Python 3.10 and PyTorch 2.0. Because they are isolated containers, they will never conflict.
    *   *Kubernetes Native:* It natively understands how to ask the Kubernetes cluster for specific hardware (e.g., "Schedule this container on a node with an Nvidia GPU").
*   **Vertex AI Pipelines:** Google Cloud's managed, serverless version of Kubeflow. It is the easiest way to run KFP without managing a complex Kubernetes cluster yourself.

## 4. The Modern Data Stack: dbt (Data Build Tool)
While Airflow orchestrates the whole system, `dbt` is explicitly used for the `Transform` phase inside Data Warehouses (Snowflake, BigQuery).
*   **Concept:** It allows data analysts to write simple SQL `SELECT` statements to build complex feature tables. `dbt` handles the boiler-plate of turning those `SELECT` statements into actual physical tables or views, handling dependencies, and running automated tests (e.g., "ensure the `age` column is never null").

## Interview Strategy
"For the offline training architecture, I would orchestrate the process using **Apache Airflow**. Airflow will trigger a **dbt** job in Snowflake to materialize the historical features. Once the data is ready, Airflow will launch an ephemeral GPU job on a Kubernetes cluster to train the model, log the artifacts to MLflow, and automatically tear down the expensive GPU node once training completes to save costs."