# Multi-Tenant AI Systems

In enterprise B2B Software as a Service (SaaS), you don't build one model for everyone. You build a platform that serves hundreds of different clients (tenants), ensuring strict data isolation and performance guarantees for each.

---

## 1. What is Multi-Tenancy in ML?
A single instance of a software application serves multiple customers (tenants).
*   **The Challenge:** Tenant A (e.g., Walmart) and Tenant B (e.g., Target) both use your ML demand forecasting platform. Their data *cannot* leak into each other's models, and Tenant A running a massive inference batch job *cannot* slow down Tenant B's real-time API.

## 2. Data Isolation (The Security Mandate)
This is the most critical aspect. If Tenant A's data leaks into Tenant B's model, it's a catastrophic security breach.

### Approach A: Physical Isolation (Siloed)
*   Every tenant gets their own dedicated database, their own S3 buckets, and their own Kubernetes cluster.
*   *Pros:* Maximum security, easy to delete a customer's data completely.
*   *Cons:* Astronomically expensive and an operational nightmare to maintain 500 different infrastructure stacks.

### Approach B: Logical Isolation (Pooled)
*   All tenants share the same massive database. Every single row in every table has a mandatory `tenant_id` column.
*   *Security Mechanism:* **Row-Level Security (RLS)** in databases like Postgres or Snowflake ensures that a query made by Tenant A's API token can physically never return rows where `tenant_id = B`.
*   *Pros:* Highly cost-effective and scalable.

## 3. Model Architecture for Multi-Tenancy

### Approach A: One Global Model
*   Train a single massive model on the combined data of all tenants (using `tenant_id` as an input feature).
*   *Pros:* The model benefits from massive data volume (transfer learning). A new tenant with no data gets good baseline predictions instantly (solving the Cold Start problem).
*   *Cons:* High risk of implicit data leakage. The model might memorize a rare pattern from Tenant A and use it to predict for Tenant B. Some enterprise clients legally forbid their data from being used to train a shared model.

### Approach B: One Model Per Tenant
*   Train 500 separate XGBoost models, one for each tenant, using only their specific data.
*   *Pros:* Strict mathematical data isolation. The model perfectly captures the unique nuances of that specific client.
*   *Cons:* How do you serve 500 models? You can't run 500 separate API servers.
    *   **The Solution (Model Multiplexing):** Use a server like **Triton Inference Server** or **Ray Serve**. You deploy one API endpoint. The server holds all 500 model weights in a fast storage backend (S3/Redis). When a request comes in for `Tenant=A`, the server dynamically loads Tenant A's specific weights into GPU RAM in milliseconds, runs the inference, and evicts it.

## 4. RAG Multi-Tenancy (LLMs)
How do you build a multi-tenant RAG chatbot?
*   **Vector DB Namespaces:** Modern vector databases (Pinecone, Milvus) have native support for "Namespaces" or "Collections."
*   When ingesting Tenant A's PDFs, you insert the vectors into the `tenant_A` namespace.
*   When querying, you pass the `tenant_id` to the Vector DB API. The database mathematically restricts the search to *only* that namespace, ensuring 100% data isolation at the infrastructure level.

## 5. The "Noisy Neighbor" Problem
*   *Scenario:* Tenant A launches a massive marketing campaign and sends 10,000 requests/sec to the shared ML inference server, using up all the GPU compute. Tenant B's routine requests start timing out.
*   *Solution:* **Rate Limiting & Quotas.** Implement strict API Gateways (like Kong or AWS API Gateway) that enforce a maximum Requests-Per-Minute quota per `tenant_id`. If Tenant A exceeds it, they get a `429 Too Many Requests` error, protecting the GPU resources for Tenant B.

## Interview Strategy
"In a B2B ML platform, security and resource isolation are paramount. I would design a **Logically Isolated Data Layer** using Row-Level Security to save costs, but a **Physically Isolated Model Layer** (training a distinct model artifact for each tenant) to guarantee zero data leakage. To serve these efficiently without massive overhead, I would utilize a **Model Multiplexer** like Triton Server to dynamically load tenant-specific weights into memory on-demand."