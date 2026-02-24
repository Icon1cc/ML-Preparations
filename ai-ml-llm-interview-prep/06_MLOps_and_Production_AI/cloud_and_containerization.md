# Cloud Deployment and Containerization for ML

## Docker for ML
Benefits:
- reproducible runtime
- dependency isolation
- portable deployment artifacts

## Kubernetes patterns
- Deployment for stateless model APIs
- HPA autoscaling by CPU/GPU/latency
- Separate training and inference namespaces

## Cloud platform overview
- AWS: SageMaker ecosystem, EC2/EKS flexibility.
- GCP: Vertex AI + BigQuery integration.
- Azure: Azure ML + enterprise governance integration.

## Managed vs DIY tradeoff
Managed:
- faster setup
- less infra burden
DIY:
- deeper control
- potentially lower long-run cost at scale

## Cost controls
- spot/preemptible for training
- right-size instances
- quantization + batching for serving

## Interview questions
1. How deploy a model on AWS quickly and safely?
2. SageMaker vs Kubernetes custom stack tradeoffs?
3. How control cloud spend for LLM workloads?

## Minimal Dockerfile pattern
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
