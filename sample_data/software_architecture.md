# Software Architecture Reference – Atlas Platform

## 1. System Overview

Atlas is a microservices-based platform composed of independently deployable
services communicating over REST and an asynchronous message bus (RabbitMQ).

---

## 2. Services

### 2.1 API Gateway
- **Technology:** Kong Gateway 3.x
- **Responsibilities:** Rate limiting, authentication, SSL termination, routing
- **Rate limits:** 1,000 requests/minute per API key; burst up to 2,000
- **Authentication:** OAuth 2.0 with JWT bearer tokens (RS256 signed)

### 2.2 Document Service
- **Technology:** Python 3.11 / FastAPI
- **Responsibilities:** Document CRUD, versioning, full-text indexing
- **Storage:** PostgreSQL (metadata) + MinIO object store (binary files)
- **Max file size:** 100 MB per upload

### 2.3 Search Service
- **Technology:** Python 3.11 / FastAPI + FAISS
- **Responsibilities:** Vector embeddings, semantic search, re-ranking
- **Embedding model:** `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Index type:** FAISS `IndexFlatIP` with L2 normalization (cosine similarity)

### 2.4 Notification Service
- **Technology:** Node.js 20 / Express
- **Responsibilities:** Email, Slack, and webhook notifications
- **Retry policy:** Exponential backoff, max 5 retries, dead-letter queue after 24 h

### 2.5 Analytics Service
- **Technology:** Python 3.11 / FastAPI + ClickHouse
- **Responsibilities:** Usage metrics, query logs, performance dashboards
- **Retention:** 90 days hot, 2 years cold (S3 Glacier)

---

## 3. Data Flow

```
Client → API Gateway → Document Service ──┐
                                           ├─→ PostgreSQL
                     → Search Service ────┘
                     → Notification Service → RabbitMQ → Email/Slack
                     → Analytics Service → ClickHouse
```

---

## 4. Deployment

### 4.1 Container Orchestration
- **Platform:** Kubernetes 1.29 on AWS EKS
- **Node groups:** t3.xlarge (4 vCPU / 16 GB) for application pods;
  r6i.2xlarge (8 vCPU / 64 GB) for ML/FAISS workloads
- **Auto-scaling:** Horizontal Pod Autoscaler (HPA); min 2, max 10 replicas
  per service

### 4.2 CI/CD Pipeline
1. Developer pushes to feature branch
2. GitHub Actions runs: lint → unit tests → integration tests → build Docker image
3. Image pushed to AWS ECR with git SHA tag
4. ArgoCD detects new image → deploys to staging environment
5. Smoke tests pass → manual approval required for production
6. ArgoCD syncs production cluster

### 4.3 Infrastructure as Code
- All infrastructure defined in Terraform (AWS provider)
- State stored in S3 with DynamoDB locking
- Secrets managed by AWS Secrets Manager; injected as Kubernetes secrets

---

## 5. Security

### 5.1 Network Security
- VPC with private subnets for all application workloads
- Public subnets for load balancers only
- Security groups: least-privilege ingress/egress rules
- All inter-service communication encrypted with mTLS (cert-manager + Linkerd)

### 5.2 Data Security
- Data at rest: AES-256 encryption (AWS KMS managed keys)
- Data in transit: TLS 1.3 minimum
- PII fields encrypted at application layer before persistence
- Database: RDS PostgreSQL with automatic minor version upgrades enabled

### 5.3 Vulnerability Management
- Container images scanned with Trivy on every build
- Dependabot alerts for dependency CVEs
- Penetration testing: annual third-party + quarterly internal

---

## 6. Observability

### 6.1 Metrics
- Prometheus + Grafana stack
- Key SLIs: request latency P50/P95/P99, error rate, throughput
- SLO: 99.5% availability, P95 latency < 200 ms

### 6.2 Logging
- Structured JSON logs (Loguru / Winston)
- Collected by Fluent Bit → OpenSearch cluster
- Log retention: 30 days hot, 1 year cold

### 6.3 Tracing
- OpenTelemetry SDK in all services
- Traces exported to Tempo (Grafana stack)
- Sampling rate: 10% in production, 100% in staging

---

## 7. Disaster Recovery

### 7.1 Backup Policy
- PostgreSQL: continuous WAL archiving + daily snapshots to S3 (cross-region copy)
- MinIO: versioned S3-compatible buckets with cross-region replication
- Recovery point objective (RPO): 1 hour
- Recovery time objective (RTO): 4 hours

### 7.2 Runbook
1. Declare incident (PagerDuty)
2. Assess blast radius
3. Initiate DR procedure from `docs/runbooks/disaster-recovery.md`
4. Restore from most recent verified backup
5. Validate data integrity checksums
6. Notify stakeholders via status page (Statuspage.io)
