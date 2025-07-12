# Leadpoet Intent Model v1.1

Leadpoet returns the **K** leads most likely to purchase any B2B product specified in a free-text prompt, achieving < **400 ms** P95 latency at ≤ **$0.002** COGS per lead under 250 QPS load.

## 🚀 Quick Start

### Prerequisites

- **AWS CLI** v2.8+ configured with appropriate permissions
- **Terraform** v1.5+ 
- **kubectl** v1.28+
- **Docker** v20.10+
- **Python** 3.11+
- **Node.js** 18+ (for frontend)

### Local Development Setup

#### 1. Clone and Configure

```bash
git clone <repository-url>
cd leadpoet
```

#### 2. Set Environment Variables

Create `.env` file:
```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_PROFILE=leadpoet

# Database
DB_ENDPOINT=your-rds-endpoint
DB_PASSWORD=your-db-password

# Redis
REDIS_ENDPOINT=your-redis-endpoint

# Kafka
KAFKA_BOOTSTRAP_BROKERS=your-kafka-brokers

# API Keys
OPENAI_API_KEY=your-openai-key
PDL_API_KEY=your-pdl-key
CLEARBIT_API_KEY=your-clearbit-key
```

#### 3. Setup Terraform Backend (First Time Only)

**Option A: Automated Setup (Recommended)**
```bash
cd terraform
chmod +x setup-backend.sh
./setup-backend.sh
```

**Option B: Manual Setup**
```bash
# Create S3 bucket for Terraform state
aws s3api create-bucket \
    --bucket leadpoet-terraform-state \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket leadpoet-terraform-state \
    --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
    --table-name leadpoet-terraform-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

#### 4. Deploy Infrastructure

```bash
# Initialize Terraform with backend
cd terraform
terraform init

# Plan deployment
terraform plan

# Apply infrastructure
terraform apply
```

#### 5. Deploy Kubernetes Resources

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name leadpoet-prod

# Apply namespace and configs
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
```

## 🏗️ Infrastructure Components

### Terraform Backend
- **S3 Bucket**: `leadpoet-terraform-state` for state storage
- **DynamoDB Table**: `leadpoet-terraform-locks` for state locking
- **Encryption**: AES256 server-side encryption
- **Versioning**: Enabled for state history
- **Public Access**: Blocked for security

### EKS Cluster
- **Kubernetes version**: 1.28
- **Node types**: t3.medium (3-10 nodes)
- **Auto-scaling**: Enabled with HPA

### Database
- **PostgreSQL**: 16.1 with TimescaleDB
- **pgvector**: For ANN similarity search
- **Instance**: db.t3.medium
- **Storage**: 100GB GP3 (auto-scaling to 1TB)

### Caching
- **Redis**: 7.x with LRU eviction
- **Cluster**: 2 nodes with failover
- **Instance**: cache.t3.micro

### Message Bus
- **Kafka**: MSK 3.5.1
- **Brokers**: 3 nodes (kafka.t3.small)
- **Topics**: `miner.snippets`, `outcome.events`

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| P95 Latency | < 400ms | TBD |
| P99 Latency | < 550ms | TBD |
| Average Cost | ≤ $0.002 | TBD |
| P99 Cost | ≤ $0.004 | TBD |
| QPS | 250 (1000 burst) | TBD |

## 🔧 Development

### Running the miner on Windows (PowerShell 7)

1. **Install PowerShell 7**
   ```powershell
   # Using winget
   winget install Microsoft.PowerShell
   
   # Or download from GitHub releases
   ```

2. **Set execution policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Run development script**
   ```powershell
   # Navigate to project directory
   cd "C:\path\to\leadpoet"
   
   # Run the development script
   .\dev\run.ps1
   ```

### API Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access Swagger docs
open http://localhost:8000/docs
```

### Database Migrations

```bash
# Run migrations
python -m alembic upgrade head

# Create new migration
python -m alembic revision --autogenerate -m "description"
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_scoring.py
```

### Load Testing
```bash
# Run load test at 250 QPS
python scripts/load_test.py --qps 250 --duration 300

# Validate latency targets
python scripts/validate_sla.py
```

### Cost Validation
```bash
# Run cost test
python scripts/cost_test.py --leads 1000

# Generate cost report
python scripts/cost_report.py
```

## 📈 Monitoring

### Prometheus Metrics
- **Endpoint**: `/metrics`
- **Key metrics**: 
  - `lead_cost_usd`
  - `llm_hit_rate`
  - `pipeline_latency_seconds`
  - `miner_reputation`

### Grafana Dashboards
- **Latency Dashboard**: Real-time P95/P99 monitoring
- **Cost Dashboard**: Per-lead cost tracking
- **Miner Dashboard**: Reputation and quality metrics

### Alerts
- P95 latency > 400ms
- Average cost > $0.002
- Miner reputation < 0.4
- LLM error rate > 20%

## 🔐 Security

### Secrets Management
- **AWS Secrets Manager**: For production secrets
- **Kubernetes Secrets**: For runtime configuration
- **TLS 1.3**: Enforced at API gateway
- **IAM**: Least-privilege access

### Data Protection
- **Encryption at rest**: All data encrypted
- **Encryption in transit**: TLS for all connections
- **No B2C PII**: Only B2B data processed
- **GDPR compliant**: No personal data stored

### Terraform State Security
- **S3 Encryption**: AES256 server-side encryption
- **State Locking**: DynamoDB table prevents concurrent modifications
- **Access Control**: IAM policies restrict backend access
- **Versioning**: State history preserved for rollback

## 🚀 Deployment

### CI/CD Pipeline
- **GitHub Actions**: Automated testing and deployment
- **Terraform**: Infrastructure as code with remote state
- **Helm**: Kubernetes package management
- **Blue-green**: Zero-downtime deployments

### Environments
- **Staging**: Automated deployment on PR
- **Production**: Manual approval required
- **Rollback**: Automated on failure detection

### State Management
- **Remote State**: Stored in S3 with DynamoDB locking
- **Collaboration**: Multiple team members can work safely
- **Backup**: State versioning enabled
- **Recovery**: Easy state restoration from S3

## 📋 Roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| W0-1 | BM25 scorer, retrieval baseline | 🔄 In Progress |
| W2 | Reputation & spam throttle | ⏳ Planned |
| W3 | Tech-stack-churn ETL | ⏳ Planned |
| W4 | ANN exemplar search | ⏳ Planned |
| W5 | LightGBM distillation | ⏳ Planned |
| W6 | A/B test & GA launch | ⏳ Planned |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Write unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before PR

## 📞 Support

- **All Questions**: hello@leadpoet.com


## 📄 License

This project is proprietary to Leadpoet.com. All rights reserved.

---

> _Built with ❤️ by the Leadpoet Engineering Team_ 