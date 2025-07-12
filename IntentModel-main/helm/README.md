# Leadpoet Intent Model - Helm Charts

This directory contains Helm charts for deploying the Leadpoet Intent Model API to Kubernetes.

## Overview

The Leadpoet Intent Model is a B2B lead scoring and ranking API that achieves < 400ms P95 latency at ≤ $0.002 COGS per lead under 250 QPS load.

## Charts

### leadpoet-api

The main API service chart that deploys the FastAPI application with all necessary Kubernetes resources.

## Prerequisites

- Kubernetes cluster (EKS recommended)
- Helm 3.x
- kubectl configured to access your cluster
- Docker registry access (for container images)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the image
docker build -t leadpoet/intent-model-api:1.1.0 .

# Push to your registry
docker push leadpoet/intent-model-api:1.1.0
```

### 2. Deploy Using Helm

```bash
# Add the chart repository (if using a remote repository)
helm repo add leadpoet https://charts.leadpoet.com
helm repo update

# Install the chart
helm install leadpoet-api ./helm/leadpoet-api -n leadpoet-prod

# Or use the deployment script
./helm/deploy.sh
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n leadpoet-prod -l app.kubernetes.io/name=leadpoet-api

# Check service status
kubectl get svc -n leadpoet-prod -l app.kubernetes.io/name=leadpoet-api

# Check HPA status
kubectl get hpa -n leadpoet-prod -l app.kubernetes.io/name=leadpoet-api
```

## Configuration

### Infrastructure Configuration

The chart now uses Helm templating for dynamic host construction instead of shell-style variable interpolation. This allows for proper variable expansion during deployment.

#### Database and Redis Host Configuration

Configure your infrastructure endpoints in the values file:

```yaml
# values-production.yaml
infrastructure:
  database:
    endpoint: "your-db-cluster.region.rds.amazonaws.com"  # Your actual DB endpoint
    prefix: "leadpoet-prod"  # Database name prefix
  redis:
    endpoint: "your-redis-cluster.region.cache.amazonaws.com"  # Your actual Redis endpoint
```

Or override during deployment:

```bash
helm install leadpoet-api ./helm/leadpoet-api \
  --namespace leadpoet-prod \
  --set infrastructure.database.endpoint=your-db-cluster.region.rds.amazonaws.com \
  --set infrastructure.redis.endpoint=your-redis-cluster.region.cache.amazonaws.com
```

The chart will automatically construct the full hostnames:
- **DB_HOST**: `leadpoet-prod.your-db-cluster.region.rds.amazonaws.com`
- **REDIS_HOST**: `your-redis-cluster.region.cache.amazonaws.com`

#### Environment-Specific Examples

**Development Environment:**
```yaml
infrastructure:
  database:
    endpoint: "localhost"
    prefix: "leadpoet-dev"
  redis:
    endpoint: "localhost"
```

**Staging Environment:**
```yaml
infrastructure:
  database:
    endpoint: "leadpoet-staging.cluster.local"
    prefix: "leadpoet-staging"
  redis:
    endpoint: "redis-staging.cluster.local"
```

**Production Environment:**
```yaml
infrastructure:
  database:
    endpoint: "leadpoet-prod.cluster.region.rds.amazonaws.com"
    prefix: "leadpoet-prod"
  redis:
    endpoint: "redis-prod.cluster.region.cache.amazonaws.com"
```

### Values File

Create a custom values file to override default settings:

```yaml
# values-production.yaml
image:
  repository: your-registry/leadpoet/intent-model-api
  tag: "1.1.0"

deployment:
  replicas: 5
  resources:
    limits:
      cpu: 3000m
      memory: 6Gi
    requests:
      cpu: 1000m
      memory: 2Gi

hpa:
  enabled: true
  minReplicas: 3
  maxReplicas: 30
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

env:
  APP_ENV: "production"
  LOG_LEVEL: "INFO"
  LATENCY_P95_THRESHOLD: "400"
  LATENCY_P99_THRESHOLD: "550"
  COST_AVG_THRESHOLD: "0.002"
  COST_P99_THRESHOLD: "0.004"
```

### Environment Variables

The chart supports the following environment variables:

#### Database Configuration
- `DB_HOST`: PostgreSQL host (constructed from `infrastructure.database.prefix` + `infrastructure.database.endpoint`)
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: Database name
- `DB_PASSWORD`: Database password (from secret)

#### Redis Configuration
- `REDIS_HOST`: Redis host (from `infrastructure.redis.endpoint`)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)

#### Application Configuration
- `APP_ENV`: Application environment (production/staging/development)
- `LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING/ERROR)

#### Performance Configuration
- `LATENCY_P95_THRESHOLD`: P95 latency threshold in milliseconds
- `LATENCY_P99_THRESHOLD`: P99 latency threshold in milliseconds
- `COST_AVG_THRESHOLD`: Average cost per lead threshold
- `COST_P99_THRESHOLD`: P99 cost per lead threshold

#### Scoring Configuration
- `BM25_THRESHOLD`: BM25 scoring threshold
- `TIME_DECAY_TAU`: Time decay parameter
- `MAX_LEADS_PER_QUERY`: Maximum leads per query
- `MIN_LEADS_PER_QUERY`: Minimum leads per query

#### LLM Configuration
- `LLM_MAX_TOKENS`: Maximum tokens for LLM calls
- `LLM_TEMPERATURE`: LLM temperature setting
- `LLM_MAX_CALL_RATIO`: Maximum LLM call ratio

## Deployment Strategies

### Standard Deployment

```bash
helm install leadpoet-api ./helm/leadpoet-api \
  --namespace leadpoet-prod \
  --values values-production.yaml
```

### Blue-Green Deployment

For zero-downtime deployments, use the blue-green deployment script:

```bash
./helm/blue-green-deploy.sh --namespace leadpoet-prod
```

This script:
1. Deploys to the inactive environment (blue/green)
2. Performs health checks
3. Switches traffic to the new environment
4. Cleans up the old environment

### Rolling Update

```bash
helm upgrade leadpoet-api ./helm/leadpoet-api \
  --namespace leadpoet-prod \
  --values values-production.yaml
```

## Monitoring and Observability

### Prometheus Metrics

The chart includes a ServiceMonitor for Prometheus monitoring:

```yaml
monitoring:
  serviceMonitor:
    enabled: true
    interval: 30s
    scrapeTimeout: 10s
    path: /metrics
    port: http
```

### Health Checks

The deployment includes liveness and readiness probes:

- **Liveness Probe**: `/healthz` endpoint
- **Readiness Probe**: `/healthz` endpoint
- **Health Check**: 30s interval, 10s timeout

### Horizontal Pod Autoscaler

The HPA automatically scales pods based on CPU and memory usage:

```yaml
hpa:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## Security

### Pod Security

The chart implements security best practices:

- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- Security contexts

### Secrets Management

API keys and sensitive data are managed through Kubernetes secrets:

```yaml
secret:
  name: "leadpoet-secrets"
  create: false  # Use existing secret
```

Required secrets:
- `OPENAI_API_KEY`: OpenAI API key
- `PDL_API_KEY`: PDL API key
- `CLEARBIT_API_KEY`: Clearbit API key
- `DB_PASSWORD`: Database password

### Network Policies

The chart includes configurable NetworkPolicies for enhanced security:

```yaml
networkPolicy:
  enabled: true
  # Namespace labels for different environments
  namespaceLabels:
    ingress:
      name: "ingress-nginx"  # Ingress controller namespace
    monitoring:
      name: "monitoring"     # Monitoring stack namespace
    database:
      name: "database"       # Database namespace
    redis:
      name: "redis"          # Redis namespace
  ingressRules: []           # Additional ingress rules
  egressRules: []            # Additional egress rules
```

#### Environment-Specific Namespace Labels

**Development Environment:**
```yaml
networkPolicy:
  namespaceLabels:
    ingress:
      name: "ingress-nginx-dev"
    monitoring:
      name: "monitoring-dev"
    database:
      name: "postgres-dev"
    redis:
      name: "redis-dev"
```

**Staging Environment:**
```yaml
networkPolicy:
  namespaceLabels:
    ingress:
      name: "ingress-nginx-staging"
    monitoring:
      name: "monitoring-staging"
    database:
      name: "postgres-staging"
    redis:
      name: "redis-staging"
```

**Production Environment:**
```yaml
networkPolicy:
  namespaceLabels:
    ingress:
      name: "ingress-nginx"
    monitoring:
      name: "monitoring"
    database:
      name: "postgres-prod"
    redis:
      name: "redis-prod"
```

#### Override During Deployment

```bash
helm install leadpoet-api ./helm/leadpoet-api \
  --namespace leadpoet-prod \
  --set networkPolicy.namespaceLabels.database.name=postgres-prod \
  --set networkPolicy.namespaceLabels.redis.name=redis-prod
```

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl describe pod <pod-name> -n leadpoet-prod
   kubectl logs <pod-name> -n leadpoet-prod
   ```

2. **Health Check Failures**
   ```bash
   kubectl port-forward svc/leadpoet-api 8000:80 -n leadpoet-prod
   curl http://localhost:8000/healthz
   ```

3. **Resource Issues**
   ```bash
   kubectl top pods -n leadpoet-prod
   kubectl describe hpa leadpoet-api -n leadpoet-prod
   ```

### Debug Commands

```bash
# Check deployment status
kubectl rollout status deployment/leadpoet-api -n leadpoet-prod

# View deployment history
kubectl rollout history deployment/leadpoet-api -n leadpoet-prod

# Rollback to previous version
kubectl rollout undo deployment/leadpoet-api -n leadpoet-prod

# Check service endpoints
kubectl get endpoints leadpoet-api -n leadpoet-prod
```

## Performance Tuning

### Resource Limits

Adjust resource limits based on your workload:

```yaml
deployment:
  resources:
    limits:
      cpu: 2000m      # 2 CPU cores
      memory: 4Gi     # 4GB RAM
    requests:
      cpu: 500m       # 0.5 CPU cores
      memory: 1Gi     # 1GB RAM
```

### Scaling Configuration

Optimize HPA settings for your traffic patterns:

```yaml
hpa:
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60   # 1 minute
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review application logs: `kubectl logs -f deployment/leadpoet-api -n leadpoet-prod`
3. Check monitoring dashboards for performance metrics
4. Contact the Leadpoet team at dev@leadpoet.com 