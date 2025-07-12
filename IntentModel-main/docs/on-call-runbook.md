# Leadpoet Intent Model API - On-Call Runbook

## Overview

This runbook provides step-by-step procedures for handling incidents and operational issues with the Leadpoet Intent Model API. It covers circuit-breaker management, performance issues, and emergency response procedures.

## Emergency Contacts

### Primary On-Call
- **Phone**: [Internal Directory](https://company.leadpoet.com/oncall) or PagerDuty mobile app
- **Slack**: @oncall-leadpoet
- **PagerDuty**: leadpoet-api-oncall

### Escalation Contacts
- **SRE Lead**: @sre-lead (escalate after 15 minutes)
- **Engineering Manager**: @eng-manager (escalate after 30 minutes)
- **CTO**: @cto (escalate after 1 hour)

## Quick Status Check

### Health Check Commands
```bash
# Check API health
curl -f https://api.leadpoet.com/healthz

# Check metrics
curl https://api.leadpoet.com/metrics

# Check Kubernetes pods
kubectl get pods -n leadpoet-prod -l app.kubernetes.io/name=leadpoet-api

# Check service status
kubectl get svc -n leadpoet-prod leadpoet-api
```

### Key Metrics to Monitor
- **Response Time**: P95 < 400ms, P99 < 550ms
- **Error Rate**: < 1%
- **Cost per Lead**: < $0.002
- **LLM Call Ratio**: < 30%
- **Circuit Breaker Status**: Closed (normal operation)

## Circuit Breaker Management

### Circuit Breaker States

1. **CLOSED**: Normal operation, requests pass through
2. **OPEN**: Circuit is open, requests fail fast
3. **HALF_OPEN**: Testing if service has recovered

### Circuit Breaker Configuration

```python
# Current settings in app/core/circuit_breaker.py
CIRCUIT_BREAKER_CONFIG = {
    'failure_threshold': 20,  # 20% error rate
    'recovery_timeout': 60,   # 60 seconds
    'expected_exception': (Exception,),
    'fallback_function': fallback_scoring
}
```

### Circuit Breaker Status Check

```bash
# Check circuit breaker status via metrics
curl https://api.leadpoet.com/metrics | grep circuit_breaker

# Expected output:
# leadpoet_circuit_breaker_state{service="llm"} 0  # 0=CLOSED, 1=OPEN, 2=HALF_OPEN
# leadpoet_circuit_breaker_failures_total{service="llm"} 15
```

### Circuit Breaker Procedures

#### 1. Circuit Breaker OPEN (Emergency)

**Symptoms:**
- High error rate (>20%)
- LLM service unavailable
- Requests failing fast
- Alerts: "Circuit breaker OPEN"

**Immediate Actions:**
1. **Acknowledge the alert** in PagerDuty/Slack
2. **Check LLM service status**:
   ```bash
   # Check OpenAI API status
   curl -I https://api.openai.com/v1/models
   
   # Check API key validity
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

3. **Verify fallback is working**:
   ```bash
   # Test query with circuit breaker open
   curl -X POST https://api.leadpoet.com/query \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "test query", "limit": 5}'
   ```

4. **Check logs for root cause**:
   ```bash
   kubectl logs -f deployment/leadpoet-api -n leadpoet-prod | grep -i "circuit\|llm\|error"
   ```

**Resolution Steps:**
1. **If LLM service is down**: Wait for service recovery, monitor status
2. **If API key issues**: Rotate API key via AWS Secrets Manager
3. **If rate limiting**: Implement backoff strategy
4. **If persistent issues**: Consider manual circuit breaker reset

#### 2. Manual Circuit Breaker Reset

**⚠️ WARNING: Only use if you're confident the underlying issue is resolved**

```bash
# Restart the deployment to reset circuit breaker
kubectl rollout restart deployment/leadpoet-api -n leadpoet-prod

# Monitor recovery
kubectl rollout status deployment/leadpoet-api -n leadpoet-prod

# Verify circuit breaker is closed
curl https://api.leadpoet.com/metrics | grep circuit_breaker
```

#### 3. Circuit Breaker HALF_OPEN

**Symptoms:**
- Circuit breaker testing recovery
- Some requests succeeding, some failing
- Transitional state

**Actions:**
1. **Monitor closely** for 5-10 minutes
2. **Check error rates** are decreasing
3. **Verify LLM service** is stable
4. **No manual intervention** needed unless issues persist

## Performance Issues

### High Latency (>400ms P95)

**Symptoms:**
- Response times exceeding SLA
- User complaints about slow API
- Alerts: "High latency detected"

**Investigation Steps:**
1. **Check resource usage**:
   ```bash
   kubectl top pods -n leadpoet-prod
   kubectl describe hpa leadpoet-api -n leadpoet-prod
   ```

2. **Check database performance**:
   ```bash
   # Check database connections
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check slow queries
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     psql $DATABASE_URL -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
   ```

3. **Check Redis performance**:
   ```bash
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     redis-cli -h $REDIS_HOST info memory
   ```

**Resolution Steps:**
1. **Scale up if needed**:
   ```bash
   kubectl scale deployment leadpoet-api -n leadpoet-prod --replicas=5
   ```

2. **Check for cache issues**:
   ```bash
   # Clear Redis cache if needed
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     redis-cli -h $REDIS_HOST FLUSHDB
   ```

3. **Restart if necessary**:
   ```bash
   kubectl rollout restart deployment/leadpoet-api -n leadpoet-prod
   ```

### High Cost per Lead (>$0.002)

**Symptoms:**
- Cost metrics exceeding budget
- Alerts: "High cost per lead"
- Query quality reputation dropping

**Investigation Steps:**
1. **Check LLM call patterns**:
   ```bash
   curl https://api.leadpoet.com/metrics | grep llm
   ```

2. **Review recent queries**:
   ```bash
   # Check query logs
   kubectl logs deployment/leadpoet-api -n leadpoet-prod --tail=100 | grep "cost\|llm"
   ```

3. **Check cache hit rates**:
   ```bash
   curl https://api.leadpoet.com/metrics | grep cache
   ```

**Resolution Steps:**
1. **Enable aggressive caching**:
   ```bash
   # Update cache TTL
   kubectl patch configmap leadpoet-config -n leadpoet-prod \
     --patch '{"data":{"CACHE_TTL":"3600"}}'
   ```

2. **Check for problematic queries**:
   ```bash
   # Monitor query patterns
   kubectl logs -f deployment/leadpoet-api -n leadpoet-prod | grep "query"
   ```

3. **Consider throttling** if costs persist

## Service Outages

### Complete API Down

**Symptoms:**
- All requests failing
- Health check failing
- No pods running

**Emergency Actions:**
1. **Check pod status**:
   ```bash
   kubectl get pods -n leadpoet-prod
   kubectl describe pods -n leadpoet-prod -l app.kubernetes.io/name=leadpoet-api
   ```

2. **Check events**:
   ```bash
   kubectl get events -n leadpoet-prod --sort-by='.lastTimestamp'
   ```

3. **Check resource constraints**:
   ```bash
   kubectl describe nodes | grep -A 10 "Conditions:"
   ```

4. **Restart deployment**:
   ```bash
   kubectl rollout restart deployment/leadpoet-api -n leadpoet-prod
   kubectl rollout status deployment/leadpoet-api -n leadpoet-prod
   ```

### Database Issues

**Symptoms:**
- Database connection errors
- Query timeouts
- Data consistency issues

**Actions:**
1. **Check database connectivity**:
   ```bash
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     psql $DATABASE_URL -c "SELECT 1;"
   ```

2. **Check database health**:
   ```bash
   # Check for locks
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     psql $DATABASE_URL -c "SELECT * FROM pg_locks WHERE NOT granted;"
   
   # Check for long-running queries
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     psql $DATABASE_URL -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"
   ```

3. **Contact database team** if issues persist

### Redis Issues

**Symptoms:**
- Cache misses increasing
- Redis connection errors
- Performance degradation

**Actions:**
1. **Check Redis connectivity**:
   ```bash
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     redis-cli -h $REDIS_HOST ping
   ```

2. **Check Redis memory**:
   ```bash
   kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- \
     redis-cli -h $REDIS_HOST info memory
   ```

3. **Restart Redis if needed**:
   ```bash
   kubectl rollout restart deployment/redis -n leadpoet-prod
   ```

## Monitoring and Alerting

### Key Alerts

| Alert | Severity | Action |
|-------|----------|--------|
| Circuit Breaker OPEN | Critical | Follow circuit breaker procedures |
| High Latency (>400ms) | Warning | Check performance |
| High Cost (>$0.002) | Warning | Investigate cost issues |
| High Error Rate (>1%) | Critical | Check logs and health |
| Service Down | Critical | Emergency restart |

### Dashboard Links
- **Grafana**: https://grafana.leadpoet.com/d/leadpoet-api
- **Prometheus**: https://prometheus.leadpoet.com
- **Datadog**: https://app.datadoghq.com/dashboard/leadpoet-api

## Post-Incident Procedures

### 1. Incident Documentation
- **Update incident log** in Confluence
- **Record timeline** of events
- **Document root cause** and resolution
- **Update runbook** if procedures were missing

### 2. Follow-up Actions
- **Schedule post-mortem** within 24 hours
- **Review monitoring** and alerting
- **Update procedures** based on learnings
- **Communicate** to stakeholders

### 3. Prevention
- **Implement fixes** to prevent recurrence
- **Update monitoring** if gaps found
- **Train team** on new procedures
- **Review runbook** for completeness

## Useful Commands

### Debugging Commands
```bash
# Get pod logs
kubectl logs -f deployment/leadpoet-api -n leadpoet-prod

# Execute commands in pod
kubectl exec -it deployment/leadpoet-api -n leadpoet-prod -- /bin/bash

# Check resource usage
kubectl top pods -n leadpoet-prod

# Check events
kubectl get events -n leadpoet-prod --sort-by='.lastTimestamp'

# Check service endpoints
kubectl get endpoints leadpoet-api -n leadpoet-prod
```

### Rollback Commands
```bash
# Rollback to previous version
kubectl rollout undo deployment/leadpoet-api -n leadpoet-prod

# Check rollout history
kubectl rollout history deployment/leadpoet-api -n leadpoet-prod

# Rollback to specific revision
kubectl rollout undo deployment/leadpoet-api -n leadpoet-prod --to-revision=2
```

### Scaling Commands
```bash
# Scale up
kubectl scale deployment leadpoet-api -n leadpoet-prod --replicas=5

# Scale down
kubectl scale deployment leadpoet-api -n leadpoet-prod --replicas=3

# Check HPA status
kubectl describe hpa leadpoet-api -n leadpoet-prod
```

## Emergency Contacts Summary

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-Call Engineer | @oncall-leadpoet | Immediate |
| SRE Lead | @sre-lead | 15 minutes |
| Engineering Manager | @eng-manager | 30 minutes |
| CTO | @cto | 1 hour |

---

*Last Updated: June 2025*
*Version: 1.1* 