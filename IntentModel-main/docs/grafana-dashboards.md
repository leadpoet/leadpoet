# Grafana Dashboards for Leadpoet Intent Model v1.1

This document contains Grafana dashboard configurations for monitoring the Leadpoet Intent Model API performance, costs, and business metrics.

## Dashboard Overview

### 1. Leadpoet API Performance Dashboard
**Dashboard ID:** `leadpoet-api-performance`  
**Description:** Real-time monitoring of API performance, latency, and throughput

### 2. Leadpoet Cost Analysis Dashboard  
**Dashboard ID:** `leadpoet-cost-analysis`  
**Description:** Cost tracking and analysis per query and lead

### 3. Leadpoet Pipeline Performance Dashboard
**Dashboard ID:** `leadpoet-pipeline-performance`  
**Description:** Detailed pipeline stage performance and bottlenecks

---

## Dashboard 1: API Performance Dashboard

### Panel 1: Request Rate (QPS)
```json
{
  "title": "Request Rate (QPS)",
  "type": "stat",
  "targets": [
    {
      "expr": "rate(requests_total[5m])",
      "legendFormat": "{{endpoint}} - {{method}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "displayMode": "gradient-gauge"
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "red", "value": 250}
        ]
      }
    }
  }
}
```

### Panel 2: Response Time Percentiles
```json
{
  "title": "Response Time Percentiles",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))",
      "legendFormat": "P95"
    },
    {
      "expr": "histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m]))",
      "legendFormat": "P99"
    },
    {
      "expr": "histogram_quantile(0.50, rate(request_latency_seconds_bucket[5m]))",
      "legendFormat": "P50"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 0.4},
          {"color": "red", "value": 0.55}
        ]
      },
      "unit": "s"
    }
  }
}
```

### Panel 3: Error Rate
```json
{
  "title": "Error Rate",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(requests_total{http_status=~\"5..\"}[5m])",
      "legendFormat": "5xx Errors"
    },
    {
      "expr": "rate(requests_total{http_status=~\"4..\"}[5m])",
      "legendFormat": "4xx Errors"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 0.01},
          {"color": "red", "value": 0.05}
        ]
      }
    }
  }
}
```

---

## Dashboard 2: Cost Analysis Dashboard

### Panel 1: Cost per Lead
```json
{
  "title": "Cost per Lead (USD)",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(lead_cost_usd_bucket[5m]))",
      "legendFormat": "P95 Cost per Lead"
    },
    {
      "expr": "histogram_quantile(0.50, rate(lead_cost_usd_bucket[5m]))",
      "legendFormat": "P50 Cost per Lead"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 0.002},
          {"color": "red", "value": 0.004}
        ]
      },
      "unit": "currencyUSD"
    }
  }
}
```

### Panel 2: LLM Usage Rate
```json
{
  "title": "LLM Usage Rate",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(llm_hit_total[5m])",
      "legendFormat": "LLM Calls per Second"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 75},
          {"color": "red", "value": 100}
        ]
      }
    }
  }
}
```

### Panel 3: Total Cost Over Time
```json
{
  "title": "Total Cost Over Time (USD)",
  "type": "timeseries",
  "targets": [
    {
      "expr": "sum(increase(lead_cost_usd_sum[1h]))",
      "legendFormat": "Hourly Cost"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 10},
          {"color": "red", "value": 20}
        ]
      },
      "unit": "currencyUSD"
    }
  }
}
```

---

## Dashboard 3: Pipeline Performance Dashboard

### Panel 1: Pipeline Stage Duration
```json
{
  "title": "Pipeline Stage Duration",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(leadpoet_pipeline_stage_duration_seconds_bucket[5m]))",
      "legendFormat": "{{stage}}.{{sub_stage}} - P95"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 0.1},
          {"color": "red", "value": 0.2}
        ]
      },
      "unit": "s"
    }
  }
}
```

### Panel 2: Database Operation Duration
```json
{
  "title": "Database Operation Duration",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(leadpoet_db_operation_duration_seconds_bucket[5m]))",
      "legendFormat": "{{operation}} on {{table}} - P95"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 0.05},
          {"color": "red", "value": 0.1}
        ]
      },
      "unit": "s"
    }
  }
}
```

### Panel 3: Cache Hit Rate
```json
{
  "title": "Cache Hit Rate",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(leadpoet_cache_hits_total[5m]) / (rate(leadpoet_cache_hits_total[5m]) + rate(leadpoet_cache_misses_total[5m]))",
      "legendFormat": "{{cache_type}} Hit Rate"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "color": {
        "mode": "palette-classic"
      },
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "barAlignment": 0,
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        },
        "hideFrom": {
          "legend": false,
          "tooltip": false,
          "vis": false
        },
        "thresholdsStyle": {
          "mode": "off"
        }
      },
      "mappings": [],
      "thresholds": {
        "steps": [
          {"color": "red", "value": null},
          {"color": "yellow", "value": 0.7},
          {"color": "green", "value": 0.9}
        ]
      },
      "unit": "percentunit"
    }
  }
}
```

---

## Alert Rules

### High Latency Alert
```yaml
groups:
  - name: leadpoet-api-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])) > 0.4
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.4s)"

      - alert: VeryHighLatency
        expr: histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m])) > 0.55
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Very high API latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 0.55s)"

      - alert: HighCostPerLead
        expr: histogram_quantile(0.95, rate(lead_cost_usd_bucket[5m])) > 0.004
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High cost per lead detected"
          description: "P95 cost per lead is ${{ $value }} (threshold: $0.004)"

      - alert: HighErrorRate
        expr: rate(requests_total{http_status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 5%)"
```

---

## Setup Instructions

1. **Import Dashboards:**
   - Copy the JSON configurations above
   - In Grafana, go to Dashboards → Import
   - Paste the JSON and click "Load"
   - Set the data source to your Prometheus instance
   - Click "Import"

2. **Configure Data Source:**
   - Ensure Prometheus is configured as a data source in Grafana
   - URL: `http://prometheus:9090` (or your Prometheus endpoint)
   - Access: Server (default)

3. **Set Up Alerts:**
   - Copy the alert rules to your Prometheus configuration
   - Restart Prometheus to load the new rules
   - Configure alert manager to send notifications

4. **Environment Variables:**
   ```bash
   # For Datadog APM integration
   export DD_ENV=production
   export DD_SERVICE=leadpoet-intent-model
   export DD_VERSION=1.1.0
   export DD_AGENT_HOST=localhost
   export DD_TRACE_AGENT_PORT=8126
   ```

---

## Dashboard URLs

Once imported, the dashboards will be available at:
- API Performance: `/d/leadpoet-api-performance/leadpoet-api-performance`
- Cost Analysis: `/d/leadpoet-cost-analysis/leadpoet-cost-analysis`  
- Pipeline Performance: `/d/leadpoet-pipeline-performance/leadpoet-pipeline-performance`

These dashboards provide comprehensive monitoring of the Leadpoet Intent Model API performance, costs, and pipeline efficiency. 