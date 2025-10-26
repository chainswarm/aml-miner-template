# API Reference

Complete reference for the Alert Scoring Template REST API.

## Base URL

```
http://localhost:8000
```

## Overview

The API provides read-only access to pre-computed alert scores, rankings, and cluster scores stored in ClickHouse. All endpoints use GET requests (except `/refresh` which is a no-op POST).

**Key Concepts:**
- **Processing Date**: The date when a batch was processed (YYYY-MM-DD format)
- **Network**: Blockchain network identifier (ethereum, bitcoin, polygon, etc.)
- **Batch Processing**: Data is processed offline before being served via API

## Authentication

Currently, the API does not require authentication. For production deployments, add authentication middleware as needed.

## Common Query Parameters

Most endpoints accept these query parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | string | `ethereum` | Blockchain network to query |

## Response Format

All successful responses return JSON with appropriate HTTP status codes:

- `200 OK` - Successful request
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error responses include a detail message:

```json
{
  "detail": "Error description here"
}
```

---

## Health & Version Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "storage": "clickhouse",
  "default_network": "ethereum"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/health
```

**Python Example:**
```python
import requests

response = requests.get("http://localhost:8000/health")
data = response.json()
print(f"Status: {data['status']}")
```

---

### GET /version

Get API and storage backend version information.

**Response:**
```json
{
  "api_version": "1.0.0",
  "storage_backend": "clickhouse"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/version
```

---

## Date Endpoints

### GET /dates/available

Get all available processing dates for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  "2024-01-15",
  "2024-01-14",
  "2024-01-13"
]
```

**cURL Example:**
```bash
curl "http://localhost:8000/dates/available?network=ethereum"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/dates/available",
    params={"network": "ethereum"}
)
dates = response.json()
print(f"Available dates: {dates}")
```

---

### GET /dates/latest

Get the latest processing date for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
{
  "processing_date": "2024-01-15",
  "network": "ethereum"
}
```

**Error Responses:**
- `404` - No processing dates available for network

**cURL Example:**
```bash
curl "http://localhost:8000/dates/latest?network=ethereum"
```

---

## Alert Score Endpoints

### GET /scores/alerts/latest

Get the latest alert scores for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
{
  "processing_date": "2024-01-15",
  "network": "ethereum",
  "total_scores": 15234,
  "scores": [
    {
      "alert_id": "alert_001",
      "score": 0.9234,
      "model_version": "1.0.0",
      "latency_ms": 12.5,
      "explain_json": "{\"top_features\": [\"volume_usd\", \"pagerank\"]}"
    },
    {
      "alert_id": "alert_002",
      "score": 0.8756,
      "model_version": "1.0.0",
      "latency_ms": 11.8,
      "explain_json": null
    }
  ]
}
```

**Error Responses:**
- `404` - No scores available for network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/alerts/latest?network=ethereum"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/scores/alerts/latest",
    params={"network": "ethereum"}
)
data = response.json()
print(f"Processing date: {data['processing_date']}")
print(f"Total scores: {data['total_scores']}")

# Get top 10 highest scores
top_scores = sorted(data['scores'], key=lambda x: x['score'], reverse=True)[:10]
for score in top_scores:
    print(f"Alert {score['alert_id']}: {score['score']:.4f}")
```

---

### GET /scores/alerts/{processing_date}

Get alert scores for a specific processing date.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `processing_date` | string | Yes | Processing date (YYYY-MM-DD) |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "alert_id": "alert_001",
    "score": 0.9234,
    "model_version": "1.0.0",
    "latency_ms": 12.5,
    "explain_json": "{\"top_features\": [\"volume_usd\"]}"
  }
]
```

**Error Responses:**
- `404` - No scores found for date and network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/alerts/2024-01-15?network=ethereum"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/scores/alerts/2024-01-15",
    params={"network": "ethereum"}
)
scores = response.json()

# Filter high-risk alerts (score > 0.8)
high_risk = [s for s in scores if s['score'] > 0.8]
print(f"Found {len(high_risk)} high-risk alerts")
```

---

## Ranking Endpoints

### GET /scores/rankings/latest

Get the latest alert rankings for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "alert_id": "alert_001",
    "rank": 1,
    "model_version": "1.0.0"
  },
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "alert_id": "alert_002",
    "rank": 2,
    "model_version": "1.0.0"
  }
]
```

**Error Responses:**
- `404` - No rankings available for network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/rankings/latest?network=ethereum"
```

---

### GET /scores/rankings/{processing_date}

Get alert rankings for a specific processing date.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `processing_date` | string | Yes | Processing date (YYYY-MM-DD) |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "alert_id": "alert_001",
    "rank": 1,
    "model_version": "1.0.0"
  }
]
```

**Error Responses:**
- `404` - No rankings found for date and network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/rankings/2024-01-15?network=ethereum"
```

---

### GET /scores/rankings/top/{n}

Get top N ranked alerts.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `n` | integer | Yes | Number of top alerts to return |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `processing_date` | string | No | Latest date | Processing date (YYYY-MM-DD) |
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "alert_id": "alert_001",
    "rank": 1,
    "model_version": "1.0.0"
  },
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "alert_id": "alert_002",
    "rank": 2,
    "model_version": "1.0.0"
  }
]
```

**Error Responses:**
- `404` - No rankings available or found

**cURL Examples:**
```bash
# Get top 100 from latest date
curl "http://localhost:8000/scores/rankings/top/100?network=ethereum"

# Get top 50 from specific date
curl "http://localhost:8000/scores/rankings/top/50?processing_date=2024-01-15&network=ethereum"
```

**Python Example:**
```python
import requests

# Get top 10 alerts
response = requests.get(
    "http://localhost:8000/scores/rankings/top/10",
    params={"network": "ethereum"}
)
rankings = response.json()

for ranking in rankings:
    print(f"Rank {ranking['rank']}: {ranking['alert_id']}")
```

---

## Cluster Score Endpoints

### GET /scores/clusters/latest

Get the latest cluster scores for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "cluster_id": "cluster_001",
    "score": 0.8845,
    "model_version": "1.0.0"
  },
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "cluster_id": "cluster_002",
    "score": 0.7623,
    "model_version": "1.0.0"
  }
]
```

**Error Responses:**
- `404` - No cluster scores available for network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/clusters/latest?network=ethereum"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/scores/clusters/latest",
    params={"network": "ethereum"}
)
clusters = response.json()

# Find high-risk clusters
high_risk_clusters = [c for c in clusters if c['score'] > 0.8]
print(f"Found {len(high_risk_clusters)} high-risk clusters")
```

---

### GET /scores/clusters/{processing_date}

Get cluster scores for a specific processing date.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `processing_date` | string | Yes | Processing date (YYYY-MM-DD) |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "cluster_id": "cluster_001",
    "score": 0.8845,
    "model_version": "1.0.0"
  }
]
```

**Error Responses:**
- `404` - No cluster scores found for date and network

**cURL Example:**
```bash
curl "http://localhost:8000/scores/clusters/2024-01-15?network=ethereum"
```

---

### GET /scores/clusters/{cluster_id}/{processing_date}

Get score for a specific cluster.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cluster_id` | string | Yes | Cluster identifier |
| `processing_date` | string | Yes | Processing date (YYYY-MM-DD) |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
{
  "processing_date": "2024-01-15",
  "network": "ethereum",
  "cluster_id": "cluster_001",
  "score": 0.8845,
  "model_version": "1.0.0"
}
```

**Error Responses:**
- `404` - Cluster score not found

**cURL Example:**
```bash
curl "http://localhost:8000/scores/clusters/cluster_001/2024-01-15?network=ethereum"
```

---

## Metadata Endpoints

### GET /metadata/latest

Get the latest batch processing metadata for a network.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
```json
{
  "processing_date": "2024-01-15",
  "network": "ethereum",
  "processed_at": "2024-01-15T14:30:00Z",
  "input_counts_alerts": 15234,
  "input_counts_features": 15234,
  "input_counts_clusters": 3456,
  "output_counts_alert_scores": 15234,
  "output_counts_alert_rankings": 15234,
  "output_counts_cluster_scores": 3456,
  "latencies_ms_alert_scoring": 2300,
  "latencies_ms_alert_ranking": 1800,
  "latencies_ms_cluster_scoring": 900,
  "latencies_ms_total": 5200,
  "model_versions_alert_scorer": "1.0.0",
  "model_versions_alert_ranker": "1.0.0",
  "model_versions_cluster_scorer": "1.0.0",
  "status": "completed",
  "error_message": null
}
```

**Error Responses:**
- `404` - No batch metadata available for network

**cURL Example:**
```bash
curl "http://localhost:8000/metadata/latest?network=ethereum"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/metadata/latest",
    params={"network": "ethereum"}
)
metadata = response.json()

print(f"Processing date: {metadata['processing_date']}")
print(f"Total processing time: {metadata['latencies_ms_total']}ms")
print(f"Alerts processed: {metadata['input_counts_alerts']}")
print(f"Status: {metadata['status']}")
```

---

### GET /metadata/{processing_date}

Get batch processing metadata for a specific date.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `processing_date` | string | Yes | Processing date (YYYY-MM-DD) |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |

**Response:**
Same as `/metadata/latest`

**Error Responses:**
- `404` - No metadata found for date and network

**cURL Example:**
```bash
curl "http://localhost:8000/metadata/2024-01-15?network=ethereum"
```

---

### GET /metadata/history

Get historical batch processing metadata.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network to query |
| `limit` | integer | No | 30 | Number of records to return (1-365) |

**Response:**
```json
[
  {
    "processing_date": "2024-01-15",
    "network": "ethereum",
    "processed_at": "2024-01-15T14:30:00Z",
    "input_counts_alerts": 15234,
    "output_counts_alert_scores": 15234,
    "latencies_ms_total": 5200,
    "status": "completed"
  },
  {
    "processing_date": "2024-01-14",
    "network": "ethereum",
    "processed_at": "2024-01-14T14:30:00Z",
    "input_counts_alerts": 14523,
    "output_counts_alert_scores": 14523,
    "latencies_ms_total": 4800,
    "status": "completed"
  }
]
```

**Error Responses:**
- `404` - No metadata history available for network

**cURL Example:**
```bash
# Get last 30 batches (default)
curl "http://localhost:8000/metadata/history?network=ethereum"

# Get last 7 batches
curl "http://localhost:8000/metadata/history?network=ethereum&limit=7"
```

**Python Example:**
```python
import requests
import pandas as pd

response = requests.get(
    "http://localhost:8000/metadata/history",
    params={"network": "ethereum", "limit": 30}
)
history = response.json()

# Convert to DataFrame for analysis
df = pd.DataFrame(history)
print(f"Average processing time: {df['latencies_ms_total'].mean():.0f}ms")
print(f"Average alerts per batch: {df['input_counts_alerts'].mean():.0f}")
```

---

## Utility Endpoints

### POST /refresh

No-op endpoint for compatibility. ClickHouse data is immediately available without refresh.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `network` | string | No | `ethereum` | Network (ignored) |

**Response:**
```json
{
  "status": "success",
  "message": "ClickHouse doesn't require refresh - data is immediately available",
  "network": "ethereum"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/refresh?network=ethereum"
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Request completed successfully |
| 404 | Not Found | Resource doesn't exist (no data for date/network) |
| 500 | Server Error | Database connection issue, query error |

### Example Error Handling

**Python:**
```python
import requests

try:
    response = requests.get(
        "http://localhost:8000/scores/alerts/2024-01-15",
        params={"network": "ethereum"}
    )
    response.raise_for_status()
    scores = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print(f"No data found: {e.response.json()['detail']}")
    else:
        print(f"Server error: {e.response.json()['detail']}")
except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")
```

---

## Python Client Examples

### Complete Client Class

```python
import requests
from typing import List, Dict, Optional
from datetime import date

class AlertScoringClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def get_latest_scores(self, network: str = "ethereum") -> Dict:
        response = requests.get(
            f"{self.base_url}/scores/alerts/latest",
            params={"network": network}
        )
        response.raise_for_status()
        return response.json()
    
    def get_scores_by_date(
        self, 
        processing_date: str, 
        network: str = "ethereum"
    ) -> List[Dict]:
        response = requests.get(
            f"{self.base_url}/scores/alerts/{processing_date}",
            params={"network": network}
        )
        response.raise_for_status()
        return response.json()
    
    def get_top_rankings(
        self, 
        n: int, 
        network: str = "ethereum",
        processing_date: Optional[str] = None
    ) -> List[Dict]:
        params = {"network": network}
        if processing_date:
            params["processing_date"] = processing_date
        
        response = requests.get(
            f"{self.base_url}/scores/rankings/top/{n}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_metadata(
        self, 
        network: str = "ethereum",
        processing_date: Optional[str] = None
    ) -> Dict:
        if processing_date:
            url = f"{self.base_url}/metadata/{processing_date}"
        else:
            url = f"{self.base_url}/metadata/latest"
        
        response = requests.get(url, params={"network": network})
        response.raise_for_status()
        return response.json()
    
    def get_available_dates(self, network: str = "ethereum") -> List[str]:
        response = requests.get(
            f"{self.base_url}/dates/available",
            params={"network": network}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = AlertScoringClient()

# Get latest scores
latest = client.get_latest_scores("ethereum")
print(f"Latest batch: {latest['processing_date']}")

# Get top 100 alerts
top_100 = client.get_top_rankings(100, "ethereum")
for ranking in top_100[:10]:
    print(f"Rank {ranking['rank']}: {ranking['alert_id']}")

# Get metadata
metadata = client.get_metadata("ethereum")
print(f"Processing time: {metadata['latencies_ms_total']}ms")
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider adding rate limiting middleware.

## CORS

CORS is configured to allow all origins. Update [`aml_miner/api/server.py`](../aml_miner/api/server.py:1) for production use:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

## API Versioning

The API is currently at version 1.0.0. Future versions will use URL prefixing (`/v2/scores/alerts/latest`).

## See Also

- **[Quick Start Guide](quickstart.md)** - Setup and first run
- **[README](../README.md)** - Project overview
- **[Architecture Docs](../docs/agent/2025-10-26/claude/)** - System architecture