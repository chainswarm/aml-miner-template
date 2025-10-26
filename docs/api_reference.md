# API Reference

Complete reference for the AML Miner Template REST API.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Version Info](#version-info)
  - [Score Alerts](#score-alerts)
  - [Rank Alerts](#rank-alerts)
  - [Score Clusters](#score-clusters)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Performance Tips](#performance-tips)

## Overview

The AML Miner Template provides a RESTful API for scoring and ranking Anti-Money Laundering (AML) alerts and transaction clusters.

**API Version:** 1.0.0  
**Protocol:** HTTP/HTTPS  
**Format:** JSON  
**Port:** 8000 (default)

## Base URL

```
http://localhost:8000
```

For production deployments, replace `localhost` with your server domain.

## Authentication

Currently, the API does not require authentication by default. For production use, implement authentication as described in the [Customization Guide](customization.md#adding-authentication).

## Endpoints

### Health Check

Check if the API server is running and healthy.

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

---

### Version Info

Get the current API version.

**Endpoint:** `GET /version`

**Request:**
```bash
curl http://localhost:8000/version
```

**Response:**
```json
{
  "version": "1.0.0",
  "api_version": "v1"
}
```

**Status Codes:**
- `200 OK` - Success

---

### Score Alerts

Score individual AML alerts using the trained alert scorer model.

**Endpoint:** `POST /score/alerts`

**Request Schema:**

```json
{
  "alerts": [
    {
      "alert_id": "string",
      "network": "string",
      "address": "string",
      "amount_usd": "number",
      "transaction_count": "integer",
      "risk_category": "string",
      "timestamp": "string (ISO 8601)"
    }
  ]
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alert_id` | string | Yes | Unique identifier for the alert |
| `network` | string | Yes | Blockchain network (e.g., "bitcoin", "ethereum") |
| `address` | string | Yes | Cryptocurrency address |
| `amount_usd` | number | Yes | Transaction amount in USD |
| `transaction_count` | integer | Yes | Number of transactions |
| `risk_category` | string | Yes | Risk category (e.g., "high_value", "medium_value") |
| `timestamp` | string | Yes | Alert timestamp in ISO 8601 format |

**Example Request:**

```bash
curl -X POST http://localhost:8000/score/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [
      {
        "alert_id": "alert-001",
        "network": "bitcoin",
        "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "amount_usd": 50000.0,
        "transaction_count": 10,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T10:30:00Z"
      },
      {
        "alert_id": "alert-002",
        "network": "ethereum",
        "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "amount_usd": 25000.0,
        "transaction_count": 5,
        "risk_category": "medium_value",
        "timestamp": "2025-01-15T11:00:00Z"
      }
    ]
  }'
```

**Response Schema:**

```json
{
  "scores": [
    {
      "alert_id": "string",
      "score": "number",
      "model_version": "string"
    }
  ]
}
```

**Example Response:**

```json
{
  "scores": [
    {
      "alert_id": "alert-001",
      "score": 0.8523,
      "model_version": "1.0.0"
    },
    {
      "alert_id": "alert-002",
      "score": 0.6234,
      "model_version": "1.0.0"
    }
  ]
}
```

**Score Interpretation:**

- `0.0 - 0.3`: Low risk
- `0.3 - 0.7`: Medium risk
- `0.7 - 1.0`: High risk

**Status Codes:**
- `200 OK` - Scoring successful
- `400 Bad Request` - Invalid request format
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

---

### Rank Alerts

Rank multiple alerts by priority using the alert ranker model.

**Endpoint:** `POST /rank/alerts`

**Request Schema:**

Same as [Score Alerts](#score-alerts) request schema.

**Example Request:**

```bash
curl -X POST http://localhost:8000/rank/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [
      {
        "alert_id": "alert-001",
        "network": "bitcoin",
        "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "amount_usd": 50000.0,
        "transaction_count": 10,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T10:30:00Z"
      },
      {
        "alert_id": "alert-002",
        "network": "ethereum",
        "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "amount_usd": 25000.0,
        "transaction_count": 5,
        "risk_category": "medium_value",
        "timestamp": "2025-01-15T11:00:00Z"
      },
      {
        "alert_id": "alert-003",
        "network": "bitcoin",
        "address": "3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy",
        "amount_usd": 100000.0,
        "transaction_count": 25,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T12:00:00Z"
      }
    ]
  }'
```

**Response Schema:**

```json
{
  "rankings": [
    {
      "alert_id": "string",
      "rank": "integer",
      "score": "number"
    }
  ]
}
```

**Example Response:**

```json
{
  "rankings": [
    {
      "alert_id": "alert-003",
      "rank": 1,
      "score": 0.9234
    },
    {
      "alert_id": "alert-001",
      "rank": 2,
      "score": 0.8523
    },
    {
      "alert_id": "alert-002",
      "rank": 3,
      "score": 0.6234
    }
  ]
}
```

**Ranking Details:**

- Rankings are sorted from highest priority (rank 1) to lowest
- Lower rank number = higher priority
- Scores represent relative priority

**Status Codes:**
- `200 OK` - Ranking successful
- `400 Bad Request` - Invalid request format
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

---

### Score Clusters

Score transaction clusters using the cluster scorer model.

**Endpoint:** `POST /score/clusters`

**Request Schema:**

```json
{
  "clusters": [
    {
      "cluster_id": "string",
      "network": "string",
      "total_volume_usd": "number",
      "transaction_count": "integer",
      "unique_addresses": "integer",
      "pattern_matches": ["string"],
      "risk_level": "string"
    }
  ]
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cluster_id` | string | Yes | Unique identifier for the cluster |
| `network` | string | Yes | Blockchain network |
| `total_volume_usd` | number | Yes | Total transaction volume in USD |
| `transaction_count` | integer | Yes | Number of transactions in cluster |
| `unique_addresses` | integer | Yes | Number of unique addresses |
| `pattern_matches` | array[string] | Yes | Detected patterns (e.g., ["mixing", "layering"]) |
| `risk_level` | string | Yes | Initial risk assessment (e.g., "high", "medium", "low") |

**Example Request:**

```bash
curl -X POST http://localhost:8000/score/clusters \
  -H "Content-Type: application/json" \
  -d '{
    "clusters": [
      {
        "cluster_id": "cluster-001",
        "network": "bitcoin",
        "total_volume_usd": 1000000.0,
        "transaction_count": 150,
        "unique_addresses": 45,
        "pattern_matches": ["mixing", "layering"],
        "risk_level": "high"
      },
      {
        "cluster_id": "cluster-002",
        "network": "ethereum",
        "total_volume_usd": 500000.0,
        "transaction_count": 80,
        "unique_addresses": 25,
        "pattern_matches": [],
        "risk_level": "medium"
      }
    ]
  }'
```

**Response Schema:**

```json
{
  "scores": [
    {
      "cluster_id": "string",
      "score": "number",
      "model_version": "string"
    }
  ]
}
```

**Example Response:**

```json
{
  "scores": [
    {
      "cluster_id": "cluster-001",
      "score": 0.9156,
      "model_version": "1.0.0"
    },
    {
      "cluster_id": "cluster-002",
      "score": 0.5678,
      "model_version": "1.0.0"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Scoring successful
- `400 Bad Request` - Invalid request format
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

---

## Request/Response Formats

### Content Type

All requests must use `Content-Type: application/json`.

### Character Encoding

UTF-8 encoding is required for all requests and responses.

### Timestamps

All timestamps must be in ISO 8601 format:

```
2025-01-15T10:30:00Z        # UTC time
2025-01-15T10:30:00+01:00   # With timezone offset
```

### Number Formats

- Integers: No decimal point (e.g., `100`)
- Floats: With decimal point (e.g., `100.50`)
- Scientific notation accepted (e.g., `1.5e6`)

### Arrays

Empty arrays are valid:

```json
{
  "pattern_matches": []
}
```

### Null Values

Null values are not accepted for required fields. Omit optional fields instead of sending null.

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

### Common Errors

**400 Bad Request**

Invalid JSON or malformed request:

```json
{
  "detail": "Invalid JSON format"
}
```

**422 Unprocessable Entity**

Validation error:

```json
{
  "detail": [
    {
      "loc": ["body", "alerts", 0, "amount_usd"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error**

Server-side error:

```json
{
  "detail": "Internal server error"
}
```

### Error Handling Example

```python
import requests

response = requests.post(
    'http://localhost:8000/score/alerts',
    json={'alerts': [...]},
    timeout=30
)

if response.status_code == 200:
    result = response.json()
    print(f"Scores: {result['scores']}")
elif response.status_code == 422:
    errors = response.json()
    print(f"Validation errors: {errors['detail']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

## Rate Limiting

By default, no rate limiting is enforced. For production deployments, consider implementing rate limiting as described in the [Customization Guide](customization.md#adding-rate-limiting).

**Recommended limits:**
- 100 requests per minute per IP
- 1000 requests per hour per IP

## Performance Tips

### Batch Processing

Score multiple alerts in a single request for better performance:

```python
# Good: Batch processing
response = requests.post(
    'http://localhost:8000/score/alerts',
    json={'alerts': [alert1, alert2, alert3, ...]}
)

# Avoid: Individual requests
for alert in alerts:
    response = requests.post(
        'http://localhost:8000/score/alerts',
        json={'alerts': [alert]}
    )
```

### Connection Pooling

Use connection pooling for multiple requests:

```python
import requests

session = requests.Session()

for batch in alert_batches:
    response = session.post(
        'http://localhost:8000/score/alerts',
        json={'alerts': batch}
    )
```

### Request Timeout

Always set a timeout to prevent hanging:

```python
response = requests.post(
    'http://localhost:8000/score/alerts',
    json={'alerts': alerts},
    timeout=30  # 30 seconds
)
```

### Compression

For large payloads, use gzip compression:

```python
import gzip
import json
import requests

data = {'alerts': large_alert_list}
compressed = gzip.compress(json.dumps(data).encode('utf-8'))

response = requests.post(
    'http://localhost:8000/score/alerts',
    data=compressed,
    headers={
        'Content-Type': 'application/json',
        'Content-Encoding': 'gzip'
    }
)
```

### Response Caching

Implement caching for repeated requests:

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def get_alert_score(alert_hash):
    # Cache results for identical alerts
    pass

alert_hash = hashlib.md5(
    json.dumps(alert, sort_keys=True).encode()
).hexdigest()
```

## Examples

### Python

```python
import requests

# Score alerts
response = requests.post(
    'http://localhost:8000/score/alerts',
    json={
        'alerts': [
            {
                'alert_id': 'alert-001',
                'network': 'bitcoin',
                'address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
                'amount_usd': 50000.0,
                'transaction_count': 10,
                'risk_category': 'high_value',
                'timestamp': '2025-01-15T10:30:00Z'
            }
        ]
    }
)

if response.status_code == 200:
    scores = response.json()['scores']
    for score in scores:
        print(f"Alert {score['alert_id']}: {score['score']:.4f}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function scoreAlerts(alerts) {
  try {
    const response = await axios.post(
      'http://localhost:8000/score/alerts',
      { alerts }
    );
    
    return response.data.scores;
  } catch (error) {
    console.error('Error scoring alerts:', error.response.data);
    throw error;
  }
}

// Usage
const alerts = [
  {
    alert_id: 'alert-001',
    network: 'bitcoin',
    address: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
    amount_usd: 50000.0,
    transaction_count: 10,
    risk_category: 'high_value',
    timestamp: '2025-01-15T10:30:00Z'
  }
];

scoreAlerts(alerts)
  .then(scores => console.log('Scores:', scores))
  .catch(err => console.error('Error:', err));
```

### cURL

```bash
# Score alerts
curl -X POST http://localhost:8000/score/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [
      {
        "alert_id": "alert-001",
        "network": "bitcoin",
        "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "amount_usd": 50000.0,
        "transaction_count": 10,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T10:30:00Z"
      }
    ]
  }'

# Rank alerts
curl -X POST http://localhost:8000/rank/alerts \
  -H "Content-Type: application/json" \
  -d @alerts.json

# Score clusters
curl -X POST http://localhost:8000/score/clusters \
  -H "Content-Type: application/json" \
  -d @clusters.json
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type Alert struct {
    AlertID          string  `json:"alert_id"`
    Network          string  `json:"network"`
    Address          string  `json:"address"`
    AmountUSD        float64 `json:"amount_usd"`
    TransactionCount int     `json:"transaction_count"`
    RiskCategory     string  `json:"risk_category"`
    Timestamp        string  `json:"timestamp"`
}

type ScoreRequest struct {
    Alerts []Alert `json:"alerts"`
}

type ScoreResponse struct {
    Scores []struct {
        AlertID      string  `json:"alert_id"`
        Score        float64 `json:"score"`
        ModelVersion string  `json:"model_version"`
    } `json:"scores"`
}

func scoreAlerts(alerts []Alert) (*ScoreResponse, error) {
    reqBody := ScoreRequest{Alerts: alerts}
    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return nil, err
    }

    resp, err := http.Post(
        "http://localhost:8000/score/alerts",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result ScoreResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return &result, nil
}
```

## Testing

### Health Check

```bash
# Check if service is running
curl http://localhost:8000/health

# Expected: {"status":"healthy","version":"1.0.0"}
```

### Integration Tests

```python
import unittest
import requests

class TestAMLMinerAPI(unittest.TestCase):
    BASE_URL = 'http://localhost:8000'
    
    def test_health(self):
        response = requests.get(f'{self.BASE_URL}/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'healthy')
    
    def test_score_alerts(self):
        alerts = [{
            'alert_id': 'test-001',
            'network': 'bitcoin',
            'address': '1A1z...',
            'amount_usd': 50000.0,
            'transaction_count': 10,
            'risk_category': 'high_value',
            'timestamp': '2025-01-15T10:30:00Z'
        }]
        
        response = requests.post(
            f'{self.BASE_URL}/score/alerts',
            json={'alerts': alerts}
        )
        
        self.assertEqual(response.status_code, 200)
        scores = response.json()['scores']
        self.assertEqual(len(scores), 1)
        self.assertIn('score', scores[0])
        self.assertTrue(0 <= scores[0]['score'] <= 1)

if __name__ == '__main__':
    unittest.main()
```

## Monitoring

### Health Checks

Implement regular health checks:

```bash
# Every 30 seconds
watch -n 30 'curl -s http://localhost:8000/health'
```

### Logging

Enable API logging for debugging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics

Track key metrics:
- Request count
- Response time
- Error rate
- Score distribution

## Support

For issues or questions:

1. Check the [Quick Start Guide](quickstart.md)
2. Review [Troubleshooting](quickstart.md#troubleshooting)
3. See [Training Guide](training_guide.md) for model-related issues
4. Open an issue on GitHub

## Related Documentation

- [Quick Start Guide](quickstart.md) - Getting started
- [Training Guide](training_guide.md) - Training custom models
- [Customization Guide](customization.md) - Extending the API
- [README](../README.md) - Project overview