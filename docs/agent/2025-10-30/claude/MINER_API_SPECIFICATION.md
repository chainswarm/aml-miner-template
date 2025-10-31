# Miner API Specification

**Date**: 2025-10-30  
**Purpose**: Detailed specification for the Miner FastAPI server that exposes scores to the validator  
**Status**: Ready for Implementation

---

## Overview

The Miner API is a FastAPI server that serves pre-computed risk scores from ClickHouse to the validator. It does **not** perform real-time scoring - all scoring is done by batch jobs ([`scripts/score_batch.py`](../../../scripts/score_batch.py)), and the API simply serves the results.

### Key Principles

- **Read-only API** - Serves data, does not modify
- **Pre-computed results** - All scores computed by batch jobs
- **ClickHouse backend** - Reads from existing tables
- **Fast responses** - Simple queries, no complex computation
- **Date-based queries** - All data organized by processing_date

---

## API Endpoints

### Health & Info

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-30T22:00:00Z",
  "version": "1.0.0",
  "database_connected": true
}
```

#### `GET /version`
API and model version information

**Response:**
```json
{
  "api_version": "1.0.0",
  "models": {
    "alert_scorer": "alert_scorer_torus_v1.0.0_20251030_030000",
    "alert_ranker": "alert_ranker_torus_v1.0.0_20251030_030000",
    "cluster_scorer": "cluster_scorer_torus_v1.0.0_20251030_030000"
  },
  "network": "torus",
  "github_url": "https://github.com/your-org/risk-scoring",
  "commit_hash": "a1b2c3d4e5f6"
}
```
---

### Network Information

#### `GET /networks`
Get list of networks supported by this miner

**Response:**
```json
{
  "networks": [
    {
      "network": "torus",
      "status": "active",
      "latest_processing_date": "2025-10-30",
      "model_versions": {
        "alert_scorer": "alert_scorer_torus_v1.0.0_20251030_030000",
        "alert_ranker": "alert_ranker_torus_v1.0.0_20251030_030000",
        "cluster_scorer": "cluster_scorer_torus_v1.0.0_20251030_030000"
      },
      "total_alerts_scored": 125430,
      "last_training_date": "2025-10-27"
    },
    {
      "network": "bitcoin",
      "status": "active",
      "latest_processing_date": "2025-10-30",
      "model_versions": {
        "alert_scorer": "alert_scorer_bitcoin_v1.0.0_20251030_030000",
        "alert_ranker": "alert_ranker_bitcoin_v1.0.0_20251030_030000",
        "cluster_scorer": "cluster_scorer_bitcoin_v1.0.0_20251030_030000"
      },
      "total_alerts_scored": 89234,
      "last_training_date": "2025-10-27"
    }
  ],
  "total_networks": 2,
  "miner_type": "multi-network"
}
```

**Note:** For single-network miners, this will return only one network.

#### `GET /networks/{network}`
Get detailed information about a specific network

**Path Parameters:**
- `network` (string, required): Network identifier (e.g., `torus`, `bitcoin`, `ethereum`)

**Example:** `GET /networks/torus`

**Response:**
```json
{
  "network": "torus",
  "status": "active",
  "latest_processing_date": "2025-10-30",
  "available_dates": ["2025-10-30", "2025-10-29", "2025-10-28"],
  "total_dates": 90,
  "model_versions": {
    "alert_scorer": {
      "version": "alert_scorer_torus_v1.0.0_20251030_030000",
      "trained_at": "2025-10-27T03:45:00Z",
      "performance": {
        "val_auc": 0.887
      }
    },
    "alert_ranker": {
      "version": "alert_ranker_torus_v1.0.0_20251030_030000",
      "trained_at": "2025-10-27T03:52:00Z"
    },
    "cluster_scorer": {
      "version": "cluster_scorer_torus_v1.0.0_20251030_030000",
      "trained_at": "2025-10-27T03:58:00Z"
    }
  },
  "statistics": {
    "total_alerts_scored": 125430,
    "total_clusters_scored": 4523,
    "avg_alerts_per_day": 4181,
    "first_date": "2025-08-01",
    "last_date": "2025-10-30"
  },
  "last_training_date": "2025-10-27",
  "next_training_date": "2025-11-03"
}
```


---

### Model Metadata & Downloads

#### `GET /models/metadata`
Get detailed metadata for all current models

**Response:**
```json
{
  "models": [
    {
      "model_type": "alert_scorer",
      "model_version": "alert_scorer_torus_v1.0.0_20251030_030000",
      "trained_at": "2025-10-30T03:45:00Z",
      "training_duration_minutes": 42,
      "algorithm": "XGBoost",
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic"
      },
      "training_data": {
        "start_date": "2025-10-01",
        "end_date": "2025-10-30",
        "window_days": 195,
        "num_samples": 8543,
        "num_positive": 1247,
        "num_negative": 7296,
        "positive_rate": 0.146,
        "num_features": 56
      },
      "performance": {
        "train_auc": 0.923,
        "val_auc": 0.887,
        "train_logloss": 0.234,
        "val_logloss": 0.289
      },
      "feature_importance_top10": [
        {"feature": "volume_usd_log", "importance": 0.156},
        {"feature": "behavioral_anomaly_score", "importance": 0.124},
        {"feature": "is_mixer_flag", "importance": 0.098}
      ],
      "model_size_mb": 2.4,
      "download_url": "/models/download/alert_scorer"
    }
  ],
  "github_url": "https://github.com/your-org/risk-scoring",
  "commit_hash": "a1b2c3d4e5f6"
}
```

#### `GET /models/metadata/{model_type}`
Get detailed metadata for a specific model type

**Path Parameters:**
- `model_type` (string, required): One of `alert_scorer`, `alert_ranker`, `cluster_scorer`

**Example:** `GET /models/metadata/alert_scorer`

**Response:**
```json
{
  "model_type": "alert_scorer",
  "model_version": "alert_scorer_torus_v1.0.0_20251030_030000",
  "trained_at": "2025-10-30T03:45:00Z",
  "training_duration_minutes": 42,
  "algorithm": "XGBoost",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
  },
  "training_data": {
    "start_date": "2025-10-01",
    "end_date": "2025-10-30",
    "window_days": 195,
    "num_samples": 8543,
    "num_positive": 1247,
    "num_negative": 7296,
    "positive_rate": 0.146,
    "num_features": 56,
    "feature_names": [
      "window_days", "alert_confidence_score", "volume_usd",
      "label_confidence", "severity_encoded", "..."
    ]
  },
  "performance": {
    "train_auc": 0.923,
    "val_auc": 0.887,
    "train_logloss": 0.234,
    "val_logloss": 0.289,
    "train_samples": 6834,
    "val_samples": 1709
  },
  "feature_importance": [
    {"feature": "volume_usd_log", "importance": 0.156, "rank": 1},
    {"feature": "behavioral_anomaly_score", "importance": 0.124, "rank": 2},
    {"feature": "is_mixer_flag", "importance": 0.098, "rank": 3},
    {"feature": "graph_anomaly_score", "importance": 0.087, "rank": 4},
    {"feature": "total_volume", "importance": 0.076, "rank": 5}
  ],
  "model_size_mb": 2.4,
  "model_path": "data/trained_models/torus/alert_scorer_torus_v1.0.0_20251030_030000.txt",
  "download_url": "/models/download/alert_scorer",
  "github_url": "https://github.com/your-org/risk-scoring",
  "commit_hash": "a1b2c3d4e5f6"
}
```

#### `GET /models/download/{model_type}`
Download the trained model file

**Path Parameters:**
- `model_type` (string, required): One of `alert_scorer`, `alert_ranker`, `cluster_scorer`

**Example:** `GET /models/download/alert_scorer`

**Response:**
- Content-Type: `application/octet-stream`
- Content-Disposition: `attachment; filename="alert_scorer_torus_v1.0.0_20251030_030000.txt"`
- Binary file download

**Note:** Returns the XGBoost model file in UBJSON format (.txt extension)

#### `GET /training/data-summary/{processing_date}`
Get summary of training data used for a specific training run

**Path Parameters:**
- `processing_date` (string, required): Training completion date in YYYY-MM-DD format

**Example:** `GET /training/data-summary/2025-10-30`

**Response:**
```json
{
  "training_date": "2025-10-30",
  "data_window": {
    "start_date": "2025-10-01",
    "end_date": "2025-10-30",
    "window_days": 195,
    "num_processing_dates": 30
  },
  "samples": {
    "total": 8543,
    "positive": 1247,
    "negative": 7296,
    "positive_rate": 0.146
  },
  "features": {
    "total_features": 56,
    "feature_groups": {
      "alert_level": 8,
      "address_level": 23,
      "temporal": 5,
      "statistical": 7,
      "cluster": 3,
      "network": 8,
      "label": 2
    }
  },
  "data_sources": {
    "alerts_table": "raw_alerts",
    "features_table": "raw_features",
    "clusters_table": "raw_clusters",
    "money_flows_table": "raw_money_flows",
    "labels_table": "raw_address_labels"
  },
  "label_distribution": {
    "high_risk": 847,
    "critical_risk": 400,
    "medium_risk": 4321,
    "low_risk": 2975
  }
}
```

---

### Date Discovery

#### `GET /dates/available`
List all available processing dates (most recent first)

**Response:**
```json
{
  "dates": [
    "2025-10-30",
    "2025-10-29",
    "2025-10-28",
    "..."
  ],
  "total_count": 90
}
```

#### `GET /dates/latest`
Get the most recent processing date

**Response:**
```json
{
  "latest_date": "2025-10-30",
  "processed_at": "2025-10-30T01:15:00Z"
}
```

---

### Alert Scores

#### `GET /scores/alerts/latest`
Get alert scores for the most recent processing date

**Response:**
```json
{
  "processing_date": "2025-10-30",
  "model_version": "alert_scorer_torus_v1.0.0_20251030_030000",
  "total_alerts": 10000,
  "scores": [
    {
      "alert_id": "alert_001",
      "score": 0.8734,
      "latency_ms": 12.5
    },
    {
      "alert_id": "alert_002",
      "score": 0.2341,
      "latency_ms": 11.8
    }
  ],
  "metadata": {
    "processed_at": "2025-10-30T01:15:00Z",
    "total_latency_ms": 125000,
    "avg_latency_ms": 12.5
  }
}
```

#### `GET /scores/alerts/{processing_date}`
Get alert scores for a specific date

**Path Parameters:**
- `processing_date` (string, required): Date in YYYY-MM-DD format

**Example:** `GET /scores/alerts/2025-10-30`

**Response:** Same as `/scores/alerts/latest`

**Error Responses:**
- `404 Not Found` - No scores for this date
```json
{
  "detail": "No scores found for processing_date: 2025-10-30"
}
```

---

### Alert Rankings

#### `GET /rankings/alerts/latest`
Get alert rankings for the most recent processing date

**Response:**
```json
{
  "processing_date": "2025-10-30",
  "model_version": "alert_ranker_torus_v1.0.0_20251030_030000",
  "total_alerts": 10000,
  "rankings": [
    {
      "alert_id": "alert_042",
      "rank": 1,
      "score": 0.9876
    },
    {
      "alert_id": "alert_123",
      "rank": 2,
      "score": 0.9654
    }
  ],
  "metadata": {
    "processed_at": "2025-10-30T01:20:00Z"
  }
}
```

#### `GET /rankings/alerts/{processing_date}`
Get alert rankings for a specific date

**Path Parameters:**
- `processing_date` (string, required): Date in YYYY-MM-DD format

**Example:** `GET /rankings/alerts/2025-10-30`

**Response:** Same as `/rankings/alerts/latest`

---

### Cluster Scores

#### `GET /scores/clusters/latest`
Get cluster scores for the most recent processing date

**Response:**
```json
{
  "processing_date": "2025-10-30",
  "model_version": "cluster_scorer_torus_v1.0.0_20251030_030000",
  "total_clusters": 500,
  "scores": [
    {
      "cluster_id": "cluster_001",
      "score": 0.7234,
      "latency_ms": 8.3
    },
    {
      "cluster_id": "cluster_002",
      "score": 0.4521,
      "latency_ms": 7.9
    }
  ],
  "metadata": {
    "processed_at": "2025-10-30T01:25:00Z",
    "total_latency_ms": 42000,
    "avg_latency_ms": 8.4
  }
}
```

#### `GET /scores/clusters/{processing_date}`
Get cluster scores for a specific date

**Path Parameters:**
- `processing_date` (string, required): Date in YYYY-MM-DD format

**Example:** `GET /scores/clusters/2025-10-30`

**Response:** Same as `/scores/clusters/latest`

---

## Data Models

### Pydantic Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_connected: bool

class VersionResponse(BaseModel):
    api_version: str
    models: Dict[str, str]
    network: str

class DatesAvailableResponse(BaseModel):
    dates: List[str]
    total_count: int

class LatestDateResponse(BaseModel):
    latest_date: str
    processed_at: datetime

class AlertScore(BaseModel):
    alert_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float

class AlertScoresMetadata(BaseModel):
    processed_at: datetime
    total_latency_ms: int
    avg_latency_ms: float

class AlertScoresResponse(BaseModel):
    processing_date: str
    model_version: str
    total_alerts: int
    scores: List[AlertScore]
    metadata: AlertScoresMetadata

class AlertRanking(BaseModel):
    alert_id: str
    rank: int
    score: float = Field(..., ge=0.0, le=1.0)

class AlertRankingsMetadata(BaseModel):
    processed_at: datetime

class AlertRankingsResponse(BaseModel):
    processing_date: str
    model_version: str
    total_alerts: int
    rankings: List[AlertRanking]
    metadata: AlertRankingsMetadata

class ClusterScore(BaseModel):
    cluster_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float

class ClusterScoresMetadata(BaseModel):
    processed_at: datetime
    total_latency_ms: int
    avg_latency_ms: float

class ClusterScoresResponse(BaseModel):
    processing_date: str
    model_version: str
    total_clusters: int
    scores: List[ClusterScore]
    metadata: ClusterScoresMetadata
```

---

## Database Queries

### ClickHouse Table Schema

The API reads from these ClickHouse tables:

**alert_scores:**
```sql
CREATE TABLE alert_scores (
    processing_date Date,
    alert_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (processing_date, alert_id);
```

**alert_rankings:**
```sql
CREATE TABLE alert_rankings (
    processing_date Date,
    alert_id String,
    rank UInt32,
    score Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (processing_date, rank);
```

**cluster_scores:**
```sql
CREATE TABLE cluster_scores (
    processing_date Date,
    cluster_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (processing_date, cluster_id);
```

**batch_metadata:**
```sql
CREATE TABLE batch_metadata (
    processing_date Date,
    status String,
    input_counts_alerts UInt32,
    output_counts_alert_scores UInt32,
    output_counts_alert_rankings UInt32,
    output_counts_cluster_scores UInt32,
    latencies_ms_total UInt32,
    model_versions_alert_scorer String,
    model_versions_alert_ranker String,
    model_versions_cluster_scorer String,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY processing_date;
```

### Example SQL Queries

**Get latest processing date:**
```sql
SELECT processing_date
FROM batch_metadata
WHERE status = 'COMPLETED'
ORDER BY processing_date DESC
LIMIT 1
```

**Get alert scores for a date:**
```sql
SELECT 
    alert_id,
    score,
    latency_ms,
    model_version
FROM alert_scores
WHERE processing_date = '2025-10-30'
ORDER BY alert_id
```

**Get alert rankings for a date:**
```sql
SELECT 
    alert_id,
    rank,
    score,
    model_version
FROM alert_rankings
WHERE processing_date = '2025-10-30'
ORDER BY rank
LIMIT 10000
```

**Get cluster scores for a date:**
```sql
SELECT 
    cluster_id,
    score,
    latency_ms,
    model_version
FROM cluster_scores
WHERE processing_date = '2025-10-30'
ORDER BY cluster_id
```

**Get available dates:**
```sql
SELECT DISTINCT processing_date
FROM batch_metadata
WHERE status = 'COMPLETED'
ORDER BY processing_date DESC
```

**Get metadata for a date:**
```sql
SELECT *
FROM batch_metadata
WHERE processing_date = '2025-10-30'
```

---

## Configuration

### Environment Variables

```bash
# ClickHouse Connection
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=default

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=1.0.0

# Network
NETWORK=torus

# CORS (if needed)
CORS_ORIGINS=["*"]
```

### Config File Example

```python
# packages/api/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "default"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "1.0.0"
    
    # Network
    network: str = "torus"
    
    # CORS
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Implementation Structure

```
packages/api/
├── __init__.py
├── config.py           # Configuration settings
├── server.py           # FastAPI application entry point
├── routes.py           # API endpoint definitions
├── models.py           # Pydantic response models
├── database.py         # ClickHouse connection and queries
└── dependencies.py     # FastAPI dependencies
```

### Key Files

**`server.py`** - FastAPI application
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from packages.api.routes import router
from packages.api.config import settings

app = FastAPI(
    title="Miner API",
    version=settings.api_version,
    description="Risk scoring miner API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "packages.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
```

**`routes.py`** - API endpoints
```python
from fastapi import APIRouter, Depends, HTTPException
from packages.api.database import DatabaseClient
from packages.api.models import *
from packages.api.dependencies import get_db

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(db: DatabaseClient = Depends(get_db)):
    # Implementation
    pass

@router.get("/scores/alerts/{processing_date}", response_model=AlertScoresResponse)
async def get_alert_scores(
    processing_date: str,
    db: DatabaseClient = Depends(get_db)
):
    # Implementation
    pass
```

**`database.py`** - ClickHouse operations
```python
from clickhouse_connect import get_client
from packages.api.config import settings

class DatabaseClient:
    def __init__(self):
        self.client = get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database
        )
    
    def get_latest_date(self) -> str:
        # Query implementation
        pass
    
    def get_alert_scores(self, processing_date: str):
        # Query implementation
        pass
```

---

## Error Handling

### Standard Error Responses

**404 Not Found:**
```json
{
  "detail": "No scores found for processing_date: 2025-10-30"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Database connection failed"
}
```

**422 Unprocessable Entity:**
```json
{
  "detail": [
    {
      "loc": ["path", "processing_date"],
      "msg": "Invalid date format. Use YYYY-MM-DD",
      "type": "value_error"
    }
  ]
}
```

---

## Testing

### Unit Tests

```python
# tests/api/test_routes.py
from fastapi.testclient import TestClient
from packages.api.server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_alert_scores():
    response = client.get("/scores/alerts/2025-10-30")
    assert response.status_code == 200
    data = response.json()
    assert data["processing_date"] == "2025-10-30"
    assert len(data["scores"]) > 0
```

### Integration Tests

```bash
# Start API server
python -m packages.api.server

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/dates/latest
curl http://localhost:8000/scores/alerts/latest
curl http://localhost:8000/rankings/alerts/2025-10-30
curl http://localhost:8000/scores/clusters/2025-10-30
```

---

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/ packages/
COPY .env .env

EXPOSE 8000

CMD ["python", "-m", "packages.api.server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  miner-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CLICKHOUSE_HOST=clickhouse
      - CLICKHOUSE_PORT=9000
    depends_on:
      - clickhouse
    restart: unless-stopped
```

### Systemd Service

```ini
# /etc/systemd/system/miner-api.service
[Unit]
Description=Miner API Service
After=network.target clickhouse-server.service

[Service]
Type=simple
User=miner
WorkingDirectory=/opt/risk-scoring
ExecStart=/usr/bin/python3 -m packages.api.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Performance Considerations

### Response Times
- **Target:** < 100ms for most queries
- **Alert scores:** ~10,000 rows, expect 50-100ms
- **Rankings:** Pre-sorted, expect 30-50ms
- **Cluster scores:** ~500 rows, expect 20-30ms

### Optimization Strategies
1. **Connection pooling** - Reuse ClickHouse connections
2. **Query optimization** - Use appropriate ORDER BY and LIMIT
3. **Caching** - Consider Redis for latest_date queries
4. **Pagination** - Add limit/offset parameters for large result sets
5. **Compression** - Enable response compression

### Monitoring
- Track response times per endpoint
- Monitor ClickHouse query performance
- Alert on error rates > 1%
- Track database connection health

---

## Security

### Authentication (Future)
```python
from fastapi import Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.get("/scores/alerts/{processing_date}")
async def get_alert_scores(
    processing_date: str,
    credentials: str = Security(security)
):
    # Validate token
    pass
```

### Rate Limiting (Future)
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/scores/alerts/{processing_date}")
@limiter.limit("100/minute")
async def get_alert_scores(processing_date: str):
    pass
```

---

## Next Steps

1. **Implementation** - Build the API using this specification
2. **Testing** - Unit and integration tests
3. **Deployment** - Docker container and systemd service
4. **Monitoring** - Add logging and metrics
5. **Documentation** - OpenAPI/Swagger docs (auto-generated by FastAPI)

---

**End of Specification**

---

## Multi-Network Support

### Deployment Strategy Options

Miners can choose between two deployment strategies:

#### Option 1: Single-Network Miner (Recommended for Most)

**Architecture:**
```
One Miner Instance = One Network
├─ Miner 1 → torus only
├─ Miner 2 → bitcoin only
└─ Miner 3 → ethereum only
```

**Advantages:**
- ✅ **Simpler** - One ClickHouse database per network
- ✅ **Isolated** - Issues in one network don't affect others
- ✅ **Scalable** - Add capacity per network independently
- ✅ **Clear separation** - Each miner focuses on one network

**Configuration:**
```bash
# .env
NETWORK=torus
CLICKHOUSE_DATABASE=torus_db
```

**When to Use:**
- Most miners (specialized per network)
- Limited resources per miner
- Want simple deployment
- Network-specific ML optimizations

---

#### Option 2: Multi-Network Miner (Advanced)

**Architecture:**
```
One Miner Instance = Multiple Networks
└─ Miner 1 → torus + bitcoin + ethereum
   ├─ Separate ClickHouse databases
   ├─ Separate models per network
   └─ Unified API serving all networks
```

**Advantages:**
- ✅ **Resource efficient** - One API server for all networks
- ✅ **Centralized** - Single deployment to manage
- ✅ **Cost effective** - Share infrastructure costs

**Challenges:**
- ⚠️ **More complex** - Must manage multiple databases
- ⚠️ **Higher load** - API serves more requests
- ⚠️ **Harder debugging** - Issues span networks

**Configuration:**
```bash
# .env
NETWORKS=torus,bitcoin,ethereum
CLICKHOUSE_HOST=localhost
# Databases created per network: torus_db, bitcoin_db, ethereum_db
```

**When to Use:**
- Advanced miners with resources
- Want to maximize subnet participation
- Can handle operational complexity
- Have expertise across networks

---

### API Behavior by Deployment Type

#### Single-Network Miner

**`GET /networks`** returns:
```json
{
  "networks": [
    {
      "network": "torus",
      "status": "active"
    }
  ],
  "total_networks": 1,
  "miner_type": "single-network"
}
```

All endpoints implicitly use the configured network.

---

#### Multi-Network Miner

**`GET /networks`** returns:
```json
{
  "networks": [
    {"network": "torus", "status": "active"},
    {"network": "bitcoin", "status": "active"},
    {"network": "ethereum", "status": "active"}
  ],
  "total_networks": 3,
  "miner_type": "multi-network"
}
```

**Network-specific endpoints:**

Add network parameter to all scoring endpoints:

```
# Option A: Path parameter (cleaner)
GET /scores/alerts/{network}/{processing_date}
GET /rankings/alerts/{network}/{processing_date}
GET /scores/clusters/{network}/{processing_date}

Example: GET /scores/alerts/torus/2025-10-30

# Option B: Query parameter (compatible)
GET /scores/alerts/{processing_date}?network=torus
GET /rankings/alerts/{processing_date}?network=bitcoin
```

**Recommended:** Use Option A (path parameter) for multi-network miners.

---

### Database Organization

#### Single-Network

```
ClickHouse
└─ torus_db (or bitcoin_db, etc.)
   ├─ raw_alerts
   ├─ raw_features
   ├─ alert_scores
   ├─ alert_rankings
   └─ cluster_scores
```

#### Multi-Network

```
ClickHouse
├─ torus_db
│  ├─ raw_alerts
│  ├─ alert_scores
│  └─ ...
├─ bitcoin_db
│  ├─ raw_alerts
│  ├─ alert_scores
│  └─ ...
└─ ethereum_db
   ├─ raw_alerts
   ├─ alert_scores
   └─ ...
```

**Query Logic:**
```python
def get_alert_scores(network: str, processing_date: str):
    database = f"{network}_db"
    query = f"SELECT * FROM {database}.alert_scores WHERE processing_date = ?"
    # ...
```

---

### Model Storage

#### Single-Network

```
data/trained_models/
└─ torus/
   ├─ alert_scorer_torus_v1.0.0_20251030_030000.txt
   ├─ alert_ranker_torus_v1.0.0_20251030_030000.txt
   └─ cluster_scorer_torus_v1.0.0_20251030_030000.txt
```

#### Multi-Network

```
data/trained_models/
├─ torus/
│  ├─ alert_scorer_torus_v1.0.0_20251030_030000.txt
│  └─ ...
├─ bitcoin/
│  ├─ alert_scorer_bitcoin_v1.0.0_20251030_030000.txt
│  └─ ...
└─ ethereum/
   ├─ alert_scorer_ethereum_v1.0.0_20251030_030000.txt
   └─ ...
```

---

### Cron Schedule

#### Single-Network Miner

```bash
# Daily ingestion
30 0 * * * python scripts/ingest_data.py --network torus --processing-date $(date +%Y-%m-%d)

# Daily scoring
0 1 * * * python scripts/score_batch.py --network torus --processing-date $(date +%Y-%m-%d)

# Weekly training
0 3 * * 0 python scripts/train_model.py --network torus --start-date $(date -d '30 days ago' +%Y-%m-%d)
```

#### Multi-Network Miner

```bash
# Daily ingestion (loop through networks)
30 0 * * * for network in torus bitcoin ethereum; do \
    python scripts/ingest_data.py --network $network --processing-date $(date +%Y-%m-%d); \
done

# Daily scoring (parallel or sequential)
0 1 * * * for network in torus bitcoin ethereum; do \
    python scripts/score_batch.py --network $network --processing-date $(date +%Y-%m-%d) & \
done; wait

# Weekly training (sequential, resource-intensive)
0 3 * * 0 for network in torus bitcoin ethereum; do \
    python scripts/train_model.py --network $network --start-date $(date -d '30 days ago' +%Y-%m-%d); \
done
```

---

### Recommendation Matrix

| Scenario | Recommended Strategy |
|----------|---------------------|
| First-time miner | Single-Network |
| Limited resources (<16GB RAM) | Single-Network |
| Specializing in one blockchain | Single-Network |
| Testing/development | Single-Network |
| Advanced miner with resources | Multi-Network |
| Want maximum subnet rewards | Multi-Network |
| Operational expertise | Multi-Network |
| >32GB RAM, multi-core CPU | Multi-Network |

---

### Migration Path

**Start Single → Scale to Multi**

1. Deploy single-network miner (e.g., torus)
2. Validate stability and performance
3. Add second network in separate miner instance
4. Once confident, consolidate to multi-network deployment

**Validator Perspective:**

Validators don't care if miner is single or multi-network - they just query:
```
GET miner:8000/scores/alerts/torus/2025-10-30
GET miner:8000/scores/alerts/bitcoin/2025-10-30
```

The miner's internal architecture is transparent to validators.
