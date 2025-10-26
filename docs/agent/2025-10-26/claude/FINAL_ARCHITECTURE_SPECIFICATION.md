# Final Architecture Specification
## AML Risk Scoring System - ClickHouse Production Setup

**Date**: 2025-10-26  
**Purpose**: Final production-ready architecture specification

---

## Repository Naming

### Recommendations:

1. **`aml-alert-scoring`** ⭐ (Recommended)
   - Clear: It's about scoring AML alerts
   - Specific: Domain + function explicit
   - Professional: Standard naming convention

2. **`aml-risk-scoring`**
   - Broader scope (could include entity risk, not just alerts)
   - Good if expanding beyond alerts later

3. **`alert-risk-scoring`**
   - Less specific about AML domain
   - Could be confused with general security alerts

**Recommendation**: Use **`aml-alert-scoring`** - it's clear, specific, and professional.

---

## Directory Structure

```
aml-alert-scoring/                 # Root repository
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .env.example
│
├── aml_scoring/                   # Main package (renamed from aml_miner)
│   ├── __init__.py
│   │
│   ├── storage/                   # ClickHouse storage layer
│   │   ├── __init__.py           # ClientFactory, migrations
│   │   ├── utils.py              # Pydantic/enum utilities
│   │   ├── schema/               # SQL DDL files
│   │   │   ├── raw_alerts.sql
│   │   │   ├── raw_features.sql
│   │   │   ├── raw_clusters.sql
│   │   │   ├── alert_scores.sql
│   │   │   ├── alert_rankings.sql
│   │   │   ├── cluster_scores.sql
│   │   │   └── batch_metadata.sql
│   │   │
│   │   └── repositories/         # Data access layer
│   │       ├── __init__.py
│   │       ├── base_repository.py
│   │       ├── alerts_repository.py
│   │       ├── features_repository.py
│   │       ├── clusters_repository.py
│   │       ├── scores_repository.py
│   │       ├── rankings_repository.py
│   │       └── metadata_repository.py
│   │
│   ├── models/                   # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── alert_scorer.py
│   │   ├── alert_ranker.py
│   │   └── cluster_scorer.py
│   │
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_builder.py
│   │   └── feature_selector.py
│   │
│   ├── training/                 # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_scorer.py
│   │   ├── train_ranker.py
│   │   └── hyperparameter_tuner.py
│   │
│   ├── validation/               # Validation framework
│   │   ├── __init__.py
│   │   ├── integrity_validator.py
│   │   ├── behavior_validator.py
│   │   ├── ground_truth_validator.py
│   │   └── validator_utils.py
│   │
│   ├── api/                      # FastAPI server
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── config/                   # Configuration
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   └── utils/                    # General utilities
│       ├── __init__.py
│       └── determinism.py
│
├── scripts/                      # Operational scripts
│   ├── download_from_sot.py     # Download from SOT
│   ├── process_batch.py         # Batch scoring
│   ├── train_models.py          # Model training
│   ├── validate_models.py       # A/B testing
│   └── init_database.py         # Database initialization
│
├── trained_models/               # Model artifacts
│   ├── alert_scorer_v1.0.0.txt
│   ├── alert_ranker_v1.0.0.txt
│   └── metadata.json
│
└── docs/                         # Documentation
    ├── quickstart.md
    ├── training_guide.md
    ├── api_reference.md
    └── deployment.md
```

---

## ClickHouse Schema (Simplified Table Names)

### `storage/schema/raw_alerts.sql`

```sql
CREATE TABLE IF NOT EXISTS raw_alerts (
    window_days Int32,
    processing_date Date,
    network String,
    alert_id String,
    address String,
    typology_type String,
    pattern_id String DEFAULT '',
    pattern_type String DEFAULT '',
    severity Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    suspected_address_type String DEFAULT 'unknown',
    suspected_address_subtype String DEFAULT '',
    alert_confidence_score Float64,
    description String,
    volume_usd Decimal(18, 2) DEFAULT 0,
    evidence_json String,
    risk_indicators Array(String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_alert_id ON raw_alerts(alert_id) TYPE bloom_filter GRANULARITY 1;
```

### `storage/schema/raw_features.sql`

```sql
CREATE TABLE IF NOT EXISTS raw_features (
    processing_date Date,
    network String,
    address String,
    feature_name String,
    feature_value Float64,
    feature_metadata String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, address, feature_name)
SETTINGS index_granularity = 8192;
```

### `storage/schema/raw_clusters.sql`

```sql
CREATE TABLE IF NOT EXISTS raw_clusters (
    window_days Int32,
    processing_date Date,
    network String,
    cluster_id String,
    cluster_type String,
    primary_address String DEFAULT '',
    pattern_id String DEFAULT '',
    primary_alert_id String,
    related_alert_ids Array(String),
    addresses_involved Array(String),
    total_alerts Int32,
    total_volume_usd Decimal(18, 2),
    severity_max Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    confidence_avg Float64,
    earliest_alert_timestamp Int64,
    latest_alert_timestamp Int64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;
```

### `storage/schema/alert_scores.sql`

```sql
CREATE TABLE IF NOT EXISTS alert_scores (
    processing_date Date,
    network String,
    alert_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    explain_json String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_score ON alert_scores(score) TYPE minmax GRANULARITY 4;
```

### `storage/schema/alert_rankings.sql`

```sql
CREATE TABLE IF NOT EXISTS alert_rankings (
    processing_date Date,
    network String,
    alert_id String,
    rank Int32,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, rank)
SETTINGS index_granularity = 8192;
```

### `storage/schema/cluster_scores.sql`

```sql
CREATE TABLE IF NOT EXISTS cluster_scores (
    processing_date Date,
    network String,
    cluster_id String,
    score Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;
```

### `storage/schema/batch_metadata.sql`

```sql
CREATE TABLE IF NOT EXISTS batch_metadata (
    processing_date Date,
    network String,
    processed_at DateTime,
    input_counts Nested(
        alerts Int32,
        features Int32,
        clusters Int32
    ),
    output_counts Nested(
        alert_scores Int32,
        alert_rankings Int32,
        cluster_scores Int32
    ),
    latencies_ms Nested(
        alert_scoring Int32,
        alert_ranking Int32,
        cluster_scoring Int32,
        total Int32
    ),
    model_versions Nested(
        alert_scorer String,
        alert_ranker String,
        cluster_scorer String
    ),
    status Enum8('PROCESSING' = 1, 'COMPLETED' = 2, 'FAILED' = 3) DEFAULT 'PROCESSING',
    error_message String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network)
SETTINGS index_granularity = 8192;
```

---

## Configuration

### `aml_scoring/config/settings.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TRAINED_MODELS_DIR: Path = BASE_DIR / "trained_models"
    
    # ClickHouse Configuration
    CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    CLICKHOUSE_DATABASE: str = os.getenv("CLICKHOUSE_DATABASE", "aml_scoring")
    CLICKHOUSE_USER: str = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    
    # Network
    NETWORK: str = os.getenv("NETWORK", "ethereum")
    
    # Models
    ALERT_SCORER_PATH: Path = TRAINED_MODELS_DIR / "alert_scorer_v1.0.0.txt"
    ALERT_RANKER_PATH: Path = TRAINED_MODELS_DIR / "alert_ranker_v1.0.0.txt"
    CLUSTER_SCORER_PATH: Path = TRAINED_MODELS_DIR / "cluster_scorer_v1.0.0.txt"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Retention
    RETENTION_DAYS: int = 90
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## Complete Workflow

### 1. Initialize Database

```bash
python scripts/init_database.py --network ethereum
```

```python
# scripts/init_database.py

from aml_scoring.storage import get_connection_params, create_database, ClientFactory, MigrateSchema
from aml_scoring.config.settings import Settings

settings = Settings()
connection_params = get_connection_params(settings.NETWORK)

# Create database
create_database(connection_params)

# Run migrations
client_factory = ClientFactory(connection_params)
with client_factory.client_context() as client:
    migrator = MigrateSchema(client)
    migrator.run_migrations()

print(f"✓ Database initialized for {settings.NETWORK}")
```

### 2. Download from SOT

```bash
python scripts/download_from_sot.py \
    --network ethereum \
    --processing-date 2025-10-26
```

### 3. Process Batch

```bash
python scripts/process_batch.py \
    --network ethereum \
    --processing-date 2025-10-26
```

### 4. Start API

```bash
python -m aml_scoring.api.server
```

### 5. Query Scores

```bash
# Get latest scores
curl http://localhost:8000/scores/alerts/latest?network=ethereum

# Get specific date
curl http://localhost:8000/scores/alerts/2025-10-26?network=ethereum

# Get available dates
curl http://localhost:8000/dates/available?network=ethereum
```

---

## API Endpoints (Final)

```python
# Health & Info
GET  /health
GET  /version

# Date Discovery
GET  /dates/available?network={network}
GET  /dates/latest?network={network}

# Scores - Latest
GET  /scores/alerts/latest?network={network}
GET  /rankings/alerts/latest?network={network}
GET  /scores/clusters/latest?network={network}

# Scores - Specific Date
GET  /scores/alerts/{processing_date}?network={network}
GET  /rankings/alerts/{processing_date}?network={network}
GET  /scores/clusters/{processing_date}?network={network}

# Metadata
GET  /metadata/{processing_date}?network={network}
GET  /metadata/latest?network={network}
```

---

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "aml-alert-scoring"
version = "1.0.0"
description = "AML Alert Risk Scoring System"

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "lightgbm>=4.3.0",
    "shap>=0.44.0",
    "loguru>=0.7.2",
    "httpx>=0.26.0",
    "pyarrow>=15.0.0",
    "clickhouse-connect>=0.7.0",
]
```

---

## Environment Configuration

### `.env.example`

```bash
# Network
NETWORK=ethereum

# ClickHouse
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE=aml_scoring
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=

# API
API_HOST=0.0.0.0
API_PORT=8000

# Retention
RETENTION_DAYS=90
```

---

## Summary

### Repository Name
**`aml-alert-scoring`** ⭐

### Key Decisions
- ✅ **ClickHouse only** - No DuckDB, consistent with SOT
- ✅ **Clean table names** - No `miner_` prefix
- ✅ **Repository pattern** - Clean data access layer
- ✅ **Production-ready** - Proper error handling, migrations
- ✅ **Network-aware** - Multi-network support (ethereum, bitcoin, polygon)

### Package Structure
- `aml_scoring` - Main package (renamed from `aml_miner`)
- `aml_scoring.storage` - ClickHouse storage layer
- `aml_scoring.models` - ML models
- `aml_scoring.validation` - Validation framework
- `aml_scoring.api` - FastAPI server

This is the **final, production-ready specification** ready for implementation!