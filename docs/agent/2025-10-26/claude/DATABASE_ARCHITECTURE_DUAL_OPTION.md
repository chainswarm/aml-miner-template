# Database Architecture: DuckDB vs ClickHouse
## Miner's Choice for Local Data Storage

**Date**: 2025-10-26  
**Purpose**: Define dual database option architecture for miner template

---

## Core Architecture Decision

### Refined Workflow

```
SOT (ClickHouse) 
    ↓
Download to Local DB (ClickHouse OR DuckDB - miner's choice)
    ↓
ML Scoring (process_batch.py reads from DB)
    ↓  
Store Results in Local DB
    ↓
API queries Local DB
```

**Key Insight**: Instead of Parquet intermediary, use database as primary storage.

---

## Option 1: ClickHouse (Recommended)

### Why ClickHouse?

✅ **Consistency with SOT** - Same technology as source  
✅ **Familiar for development** - Same queries work everywhere  
✅ **Production-grade** - Battle-tested for analytics  
✅ **Better performance** - Optimized for OLAP workloads  
✅ **Direct replication** - Can mirror SOT schema exactly  

### Architecture

```
┌─────────────────────────────────────────────────┐
│           CLICKHOUSE ARCHITECTURE                │
└─────────────────────────────────────────────────┘

SOT ClickHouse (Remote)
    ↓ SQL query or Parquet export
Local ClickHouse (Miner's machine)
    ├─ Database: miner_db
    ├─ Tables:
    │   ├─ raw_alerts          (from SOT)
    │   ├─ raw_features         (from SOT)
    │   ├─ raw_clusters         (from SOT)
    │   ├─ alert_scores         (computed)
    │   ├─ alert_rankings       (computed)
    │   └─ cluster_scores       (computed)
    │
    └─ ML Processing
        ├─ Read: SELECT * FROM raw_alerts WHERE processing_date = '2025-10-26'
        ├─ Score: ML models
        └─ Write: INSERT INTO alert_scores ...
```

### Schema

```sql
-- Raw data from SOT
CREATE TABLE miner_db.raw_alerts (
    window_days Int32,
    processing_date Date,
    alert_id String,
    address String,
    typology_type String,
    pattern_id String,
    severity String,
    alert_confidence_score Float64,
    description String,
    volume_usd Decimal(18, 2),
    evidence_json String,
    network String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, alert_id);

-- Computed scores
CREATE TABLE miner_db.alert_scores (
    processing_date Date,
    network String,
    alert_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    explain_json String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, alert_id);

-- Computed rankings
CREATE TABLE miner_db.alert_rankings (
    processing_date Date,
    network String,
    alert_id String,
    rank Int32,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, rank);
```

### Implementation

```python
# aml_miner/storage/clickhouse_storage.py

from clickhouse_driver import Client
from typing import List, Dict
import pandas as pd
from loguru import logger

class ClickHouseStorage:
    def __init__(self, host: str = 'localhost', port: int = 9000, database: str = 'miner_db'):
        self.client = Client(host=host, port=port, database=database)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        # Create database
        self.client.execute('CREATE DATABASE IF NOT EXISTS miner_db')
        
        # Create tables (schemas above)
        self.client.execute('''
            CREATE TABLE IF NOT EXISTS miner_db.raw_alerts (
                -- schema from above
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(processing_date)
            ORDER BY (processing_date, alert_id)
        ''')
        
        # ... create other tables
    
    def insert_alerts(self, alerts_df: pd.DataFrame, processing_date: str, network: str):
        """Insert raw alerts from SOT"""
        alerts_df['processing_date'] = processing_date
        alerts_df['network'] = network
        
        self.client.insert_dataframe(
            'INSERT INTO miner_db.raw_alerts VALUES',
            alerts_df
        )
        
        logger.info(f"Inserted {len(alerts_df)} alerts for {processing_date}")
    
    def get_alerts(self, processing_date: str, network: str) -> pd.DataFrame:
        """Get alerts for processing"""
        query = '''
            SELECT * FROM miner_db.raw_alerts
            WHERE processing_date = %(date)s AND network = %(network)s
        '''
        
        result = self.client.execute(query, {'date': processing_date, 'network': network})
        
        return pd.DataFrame(result, columns=[
            'window_days', 'processing_date', 'alert_id', 'address',
            'typology_type', 'pattern_id', 'severity', 'alert_confidence_score',
            'description', 'volume_usd', 'evidence_json', 'network'
        ])
    
    def insert_scores(self, scores_df: pd.DataFrame, processing_date: str, network: str):
        """Insert computed scores"""
        scores_df['processing_date'] = processing_date
        scores_df['network'] = network
        
        self.client.insert_dataframe(
            'INSERT INTO miner_db.alert_scores VALUES',
            scores_df
        )
        
        logger.info(f"Inserted {len(scores_df)} scores for {processing_date}")
    
    def get_scores(self, processing_date: str, network: str) -> pd.DataFrame:
        """Get scores for API"""
        query = '''
            SELECT alert_id, score, model_version, latency_ms, explain_json
            FROM miner_db.alert_scores
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY score DESC
        '''
        
        result = self.client.execute(query, {'date': processing_date, 'network': network})
        
        return pd.DataFrame(result, columns=[
            'alert_id', 'score', 'model_version', 'latency_ms', 'explain_json'
        ])
    
    def get_latest_date(self, network: str) -> str:
        """Get latest processing date"""
        query = '''
            SELECT max(processing_date) as latest
            FROM miner_db.alert_scores
            WHERE network = %(network)s
        '''
        
        result = self.client.execute(query, {'network': network})
        
        return str(result[0][0]) if result and result[0][0] else None
    
    def get_available_dates(self, network: str) -> List[str]:
        """Get all available processing dates"""
        query = '''
            SELECT DISTINCT processing_date
            FROM miner_db.alert_scores
            WHERE network = %(network)s
            ORDER BY processing_date DESC
        '''
        
        result = self.client.execute(query, {'network': network})
        
        return [str(row[0]) for row in result]
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Remove data older than retention period"""
        query = '''
            ALTER TABLE miner_db.alert_scores
            DELETE WHERE processing_date < today() - INTERVAL %(days)s DAY
        '''
        
        self.client.execute(query, {'days': retention_days})
        
        logger.info(f"Cleaned up data older than {retention_days} days")
```

---

## Option 2: DuckDB (Lightweight)

### Why DuckDB?

✅ **Zero infrastructure** - No server to run  
✅ **Simple deployment** - Just pip install  
✅ **Embedded** - Runs in-process  
✅ **Good for development** - Fast iteration  
✅ **Lower resource usage** - Minimal memory/CPU  

### Architecture

```
┌─────────────────────────────────────────────────┐
│            DUCKDB ARCHITECTURE                   │
└─────────────────────────────────────────────────┘

SOT → Parquet files
    ↓
DuckDB (in-process)
    ├─ Database: miner.db (single file)
    ├─ Tables:
    │   ├─ raw_alerts          (created from Parquet)
    │   ├─ raw_features         (created from Parquet)
    │   ├─ raw_clusters         (created from Parquet)
    │   ├─ alert_scores         (computed)
    │   ├─ alert_rankings       (computed)
    │   └─ cluster_scores       (computed)
    │
    └─ ML Processing
        ├─ Read: SELECT * FROM raw_alerts WHERE processing_date = '2025-10-26'
        ├─ Score: ML models
        └─ Write: INSERT INTO alert_scores ...
```

### Implementation

```python
# aml_miner/storage/duckdb_storage.py

import duckdb
from pathlib import Path
from typing import List
import pandas as pd
from loguru import logger

class DuckDBStorage:
    def __init__(self, db_path: str = "miner.db"):
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS raw_alerts (
                window_days INTEGER,
                processing_date DATE,
                alert_id VARCHAR,
                address VARCHAR,
                typology_type VARCHAR,
                pattern_id VARCHAR,
                severity VARCHAR,
                alert_confidence_score DOUBLE,
                description VARCHAR,
                volume_usd DECIMAL(18, 2),
                evidence_json VARCHAR,
                network VARCHAR
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS alert_scores (
                processing_date DATE,
                network VARCHAR,
                alert_id VARCHAR,
                score DOUBLE,
                model_version VARCHAR,
                latency_ms DOUBLE,
                explain_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_alert_scores_date 
            ON alert_scores(processing_date, network)
        ''')
    
    def insert_alerts_from_parquet(self, parquet_path: str, processing_date: str, network: str):
        """Load alerts from Parquet file"""
        self.conn.execute(f'''
            INSERT INTO raw_alerts 
            SELECT *, '{processing_date}'::DATE as processing_date, '{network}' as network
            FROM read_parquet('{parquet_path}')
        ''')
        
        count = self.conn.execute('SELECT count(*) FROM raw_alerts WHERE processing_date = ?', [processing_date]).fetchone()[0]
        logger.info(f"Inserted {count} alerts for {processing_date}")
    
    def get_alerts(self, processing_date: str, network: str) -> pd.DataFrame:
        """Get alerts for processing"""
        return self.conn.execute('''
            SELECT * FROM raw_alerts
            WHERE processing_date = ? AND network = ?
        ''', [processing_date, network]).df()
    
    def insert_scores(self, scores_df: pd.DataFrame, processing_date: str, network: str):
        """Insert computed scores"""
        scores_df['processing_date'] = processing_date
        scores_df['network'] = network
        
        self.conn.register('scores_temp', scores_df)
        self.conn.execute('''
            INSERT INTO alert_scores 
            SELECT * FROM scores_temp
        ''')
        
        logger.info(f"Inserted {len(scores_df)} scores for {processing_date}")
    
    def get_scores(self, processing_date: str, network: str) -> pd.DataFrame:
        """Get scores for API"""
        return self.conn.execute('''
            SELECT alert_id, score, model_version, latency_ms, explain_json
            FROM alert_scores
            WHERE processing_date = ? AND network = ?
            ORDER BY score DESC
        ''', [processing_date, network]).df()
    
    def get_latest_date(self, network: str) -> str:
        """Get latest processing date"""
        result = self.conn.execute('''
            SELECT max(processing_date) as latest
            FROM alert_scores
            WHERE network = ?
        ''', [network]).fetchone()
        
        return str(result[0]) if result and result[0] else None
    
    def get_available_dates(self, network: str) -> List[str]:
        """Get all available processing dates"""
        result = self.conn.execute('''
            SELECT DISTINCT processing_date
            FROM alert_scores
            WHERE network = ?
            ORDER BY processing_date DESC
        ''', [network]).fetchall()
        
        return [str(row[0]) for row in result]
```

---

## Unified Interface

### Storage Factory

```python
# aml_miner/storage/__init__.py

from enum import Enum
from aml_miner.storage.clickhouse_storage import ClickHouseStorage
from aml_miner.storage.duckdb_storage import DuckDBStorage

class StorageType(Enum):
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"

def create_storage(storage_type: StorageType, **kwargs):
    if storage_type == StorageType.CLICKHOUSE:
        return ClickHouseStorage(**kwargs)
    elif storage_type == StorageType.DUCKDB:
        return DuckDBStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
```

### Configuration

```python
# aml_miner/config/settings.py

from pydantic_settings import BaseSettings
from aml_miner.storage import StorageType

class Settings(BaseSettings):
    # Storage configuration
    STORAGE_TYPE: StorageType = StorageType.DUCKDB  # or CLICKHOUSE
    
    # ClickHouse settings (if STORAGE_TYPE = CLICKHOUSE)
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_DATABASE: str = "miner_db"
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: str = ""
    
    # DuckDB settings (if STORAGE_TYPE = DUCKDB)
    DUCKDB_PATH: str = "miner.db"
    
    # Network
    NETWORK: str = "ethereum"
    
    # Retention
    RETENTION_DAYS: int = 90
```

---

## Updated Workflow

### 1. Download from SOT

```python
# scripts/download_from_sot.py

from aml_miner.storage import create_storage, StorageType
from aml_miner.config.settings import Settings

settings = Settings()
storage = create_storage(settings.STORAGE_TYPE)

# For ClickHouse: Direct SQL query from SOT
if settings.STORAGE_TYPE == StorageType.CLICKHOUSE:
    # Query SOT ClickHouse directly
    sot_client = Client(host='sot.example.com')
    alerts = sot_client.execute('''
        SELECT * FROM analyzers_alerts 
        WHERE processing_date = '2025-10-26' AND network = 'ethereum'
    ''')
    
    # Insert into local ClickHouse
    storage.insert_alerts(alerts_df, '2025-10-26', 'ethereum')

# For DuckDB: Download Parquet, then load
else:
    # Download Parquet files
    download_parquet('2025-10-26', 'input/2025-10-26/')
    
    # Load into DuckDB
    storage.insert_alerts_from_parquet(
        'input/2025-10-26/alerts.parquet',
        '2025-10-26',
        'ethereum'
    )
```

### 2. Process Batch

```python
# scripts/process_batch.py (updated)

from aml_miner.storage import create_storage
from aml_miner.models.alert_scorer import AlertScorerModel

storage = create_storage(settings.STORAGE_TYPE)

# Read from database
alerts_df = storage.get_alerts('2025-10-26', 'ethereum')
features_df = storage.get_features('2025-10-26', 'ethereum')

# Score
scorer = AlertScorerModel()
X = scorer.prepare_features(alerts_df, features_df)
scores = scorer.predict(X)

# Create scores dataframe
scores_df = pd.DataFrame({
    'alert_id': alerts_df['alert_id'],
    'score': scores,
    'model_version': scorer.model_version,
    'latency_ms': latency_per_alert
})

# Write back to database
storage.insert_scores(scores_df, '2025-10-26', 'ethereum')
```

### 3. API Serves from Database

```python
# aml_miner/api/routes.py (updated)

from aml_miner.storage import create_storage
from aml_miner.config.settings import Settings

settings = Settings()
storage = create_storage(settings.STORAGE_TYPE)

@router.get("/scores/alerts/{processing_date}")
def get_alert_scores(processing_date: str, network: str = settings.NETWORK):
    scores_df = storage.get_scores(processing_date, network)
    
    if scores_df.empty:
        raise HTTPException(404, f"No scores for {processing_date}")
    
    return {
        "processing_date": processing_date,
        "network": network,
        "scores": scores_df.to_dict(orient='records')
    }

@router.get("/scores/alerts/latest")
def get_latest_alert_scores(network: str = settings.NETWORK):
    latest_date = storage.get_latest_date(network)
    
    if not latest_date:
        raise HTTPException(404, "No scores available")
    
    scores_df = storage.get_scores(latest_date, network)
    
    return {
        "processing_date": latest_date,
        "network": network,
        "scores": scores_df.to_dict(orient='records'),
        "metadata": {
            "latest": True,
            "total_scores": len(scores_df)
        }
    }
```

---

## Comparison Matrix

| Feature | ClickHouse | DuckDB |
|---------|------------|--------|
| **Setup** | Install server | pip install |
| **Consistency with SOT** | ✅ Same tech | ❌ Different |
| **Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very good |
| **Resource Usage** | Medium-High | Low |
| **Scalability** | Distributed | Single-node |
| **Query Features** | Full SQL | Full SQL |
| **Best For** | Production, consistency | Development, simplicity |

---

## Recommendation

### Default: DuckDB
- Easier onboarding for miners
- Lower barrier to entry
- Good enough performance

### Advanced: ClickHouse
- For miners who want production setup
- Direct SOT integration
- Better long-term performance

### Implementation
Provide both options in template, default to DuckDB, document ClickHouse upgrade path.