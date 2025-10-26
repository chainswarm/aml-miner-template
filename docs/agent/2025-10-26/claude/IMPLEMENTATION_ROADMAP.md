# Implementation Roadmap
## ClickHouse Storage Layer - Step by Step

**Date**: 2025-10-26  
**Status**: In Progress

---

## ✅ Completed

1. **Architecture Documentation** (9 documents)
2. **Validation Framework** (IntegrityValidator, BehaviorValidator, GroundTruthValidator)
3. **Storage Layer Foundation**:
   - [`aml_miner/storage/__init__.py`](aml_miner/storage/__init__.py:1) - ClientFactory, migrations
   - [`aml_miner/storage/utils.py`](aml_miner/storage/utils.py:1) - Pydantic helpers
   - [`aml_miner/storage/schema/alert_scores.sql`](aml_miner/storage/schema/alert_scores.sql:1) - First schema file

---

## 🔄 In Progress: Storage Layer Implementation

### Remaining Schema Files (6 files)

Create in `aml_miner/storage/schema/`:

1. `raw_alerts.sql` - Raw alerts from SOT
2. `raw_features.sql` - Raw features from SOT
3. `raw_clusters.sql` - Raw clusters from SOT
4. `alert_rankings.sql` - Computed rankings
5. `cluster_scores.sql` - Computed cluster scores
6. `batch_metadata.sql` - Processing metadata

### Repository Classes (7 files)

Create in `aml_miner/storage/repositories/`:

1. `__init__.py` - Repository exports
2. `base_repository.py` - ABC base class
3. `alerts_repository.py` - Raw alerts CRUD
4. `features_repository.py` - Raw features CRUD
5. `clusters_repository.py` - Raw clusters CRUD
6. `scores_repository.py` - Alert scores CRUD
7. `metadata_repository.py` - Batch metadata CRUD

---

## 📋 Next Steps

### 1. Update Dependencies

**File**: `pyproject.toml`

Add:
```toml
dependencies = [
    # ... existing
    "clickhouse-connect>=0.7.0",  # ClickHouse client
]
```

Remove:
```toml
"duckdb>=0.9.0",  # No longer needed
```

### 2. Update Configuration

**File**: `aml_miner/config/settings.py`

```python
class Settings(BaseSettings):
    # ClickHouse Configuration
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_DATABASE: str = "alert_scoring"
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: str = ""
    
    # Network
    NETWORK: str = "ethereum"
    
    # Retention
    RETENTION_DAYS: int = 90
```

### 3. Create Initialization Script

**File**: `scripts/init_database.py`

```python
from aml_miner.storage import get_connection_params, create_database, ClientFactory, MigrateSchema

def main():
    network = "ethereum"
    connection_params = get_connection_params(network)
    
    # Create database
    create_database(connection_params)
    
    # Run migrations
    client_factory = ClientFactory(connection_params)
    with client_factory.client_context() as client:
        migrator = MigrateSchema(client)
        migrator.run_migrations()
    
    print(f"✓ Database initialized for {network}")

if __name__ == "__main__":
    main()
```

### 4. Update API Routes

**File**: `aml_miner/api/routes.py`

Replace DuckDB database with ClickHouse storage repositories:

```python
from aml_miner.storage import ClientFactory, get_connection_params
from aml_miner.storage.repositories.scores_repository import ScoresRepository
from aml_miner.config.settings import Settings

settings = Settings()
connection_params = get_connection_params(settings.NETWORK)
client_factory = ClientFactory(connection_params)

@router.get("/scores/alerts/latest")
def get_latest_alert_scores(network: str = settings.NETWORK):
    with client_factory.client_context() as client:
        scores_repo = ScoresRepository(client)
        latest_date = scores_repo.get_latest_date(network)
        
        if not latest_date:
            raise HTTPException(404, "No scores available")
        
        scores = scores_repo.get_scores(latest_date, network)
        
        return {
            "processing_date": latest_date,
            "network": network,
            "scores": [score.dict() for score in scores]
        }
```

### 5. Update Process Batch Script

**File**: `scripts/process_batch.py`

Replace file-based storage with ClickHouse repositories:

```python
from aml_miner.storage import ClientFactory, get_connection_params
from aml_miner.storage.repositories.scores_repository import ScoresRepository

# After scoring
with client_factory.client_context() as client:
    scores_repo = ScoresRepository(client)
    scores_repo.insert_scores(scores_df, processing_date, network)
```

### 6. Update Download Script

**File**: `scripts/download_from_sot.py`

Connect to SOT ClickHouse and copy to local:

```python
from aml_miner.storage import ClientFactory, get_connection_params
from aml_miner.storage.repositories.alerts_repository import AlertsRepository

# Connect to SOT
sot_params = {
    "host": "sot.example.com",
    "database": "sot_ethereum",
    # ...
}
sot_client = ClientFactory(sot_params)

# Connect to local
local_params = get_connection_params("ethereum")
local_client = ClientFactory(local_params)

# Copy data
with sot_client.client_context() as sot, local_client.client_context() as local:
    # Query SOT
    alerts = sot.query("SELECT * FROM analyzers_alerts WHERE processing_date = '2025-10-26'")
    
    # Insert to local
    alerts_repo = AlertsRepository(local)
    alerts_repo.insert_alerts(alerts_df, '2025-10-26', 'ethereum')
```

---

## 🎯 Implementation Order

### Phase 1: Complete Storage Layer (Current)
1. ✅ Create storage `__init__.py`
2. ✅ Create `utils.py`
3. ✅ Create `alert_scores.sql`
4. ⏳ Create remaining 6 SQL schema files
5. ⏳ Create 7 repository classes

### Phase 2: Configuration & Dependencies
1. Update `pyproject.toml` (add clickhouse-connect)
2. Update `aml_miner/config/settings.py` (ClickHouse settings)
3. Create `scripts/init_database.py`

### Phase 3: Update Application Layer
1. Update `aml_miner/api/routes.py` (use repositories)
2. Update `scripts/process_batch.py` (use repositories)
3. Update `scripts/download_from_sot.py` (SOT integration)

### Phase 4: Testing & Documentation
1. Test database initialization
2. Test end-to-end workflow
3. Update user documentation

---

## 🚀 Quick Start (After Implementation)

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with ClickHouse credentials

# 3. Initialize database
python scripts/init_database.py --network ethereum

# 4. Download batch from SOT
python scripts/download_from_sot.py \
    --network ethereum \
    --processing-date 2025-10-26

# 5. Process batch
python scripts/process_batch.py \
    --network ethereum \
    --processing-date 2025-10-26

# 6. Start API
python -m aml_miner.api.server

# 7. Query scores
curl "http://localhost:8000/scores/alerts/latest?network=ethereum"
```

---

## 📊 Progress Summary

**Total Tasks**: 30  
**Completed**: 12 (40%)  
**In Progress**: 7 (23%)  
**Remaining**: 11 (37%)

**Estimated Time to Complete**: 2-3 hours for full implementation and testing

---

## 🔧 Current Session Status

Due to conversation length and complexity, I recommend:

1. **This Session**: Complete storage layer foundation (schema files + base repository)
2. **Next Session**: Complete repository implementations + API updates
3. **Final Session**: End-to-end testing + documentation

**Files Created So Far**:
- `aml_miner/storage/__init__.py` ✅
- `aml_miner/storage/utils.py` ✅
- `aml_miner/storage/schema/alert_scores.sql` ✅

**Next Immediate Files**:
- Remaining 6 SQL schemas
- `base_repository.py`
- `scores_repository.py`