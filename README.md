# Alert Scoring Template

ML-powered alert scoring system with ClickHouse storage and FastAPI.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Architecture

**Batch Processing Workflow:**
1. Download data from SOT ClickHouse → Local ClickHouse
2. Process alerts offline with ML models
3. Store scores/rankings in ClickHouse
4. Serve via REST API

**Storage:** ClickHouse (production-grade OLAP database)  
**API:** FastAPI with GET endpoints  
**Validation:** Three-tier framework (Integrity, Behavior, Ground Truth)

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Initialize Database
```bash
python scripts/init_database.py --network ethereum
```

### 3. Download Data from SOT
```bash
python scripts/download_from_sot.py \
    --processing-date 2024-01-15 \
    --network ethereum \
    --sot-host sot.clickhouse.example.com
```

### 4. Process Batch
```bash
python scripts/process_batch.py \
    --processing-date 2024-01-15 \
    --network ethereum
```

### 5. Start API Server
```bash
python -m aml_miner.api.server
# Server at http://localhost:8000
```

### 6. Query Scores
```bash
curl http://localhost:8000/scores/alerts/latest?network=ethereum
```

📖 **Full setup guide:** [Quick Start Documentation](docs/quickstart.md)

## Configuration

Edit [`aml_miner/config/settings.py`](aml_miner/config/settings.py:1) or use environment variables:

```bash
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE=alert_scoring
NETWORK=ethereum
```

## API Endpoints

### Alert Scores
- `GET /scores/alerts/latest?network={network}` - Latest scores
- `GET /scores/alerts/{date}?network={network}` - Scores for date
- `GET /dates/available?network={network}` - All dates
- `GET /dates/latest?network={network}` - Latest date

### Rankings
- `GET /scores/rankings/latest?network={network}` - Latest rankings
- `GET /scores/rankings/{date}?network={network}` - Rankings for date
- `GET /scores/rankings/top/{n}?network={network}` - Top N alerts

### Cluster Scores
- `GET /scores/clusters/latest?network={network}` - Latest cluster scores
- `GET /scores/clusters/{date}?network={network}` - Cluster scores for date

### Metadata
- `GET /metadata/latest?network={network}` - Latest batch metadata
- `GET /metadata/{date}?network={network}` - Metadata for date
- `GET /metadata/history?network={network}&limit=30` - Metadata history

📖 **Full API documentation:** [API Reference](docs/api_reference.md)

## Project Structure

```
alert-scoring/
├── aml_miner/
│   ├── api/              # FastAPI server & routes
│   ├── config/           # Configuration
│   ├── features/         # Feature engineering
│   ├── models/           # ML models
│   ├── storage/          # ClickHouse storage layer
│   │   ├── repositories/ # Data access layer
│   │   └── schema/       # Database schemas
│   ├── training/         # Model training
│   └── validation/       # Validation framework
├── scripts/
│   ├── init_database.py      # Database setup
│   ├── download_from_sot.py  # Download from SOT
│   ├── process_batch.py      # Process & score
│   └── validate_models.py    # A/B testing
├── docs/                 # Documentation
└── trained_models/       # Pretrained models
```

## Validation Framework

Three-tier validation matching Bittensor validator logic:

1. **Integrity (0-0.2):** Schema, completeness, latency, determinism
2. **Behavior (0-0.3):** Pattern traps, plagiarism detection, variance
3. **Ground Truth (0-0.5):** AUC-ROC, AUC-PR, F1 vs T+τ labels

Total score: 0-1.0

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│               SOT ClickHouse (Source of Truth)           │
│  - analyzers_alerts                                      │
│  - analyzers_features                                    │
│  - analyzers_alert_clusters                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 1. Download (download_from_sot.py)
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Local ClickHouse Database                   │
│  - raw_alerts                                            │
│  - raw_features                                          │
│  - raw_clusters                                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 2. Process (process_batch.py)
                       ▼
┌─────────────────────────────────────────────────────────┐
│         ML Models (XGBoost + LambdaMART)                 │
│  - Alert Scorer                                          │
│  - Alert Ranker                                          │
│  - Cluster Scorer                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 3. Store Results
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Local ClickHouse Database                   │
│  - alert_scores                                          │
│  - alert_rankings                                        │
│  - cluster_scores                                        │
│  - batch_metadata                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 4. Serve (FastAPI)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   REST API Endpoints                     │
│  GET /scores/alerts/latest                               │
│  GET /scores/rankings/latest                             │
│  GET /scores/clusters/latest                             │
└─────────────────────────────────────────────────────────┘
```

## Development

### Run Tests
```bash
pytest tests/
```

### Validate Models
```bash
python scripts/validate_models.py \
    --model-a trained_models/model_v1.txt \
    --model-b trained_models/model_v2.txt \
    --processing-date 2024-01-15
```

### Database Management
```bash
# Initialize database
python scripts/init_database.py --network ethereum

# Check database status
python -c "from aml_miner.storage import ClientFactory, get_connection_params; \
    factory = ClientFactory(get_connection_params('ethereum')); \
    with factory.client_context() as c: print(c.query('SHOW TABLES').result_rows)"
```

## Example Usage

### Complete Workflow

```bash
# Step 1: Initialize database for Ethereum
python scripts/init_database.py --network ethereum

# Step 2: Download data from SOT for specific date
python scripts/download_from_sot.py \
    --processing-date 2024-01-15 \
    --network ethereum \
    --sot-host sot.clickhouse.example.com \
    --sot-port 8123

# Step 3: Process the batch (score & rank)
python scripts/process_batch.py \
    --processing-date 2024-01-15 \
    --network ethereum

# Step 4: Start API server
python -m aml_miner.api.server

# Step 5: Query results
curl http://localhost:8000/scores/alerts/latest?network=ethereum
curl http://localhost:8000/scores/rankings/top/100?network=ethereum
curl http://localhost:8000/metadata/latest?network=ethereum
```

### API Examples

```bash
# Get latest alert scores
curl http://localhost:8000/scores/alerts/latest?network=ethereum

# Get scores for specific date
curl http://localhost:8000/scores/alerts/2024-01-15?network=ethereum

# Get top 100 ranked alerts
curl http://localhost:8000/scores/rankings/top/100?network=ethereum

# Get all available processing dates
curl http://localhost:8000/dates/available?network=ethereum

# Get batch metadata with processing stats
curl http://localhost:8000/metadata/latest?network=ethereum
```

## Multi-Network Support

The system supports multiple blockchain networks:

```bash
# Ethereum
python scripts/init_database.py --network ethereum
python scripts/download_from_sot.py --network ethereum --processing-date 2024-01-15
python scripts/process_batch.py --network ethereum --processing-date 2024-01-15

# Bitcoin
python scripts/init_database.py --network bitcoin
python scripts/download_from_sot.py --network bitcoin --processing-date 2024-01-15
python scripts/process_batch.py --network bitcoin --processing-date 2024-01-15
```

## Storage Architecture

### ClickHouse Tables

**Raw Data Tables:**
- `raw_alerts` - Downloaded alert data from SOT
- `raw_features` - Downloaded feature data from SOT
- `raw_clusters` - Downloaded cluster data from SOT

**Processed Data Tables:**
- `alert_scores` - ML model scores for alerts
- `alert_rankings` - Ranked alerts by priority
- `cluster_scores` - ML model scores for clusters
- `batch_metadata` - Processing statistics and metadata

### Data Partitioning

All tables are partitioned by:
- `processing_date` - Date when batch was processed
- `network` - Blockchain network (ethereum, bitcoin, etc.)

This enables efficient queries and data management.

## License

MIT

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get running in minutes
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Training Guide](docs/training_guide.md)** - Train custom models
- **[Customization Guide](docs/customization.md)** - Extend functionality

## Support

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Open an issue on GitHub
- **Architecture**: See [docs/agent/2025-10-26/claude/](docs/agent/2025-10-26/claude/) for detailed architecture docs

---

**Built with ❤️ for the AML community**