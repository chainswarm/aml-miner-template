# Quick Start Guide

Get the Alert Scoring Template running in minutes with this step-by-step guide.

## Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **ClickHouse Server**
   
   **Option A: Docker (Recommended)**
   ```bash
   docker run -d --name clickhouse-server \
     -p 8123:8123 \
     -p 9000:9000 \
     --ulimit nofile=262144:262144 \
     clickhouse/clickhouse-server
   ```
   
   **Option B: Native Installation**
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get install -y apt-transport-https ca-certificates dirmngr
     sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 8919F6BD2B48D754
     echo "deb https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list
     sudo apt-get update
     sudo apt-get install -y clickhouse-server clickhouse-client
     sudo service clickhouse-server start
     ```
   
   - **macOS:**
     ```bash
     brew install clickhouse
     clickhouse-server
     ```

3. **Git**
   ```bash
   git --version
   ```

### System Requirements

- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 10GB free space
- **OS:** Linux, macOS, or Windows with WSL

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/aml-miner-template.git
cd aml-miner-template
```

### Step 2: Install Dependencies

**Option A: Using pip**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .
```

**Option B: Using UV (10-100x faster)**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Step 3: Verify Installation

```bash
# Check imports work
python -c "from aml_miner.api import server; print('✓ Installation successful')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# ClickHouse Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE=alert_scoring
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=

# Default Network
NETWORK=ethereum

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Alternative: Python Configuration

Edit [`aml_miner/config/settings.py`](../aml_miner/config/settings.py:1) directly:

```python
class Settings(BaseSettings):
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_DATABASE: str = "alert_scoring"
    NETWORK: str = "ethereum"
```

## First Run

### 1. Initialize Database

Create the ClickHouse database and tables:

```bash
python scripts/init_database.py --network ethereum
```

**Expected output:**
```
Creating database alert_scoring...
Creating table raw_alerts...
Creating table raw_features...
Creating table raw_clusters...
Creating table alert_scores...
Creating table alert_rankings...
Creating table cluster_scores...
Creating table batch_metadata...
✓ Database initialized successfully
```

### 2. Download Data from SOT

Download data from the Source of Truth (SOT) ClickHouse:

```bash
python scripts/download_from_sot.py \
    --processing-date 2024-01-15 \
    --network ethereum \
    --sot-host sot.clickhouse.example.com \
    --sot-port 8123 \
    --sot-database sot_production
```

**Parameters:**
- `--processing-date`: Date to process (YYYY-MM-DD format)
- `--network`: Blockchain network (ethereum, bitcoin, polygon, etc.)
- `--sot-host`: SOT ClickHouse server hostname
- `--sot-port`: SOT ClickHouse server port (default: 8123)
- `--sot-database`: SOT database name (default: sot_production)

**Expected output:**
```
Downloading alerts from SOT...
Downloaded 15,234 alerts
Downloading features from SOT...
Downloaded 15,234 feature records
Downloading clusters from SOT...
Downloaded 3,456 clusters
✓ Data downloaded successfully to local ClickHouse
```

### 3. Process Batch

Process the downloaded data with ML models:

```bash
python scripts/process_batch.py \
    --processing-date 2024-01-15 \
    --network ethereum
```

**Expected output:**
```
Loading models...
✓ Loaded alert scorer: trained_models/alert_scorer_v1.0.0.txt
✓ Loaded alert ranker: trained_models/alert_ranker_v1.0.0.txt
✓ Loaded cluster scorer: trained_models/cluster_scorer_v1.0.0.txt

Processing 15,234 alerts...
✓ Scored 15,234 alerts in 2.3s
✓ Ranked 15,234 alerts in 1.8s

Processing 3,456 clusters...
✓ Scored 3,456 clusters in 0.9s

Saving results to ClickHouse...
✓ Saved 15,234 alert scores
✓ Saved 15,234 alert rankings
✓ Saved 3,456 cluster scores
✓ Saved batch metadata

Total processing time: 5.2s
```

### 4. Start API Server

Launch the FastAPI server:

```bash
python -m aml_miner.api.server
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The API is now running at `http://localhost:8000`

### 5. Query Results

**Test health endpoint:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "storage": "clickhouse",
  "default_network": "ethereum"
}
```

**Get latest alert scores:**
```bash
curl http://localhost:8000/scores/alerts/latest?network=ethereum
```

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
      "model_version": "1.0.0"
    },
    ...
  ]
}
```

**Get top 10 ranked alerts:**
```bash
curl http://localhost:8000/scores/rankings/top/10?network=ethereum
```

**Get batch metadata:**
```bash
curl http://localhost:8000/metadata/latest?network=ethereum
```

**Response:**
```json
{
  "processing_date": "2024-01-15",
  "network": "ethereum",
  "processed_at": "2024-01-15T14:30:00Z",
  "input_counts": {
    "alerts": 15234,
    "features": 15234,
    "clusters": 3456
  },
  "output_counts": {
    "alert_scores": 15234,
    "alert_rankings": 15234,
    "cluster_scores": 3456
  },
  "latencies_ms": {
    "alert_scoring": 2300,
    "alert_ranking": 1800,
    "cluster_scoring": 900,
    "total": 5200
  },
  "model_versions": {
    "alert_scorer": "1.0.0",
    "alert_ranker": "1.0.0",
    "cluster_scorer": "1.0.0"
  },
  "status": "completed"
}
```

## Daily Workflow

For production use, run these commands daily:

```bash
#!/bin/bash
# daily_processing.sh

DATE=$(date +%Y-%m-%d)
NETWORK="ethereum"

echo "Processing batch for $DATE..."

# Step 1: Download data
python scripts/download_from_sot.py \
    --processing-date $DATE \
    --network $NETWORK \
    --sot-host sot.clickhouse.example.com

# Step 2: Process batch
python scripts/process_batch.py \
    --processing-date $DATE \
    --network $NETWORK

echo "✓ Processing complete for $DATE"
```

Make it executable:
```bash
chmod +x daily_processing.sh
./daily_processing.sh
```

## Multiple Networks

Process multiple blockchain networks:

```bash
# Ethereum
python scripts/init_database.py --network ethereum
python scripts/download_from_sot.py --network ethereum --processing-date 2024-01-15
python scripts/process_batch.py --network ethereum --processing-date 2024-01-15

# Bitcoin
python scripts/init_database.py --network bitcoin
python scripts/download_from_sot.py --network bitcoin --processing-date 2024-01-15
python scripts/process_batch.py --network bitcoin --processing-date 2024-01-15

# Polygon
python scripts/init_database.py --network polygon
python scripts/download_from_sot.py --network polygon --processing-date 2024-01-15
python scripts/process_batch.py --network polygon --processing-date 2024-01-15
```

Query specific network:
```bash
curl http://localhost:8000/scores/alerts/latest?network=bitcoin
curl http://localhost:8000/scores/alerts/latest?network=polygon
```

## Troubleshooting

### ClickHouse Connection Issues

**Problem:** Cannot connect to ClickHouse

**Solution:**
```bash
# Check if ClickHouse is running
docker ps | grep clickhouse
# OR
sudo service clickhouse-server status

# Test connection
curl http://localhost:8123/
# Expected: "Ok."

# Check logs
docker logs clickhouse-server
# OR
tail -f /var/log/clickhouse-server/clickhouse-server.log
```

### Database Not Found

**Problem:** `Database alert_scoring doesn't exist`

**Solution:**
```bash
# Reinitialize database
python scripts/init_database.py --network ethereum

# Or manually create
echo "CREATE DATABASE IF NOT EXISTS alert_scoring" | curl -X POST http://localhost:8123/ --data-binary @-
```

### No Data Available

**Problem:** API returns 404 "No scores available"

**Solution:**
```bash
# Check if data was downloaded
echo "SELECT COUNT(*) FROM raw_alerts WHERE network = 'ethereum'" | \
  curl -X POST http://localhost:8123/?database=alert_scoring --data-binary @-

# Re-download and process
python scripts/download_from_sot.py --processing-date 2024-01-15 --network ethereum
python scripts/process_batch.py --processing-date 2024-01-15 --network ethereum
```

### Model Files Not Found

**Problem:** `FileNotFoundError: trained_models/alert_scorer_v1.0.0.txt`

**Solution:**
```bash
# Train models (if you have training data)
python scripts/train_models.py

# OR use placeholder models for testing
mkdir -p trained_models
echo "placeholder_model" > trained_models/alert_scorer_v1.0.0.txt
echo "placeholder_model" > trained_models/alert_ranker_v1.0.0.txt
echo "placeholder_model" > trained_models/cluster_scorer_v1.0.0.txt
```

### Port Already in Use

**Problem:** `Address already in use` when starting API

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000
# OR on Windows
netstat -ano | findstr :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m aml_miner.api.server --port 8001
```

### Memory Issues

**Problem:** Out of memory during processing

**Solution:**
```bash
# Process in smaller batches
python scripts/process_batch.py \
    --processing-date 2024-01-15 \
    --network ethereum \
    --batch-size 1000  # Process 1000 records at a time

# Or increase Docker memory limit
docker run -d --name clickhouse-server \
    -p 8123:8123 \
    --memory="4g" \
    clickhouse/clickhouse-server
```

## Next Steps

- **[API Reference](api_reference.md)** - Explore all API endpoints
- **[Training Guide](training_guide.md)** - Train custom models
- **[Customization Guide](customization.md)** - Extend functionality
- **[Architecture Docs](../docs/agent/2025-10-26/claude/)** - Understand system design

## Common Tasks

### View Available Dates

```bash
curl http://localhost:8000/dates/available?network=ethereum
```

### Get Historical Data

```bash
# Get scores for specific date
curl http://localhost:8000/scores/alerts/2024-01-10?network=ethereum

# Get metadata history (last 30 batches)
curl http://localhost:8000/metadata/history?network=ethereum&limit=30
```

### Delete Old Data

```bash
# Delete data older than 30 days
echo "ALTER TABLE alert_scores DELETE WHERE processing_date < today() - 30" | \
  curl -X POST http://localhost:8123/?database=alert_scoring --data-binary @-

echo "ALTER TABLE alert_rankings DELETE WHERE processing_date < today() - 30" | \
  curl -X POST http://localhost:8123/?database=alert_scoring --data-binary @-

echo "OPTIMIZE TABLE alert_scores FINAL" | \
  curl -X POST http://localhost:8123/?database=alert_scoring --data-binary @-
```

### Backup Database

```bash
# Export to SQL
clickhouse-client --query="SELECT * FROM alert_scoring.alert_scores" \
  --format=TabSeparated > alert_scores_backup.tsv

# Restore from SQL
cat alert_scores_backup.tsv | \
  clickhouse-client --query="INSERT INTO alert_scoring.alert_scores FORMAT TabSeparated"
```

## Support

For issues or questions:
- Check the [documentation](../README.md)
- Review [architecture docs](../docs/agent/2025-10-26/claude/)
- Open an issue on GitHub