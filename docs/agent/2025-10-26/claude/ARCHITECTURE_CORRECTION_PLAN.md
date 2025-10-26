# Architecture Correction Plan
**Date**: 2025-10-26  
**Purpose**: Plan to correct aml-miner-template to match proper workflow

---

## Problem Statement

Current implementation has the API accepting batch data via POST and scoring in real-time. This is **incorrect**.

## Correct Architecture

### Workflow Components

1. **Download Script** - Get Parquet files from S3
2. **Processing Scripts** - Batch score/rank locally
3. **API Server** - Serve pre-computed results
4. **Bittensor Miner** - Proxy API to validator
5. **Validator** - Reads SOT and validates miner scores

### Data Flow

```
SOT (S3) → Download Script → input/ → Processing Scripts → output/ → API → Subnet Miner → Validator
                                                                              ↑
                                                                              └─ Also reads SOT directly
```

---

## Changes Required

### 1. Download Script (Existing - Needs Enhancement)
**File**: `scripts/download_batch.py` (rename from .sh)

**Current**: Placeholder bash script  
**Required**: Real S3 downloader

**Function**:
```python
def download_batch(network: str, processing_date: str, days: int, output_dir: Path):
    # Download from S3:
    # - alerts.parquet
    # - features.parquet  
    # - clusters.parquet
    # - money_flows.parquet (optional)
    # Save to: output_dir/{processing_date}/
```

**CLI**:
```bash
python scripts/download_batch.py \
  --network bitcoin \
  --processing-date 2025-10-26 \
  --days 7 \
  --output-dir ./input
```

### 2. NEW: Process Batch Script
**File**: `scripts/process_batch.py` (NEW FILE)

**Function**:
```python
def process_batch(input_dir: Path, output_dir: Path, processing_date: str):
    # 1. Load Parquet files from input/{processing_date}/
    loader = BatchDataLoader()
    batch = loader.load_batch(input_dir / processing_date)
    
    # 2. Load models
    scorer = AlertScorerModel()
    scorer.load_model("trained_models/alert_scorer_v1.0.0.txt")
    ranker = AlertRankerModel()
    ranker.load_model("trained_models/alert_ranker_v1.0.0.txt")
    cluster = ClusterScorerModel()
    cluster.load_model("trained_models/cluster_scorer_v1.0.0.txt")
    
    # 3. Score alerts
    X_alerts = scorer.prepare_features(batch['alerts'], batch['features'], batch['clusters'])
    alert_scores = scorer.predict(X_alerts)
    
    # Save: output/{processing_date}/alerts_scored.parquet
    scored_df = batch['alerts'].copy()
    scored_df['score'] = alert_scores
    scored_df['model_version'] = scorer.model_version
    scored_df.to_parquet(output_dir / processing_date / "alerts_scored.parquet")
    
    # 4. Rank alerts
    rankings = ranker.rank_alerts(X_alerts, batch['alerts']['alert_id'].tolist())
    
    # Save: output/{processing_date}/alerts_ranked.parquet
    rankings.to_parquet(output_dir / processing_date / "alerts_ranked.parquet")
    
    # 5. Score clusters
    X_clusters = cluster.prepare_features(batch['clusters'], batch['features'], batch.get('money_flows'))
    cluster_scores = cluster.predict(X_clusters)
    
    # Save: output/{processing_date}/clusters_scored.parquet
    scored_clusters = batch['clusters'].copy()
    scored_clusters['score'] = cluster_scores
    scored_clusters['model_version'] = cluster.model_version
    scored_clusters.to_parquet(output_dir / processing_date / "clusters_scored.parquet")
    
    # 6. Save metadata
    metadata = {
        "processing_date": processing_date,
        "alert_count": len(alert_scores),
        "cluster_count": len(cluster_scores),
        "model_versions": {
            "scorer": scorer.model_version,
            "ranker": ranker.model_version,
            "cluster": cluster.model_version
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(output_dir / processing_date / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

**CLI**:
```bash
python scripts/process_batch.py \
  --input-dir ./input \
  --output-dir ./output \
  --processing-date 2025-10-26
```

### 3. API Routes (MAJOR CHANGE)
**File**: `aml_miner/api/routes.py`

**Current**: POST endpoints that accept batch data and score in real-time  
**Required**: GET endpoints that serve pre-computed results

**New Endpoints**:

```python
@router.get("/scores/alerts/{processing_date}")
async def get_alert_scores(processing_date: str):
    # Read: output/{processing_date}/alerts_scored.parquet
    # Return as JSON
    
@router.get("/rankings/alerts/{processing_date}")
async def get_alert_rankings(processing_date: str):
    # Read: output/{processing_date}/alerts_ranked.parquet
    # Return as JSON
    
@router.get("/scores/clusters/{processing_date}")
async def get_cluster_scores(processing_date: str):
    # Read: output/{processing_date}/clusters_scored.parquet
    # Return as JSON

@router.get("/metadata/{processing_date}")
async def get_metadata(processing_date: str):
    # Read: output/{processing_date}/metadata.json
    # Return metadata
```

### 4. API Schemas (CHANGE)
**File**: `aml_miner/api/schemas.py`

**Remove**: BatchData input schema (no longer needed)  
**Keep**: Response schemas (ScoreResponse, RankResponse, etc.)  
**Add**: Metadata schema

### 5. Settings (ADD)
**File**: `aml_miner/config/settings.py`

**Add fields**:
```python
INPUT_DIR: Path = BASE_DIR / "input"
OUTPUT_DIR: Path = BASE_DIR / "output"
```

---

## Directory Structure Changes

### Add Directories:
```
aml-miner-template/
├── input/                  # Downloaded Parquet files from SOT
│   └── {processing_date}/
│       ├── alerts.parquet
│       ├── features.parquet
│       ├── clusters.parquet
│       └── money_flows.parquet
│
├── output/                 # Processed results
│   └── {processing_date}/
│       ├── alerts_scored.parquet
│       ├── alerts_ranked.parquet
│       ├── clusters_scored.parquet
│       └── metadata.json
│
└── ...
```

---

## Files to Create

1. `scripts/download_batch.py` - S3 downloader (replace .sh)
2. `scripts/process_batch.py` - Batch processor (NEW)

## Files to Modify

1. `aml_miner/api/routes.py` - Change POST to GET, read from output/
2. `aml_miner/api/schemas.py` - Remove BatchData input
3. `aml_miner/api/server.py` - Update startup (no model loading needed in API)
4. `aml_miner/config/settings.py` - Add INPUT_DIR, OUTPUT_DIR
5. `README.md` - Update workflow documentation
6. `docs/api_reference.md` - Update endpoints
7. `docs/quickstart.md` - Update usage flow

## Files to Keep As-Is

- All model files (models/)
- Feature engineering (features/)
- Training scripts (training/)
- Data loader (utils/data_loader.py)
- Configuration (config/)

---

## Implementation Priority

### Priority 1: Core Workflow
1. Create `scripts/process_batch.py`
2. Update `aml_miner/api/routes.py` (change to GET endpoints)
3. Update `aml_miner/config/settings.py` (add dirs)

### Priority 2: S3 Integration
4. Update `scripts/download_batch.py` (real S3 downloader)

### Priority 3: Documentation
5. Update `README.md`
6. Update `docs/api_reference.md`
7. Update `docs/quickstart.md`

---

## Testing Workflow

```bash
# 1. Manually place test Parquet files in input/
mkdir -p input/2025-10-26
# Copy test files to input/2025-10-26/

# 2. Process batch
python scripts/process_batch.py \
  --input-dir ./input \
  --output-dir ./output \
  --processing-date 2025-10-26

# 3. Start API
python -m aml_miner.api.server

# 4. Query API
curl http://localhost:8000/scores/alerts/2025-10-26

# Should return pre-computed scores
```

---

## Summary

The miner template is a **batch processing system** that:
1. Downloads data from SOT (S3)
2. Processes locally (score + rank)
3. Saves results to disk
4. Serves results via simple file-serving API

The API does NOT do real-time inference. It's a simple **result-serving layer**.