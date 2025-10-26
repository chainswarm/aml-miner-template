# Corrected Workflow Architecture
**Date**: 2025-10-26  
**Purpose**: Document the correct workflow for aml-miner-template

---

## Correct Workflow

### Overview

The miner template operates as a **batch processing system** with an API that serves pre-computed results:

```
┌─────────────────────────────────────────────────────────────┐
│                    SOT (Source of Truth)                     │
│                    S3 Bucket with Parquet Files              │
│  - alerts.parquet                                            │
│  - features.parquet                                          │
│  - clusters.parquet                                          │
│  - money_flows.parquet                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1. Download Script
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Miner Template (Local)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ input/                                                 │  │
│  │  └─ 2025-10-26/                                        │  │
│  │      ├─ alerts.parquet    (downloaded from S3)         │  │
│  │      ├─ features.parquet                               │  │
│  │      ├─ clusters.parquet                               │  │
│  │      └─ money_flows.parquet                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 2. Processing Script
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                       │
│  1. Load Parquet files from input/                          │
│  2. Build features                                           │
│  3. Score alerts → output/alerts_scored.parquet             │
│  4. Rank alerts → output/alerts_ranked.parquet              │
│  5. Score clusters → output/clusters_scored.parquet         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 3. Results saved to output/
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                            │
│  - Reads from output/ directory                             │
│  - Serves pre-computed results                              │
│  - GET /alerts/{processing_date}/scores                     │
│  - GET /alerts/{processing_date}/rankings                   │
│  - GET /clusters/{processing_date}/scores                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 4. HTTP calls
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Bittensor Subnet Miner (Proxy)                  │
│  - Calls template API to get scores                         │
│  - Forwards to validator                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 5. Bittensor protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Validator                                 │
│  - Also reads from SOT directly                             │
│  - Validates miner submissions                              │
│  - Compares to ground truth                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Download Script
**File**: `scripts/download_batch.py`
- Downloads Parquet files from S3
- Parameters: `--network`, `--processing-date`, `--days`
- Saves to: `input/{processing_date}/`
- Files downloaded:
  - alerts.parquet
  - features.parquet
  - clusters.parquet
  - money_flows.parquet (optional)

### 2. Processing Script
**File**: `scripts/process_batch.py`
- Reads Parquet files from `input/{processing_date}/`
- Loads all models
- Processes data:
  - Score all alerts → `output/{processing_date}/alerts_scored.parquet`
  - Rank all alerts → `output/{processing_date}/alerts_ranked.parquet`
  - Score all clusters → `output/{processing_date}/clusters_scored.parquet`
- Saves metadata: `output/{processing_date}/metadata.json`

### 3. FastAPI Server
**File**: `aml_miner/api/server.py`
- Serves pre-computed results from `output/` directory
- Endpoints:
  - `GET /scores/alerts/{processing_date}` - Return scored alerts
  - `GET /rankings/alerts/{processing_date}` - Return ranked alerts
  - `GET /scores/clusters/{processing_date}` - Return scored clusters
  - `GET /health` - Health check
  - `GET /version` - Version info

### 4. Directory Structure

```
aml-miner-template/
├── input/                  # Downloaded Parquet files
│   └── 2025-10-26/
│       ├── alerts.parquet
│       ├── features.parquet
│       ├── clusters.parquet
│       └── money_flows.parquet
│
├── output/                 # Processed results
│   └── 2025-10-26/
│       ├── alerts_scored.parquet
│       ├── alerts_ranked.parquet
│       ├── clusters_scored.parquet
│       └── metadata.json
│
├── trained_models/         # ML models
│   ├── alert_scorer_v1.0.0.txt
│   ├── alert_ranker_v1.0.0.txt
│   └── cluster_scorer_v1.0.0.txt
│
└── aml_miner/             # Application code
    └── api/               # FastAPI server
```

---

## Workflow Steps

### Step 1: Download Data (Daily)
```bash
# Download today's batch from SOT S3
python scripts/download_batch.py \
  --network bitcoin \
  --processing-date 2025-10-26 \
  --days 7 \
  --output-dir ./input/2025-10-26
```

### Step 2: Process Batch (Daily)
```bash
# Score all alerts, rank them, score clusters
python scripts/process_batch.py \
  --input-dir ./input/2025-10-26 \
  --output-dir ./output/2025-10-26
```

This creates:
- `output/2025-10-26/alerts_scored.parquet`
- `output/2025-10-26/alerts_ranked.parquet`
- `output/2025-10-26/clusters_scored.parquet`

### Step 3: Start API Server
```bash
# API serves results from output/ directory
python -m aml_miner.api.server
```

### Step 4: Validator Queries Miner
```bash
# Bittensor subnet miner proxies this to template API
curl http://localhost:8000/scores/alerts/2025-10-26
```

Returns pre-computed scores from `output/2025-10-26/alerts_scored.parquet`

---

## Key Differences from Current Implementation

### Current (Wrong):
- API accepts batch data in POST body
- API scores in real-time when called
- No separation of processing and serving

### Correct:
- Script downloads data from S3 to input/
- Script processes and saves results to output/
- API reads from output/ and serves results
- API is a simple file server for pre-computed results

---

## Changes Required

### 1. Download Script
- Change from placeholder to real S3 downloader
- Parameters: network, processing_date, days
- Save to input/{processing_date}/

### 2. NEW: Processing Script
- `scripts/process_batch.py` (doesn't exist yet)
- Load from input/
- Score all data
- Save to output/

### 3. API Routes (Major Change)
- Change from POST endpoints accepting data
- To GET endpoints serving files
- Read Parquet from output/
- Return as JSON

### 4. API Schemas (Change)
- Remove BatchData input schema
- Add response-only schemas
- Keep ScoreResponse, RankResponse, etc.

---

## Summary

The template should work as a **batch processing + file-serving** system:

1. **Download** (script) - Get Parquet from S3 → input/
2. **Process** (script) - Score data → output/
3. **Serve** (API) - Read output/ → return JSON
4. **Proxy** (subnet miner) - Call API → forward to validator

NOT a real-time inference API.