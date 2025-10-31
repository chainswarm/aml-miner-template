# Miner-Validator Workflow and ML Service Architecture

**Date**: 2025-10-30  
**Purpose**: Complete architecture for miner daily operations, validator interaction, and real-world ML service deployment  
**Status**: Approved Design

---

## Executive Summary

This document defines the complete operational architecture for the risk-scoring system, covering:

1. **Daily Miner Workflow** - How miners ingest data, train models, and score alerts on a daily/weekly basis
2. **Miner-Validator Interaction** - Communication protocol and validation flow
3. **ML Service Deployment** - How the validator serves ensemble scores as a production ML service
4. **Implementation Roadmap** - What needs to be built to go live

### Key Architecture Principles

✅ **Decentralized ML** - Multiple miners compete on model quality  
✅ **Ensemble Robustness** - Validator aggregates scores for better accuracy  
✅ **Quality Assurance** - Multi-tier validation prevents gaming  
✅ **Simple Integration** - Consumers query one API endpoint  
✅ **Continuous Improvement** - Weekly model retraining

---

## System Overview

### Complete Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                   COMPLETE ECOSYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day 0 Morning:                                             │
│  ══════════════                                             │
│                                                              │
│  ┌──────────────┐                                           │
│  │     SOT      │  Publishes daily batch to S3             │
│  │ (Data Owner) │  ├─ alerts.parquet                        │
│  └──────┬───────┘  ├─ features.parquet                      │
│         │          ├─ clusters.parquet                      │
│         │          └─ money_flows.parquet                   │
│         │                                                    │
│         ├────────────────────────────────────┐             │
│         │                │                   │              │
│         ▼                ▼                   ▼              │
│  ┌─────────────┐  ┌─────────────┐    ┌─────────────┐      │
│  │  Miner 1    │  │  Miner 2    │    │  Miner N    │      │
│  │             │  │             │    │             │      │
│  │ 1. Ingest   │  │ 1. Ingest   │    │ 1. Ingest   │      │
│  │ 2. Score    │  │ 2. Score    │    │ 2. Score    │      │
│  │             │  │             │    │             │      │
│  │ Model A     │  │ Model B     │    │ Model X     │      │
│  │ (XGBoost)   │  │ (LightGBM)  │    │ (Neural)    │      │
│  └──────┬──────┘  └──────┬──────┘    └──────┬──────┘      │
│         │                │                   │              │
│         │ Individual     │ Individual        │ Individual   │
│         │ Scores         │ Scores            │ Scores       │
│         │                │                   │              │
│         └────────────────┼───────────────────┘             │
│                          ▼                                   │
│              ┌───────────────────────┐                      │
│              │      VALIDATOR        │                      │
│              │                       │                      │
│              │ 1. Collect all scores │                      │
│              │ 2. Validate quality   │                      │
│              │ 3. Compute ensemble   │                      │
│              │    (weighted average) │                      │
│              │ 4. Store ensemble     │                      │
│              └───────────┬───────────┘                      │
│                          │                                   │
│                          │ Ensemble Scores                  │
│                          │ (Canonical ML Results)           │
│                          │                                   │
│         ┌────────────────┼────────────────┐                │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     SOT     │  │  Alert App  │  │  Compliance │        │
│  │  (Consumer) │  │             │  │    Tools    │        │
│  │             │  │             │  │             │        │
│  │ Uses scores │  │ Uses scores │  │ Uses scores │        │
│  │ for storage │  │ for display │  │ for reports │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
SOT → Miners (parallel) → Validator → Ensemble → Consumers
```

**Key Points:**
- SOT publishes raw data to all miners
- Each miner scores independently using their own ML model
- Validator collects all scores, validates quality, creates ensemble
- Consumers query validator for canonical scores (not individual miners)

---

## Component 1: Miner Architecture

### Current Implementation Status

**What Exists:**
- ✅ [`scripts/ingest_data.py`](../../../scripts/ingest_data.py) - Data ingestion from S3 to ClickHouse
- ✅ [`scripts/train_model.py`](../../../scripts/train_model.py) - ML model training
- ✅ [`scripts/score_batch.py`](../../../scripts/score_batch.py) - Alert/cluster scoring
- ✅ [`packages/training/`](../../../packages/training/) - Feature engineering and training logic
- ✅ [`packages/scoring/`](../../../packages/scoring/) - Scoring pipeline
- ✅ ClickHouse tables for storage

**What's Missing:**
- ❌ FastAPI server to expose scores to validator

### Miner Internal Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  MINER ARCHITECTURE                        │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  Storage Layer (ClickHouse)                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Raw Tables:          Score Tables:               │    │
│  │ ├─ raw_alerts        ├─ alert_scores             │    │
│  │ ├─ raw_features      ├─ alert_rankings           │    │
│  │ ├─ raw_clusters      ├─ cluster_scores           │    │
│  │ ├─ raw_money_flows   └─ batch_metadata           │    │
│  │ └─ raw_address_labels                            │    │
│  └────────┬────────────────────┬──────────────────────┘    │
│           │                    │                           │
│           ▼                    ▼                           │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │ Training        │  │ Scoring         │                │
│  │ Pipeline        │  │ Pipeline        │                │
│  │                 │  │                 │                │
│  │ 1. Extract      │  │ 1. Extract      │                │
│  │ 2. Build        │  │ 2. Build        │                │
│  │    Features     │  │    Features     │                │
│  │ 3. Train Model  │  │ 3. Load Model   │                │
│  │ 4. Save Model   │  │ 4. Score        │                │
│  │                 │  │ 5. Write Results│                │
│  └─────────────────┘  └────────┬────────┘                │
│                                 │                          │
│  Models Storage                 ▼                          │
│  ┌──────────────────┐  ┌─────────────────┐               │
│  │ data/            │  │ FastAPI Server  │               │
│  │ trained_models/  │  │   [TO BUILD]    │               │
│  │ ├─ alert_scorer  │  │                 │               │
│  │ ├─ alert_ranker  │  │ Routes:         │               │
│  │ └─ cluster_scorer│  │ /scores/alerts  │               │
│  └──────────────────┘  │ /rankings       │               │
│                        │ /scores/clusters│               │
│                        │ /dates/available│               │
│                        └────────┬────────┘               │
│                                 │                          │
│                                 │ HTTP API                 │
│                                 ▼                          │
│                        ┌─────────────────┐                │
│                        │   Validator     │                │
│                        └─────────────────┘                │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

### Daily Miner Workflow

```
┌─────────────────────────────────────────────────────────┐
│         MINER DAILY WORKFLOW (ClickHouse)               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 00:00 UTC - SOT Publishes to S3                         │
│                                                          │
│ 00:30 UTC - Step 1: Data Ingestion                      │
│ $ python scripts/ingest_data.py \                       │
│     --network torus \                                    │
│     --processing-date 2025-10-30                         │
│ ├─ Downloads from S3                                    │
│ ├─ Writes to ClickHouse:                                │
│ │   ├─ raw_alerts                                       │
│ │   ├─ raw_features                                     │
│ │   ├─ raw_clusters                                     │
│ │   ├─ raw_money_flows                                  │
│ │   └─ raw_address_labels                               │
│ └─ Duration: ~5 minutes                                 │
│                                                          │
│ 01:00 UTC - Step 2: Risk Scoring                        │
│ $ python scripts/score_batch.py \                       │
│     --network torus \                                    │
│     --processing-date 2025-10-30 \                      │
│     --window-days 195                                    │
│ ├─ Reads from ClickHouse (raw tables)                   │
│ ├─ Builds features                                      │
│ ├─ Loads trained models                                 │
│ ├─ Scores alerts/clusters                               │
│ ├─ Writes to ClickHouse:                                │
│ │   ├─ alert_scores                                     │
│ │   ├─ alert_rankings                                   │
│ │   ├─ cluster_scores                                   │
│ │   └─ batch_metadata                                   │
│ └─ Duration: ~10 minutes                                │
│                                                          │
│ 01:30 UTC - FastAPI Server Running                      │
│ ├─ Serves scores from ClickHouse via API                │
│ ├─ GET /scores/alerts/2025-10-30                        │
│ └─ Validator can query for scores                       │
│                                                          │
│ Weekly (Sunday 03:00 UTC) - Model Retraining            │
│ $ python scripts/train_model.py \                       │
│     --network torus \                                    │
│     --start-date 2025-10-01 \                           │
│     --end-date 2025-10-30 \                             │
│     --window-days 195                                    │
│ ├─ Reads historical data from ClickHouse                │
│ ├─ Trains new models                                    │
│ ├─ Saves to data/trained_models/                        │
│ └─ Duration: ~60 minutes                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Miner API Specification

**Endpoints to Implement:**

```python
# Health & Info
GET  /health                          # Health check
GET  /version                         # API and model versions

# Date Discovery
GET  /dates/available                 # All processed dates (sorted desc)
GET  /dates/latest                    # Latest processed date

# Scores - Latest
GET  /scores/alerts/latest            # Latest alert scores
GET  /rankings/alerts/latest          # Latest alert rankings  
GET  /scores/clusters/latest          # Latest cluster scores

# Scores - Specific Date
GET  /scores/alerts/{processing_date}    # Alert scores for date
GET  /rankings/alerts/{processing_date}  # Alert rankings for date
GET  /scores/clusters/{processing_date}  # Cluster scores for date
```

**Response Format:**

```json
{
  "processing_date": "2025-10-30",
  "scores": [
    {
      "alert_id": "alert_001",
      "score": 0.87,
      "model_version": "alert_scorer_torus_v1.0.0_20251030_030000",
      "latency_ms": 12.5
    }
  ],
  "metadata": {
    "total_alerts": 1000,
    "model_version": "alert_scorer_torus_v1.0.0_20251030_030000",
    "processed_at": "2025-10-30T01:00:00Z"
  }
}
```

---

## Component 2: Validator Architecture

### Validator Responsibilities

1. **Collection** - Query all miners for scores
2. **Validation** - Quality checks (integrity, behavior, ground truth)
3. **Aggregation** - Compute weighted ensemble
4. **Service** - Expose ensemble API for consumers

### Validator Daily Workflow

```
┌──────────────────────────────────────────────────────────┐
│              VALIDATOR DAILY WORKFLOW                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ 01:00 UTC - Step 1: Collect Scores                       │
│ ├─ Query all registered miners                           │
│ │   GET miner_1:8000/scores/alerts/2025-10-30           │
│ │   GET miner_2:8000/scores/alerts/2025-10-30           │
│ │   ...                                                  │
│ │   GET miner_15:8000/scores/alerts/2025-10-30          │
│ ├─ Store individual submissions                          │
│ └─ Track miner availability                              │
│                                                           │
│ 01:15 UTC - Step 2: Immediate Validation                 │
│ ├─ Tier 1: Integrity (0-0.2 points)                     │
│ │   ├─ Schema validation                                 │
│ │   ├─ Completeness check                               │
│ │   ├─ Score range validation                           │
│ │   └─ Latency check                                     │
│ ├─ Tier 2: Behavior (0-0.3 points)                      │
│ │   ├─ Pattern trap detection                           │
│ │   ├─ Plagiarism check                                 │
│ │   └─ Gaming detection                                  │
│ └─ Compute immediate_score (0-0.5)                      │
│                                                           │
│ 01:30 UTC - Step 3: Ensemble Aggregation                 │
│ ├─ Weighted average calculation                          │
│ │   ensemble_score = Σ(weight[i] * score[i])           │
│ ├─ Confidence computation                                │
│ │   confidence = 1 - std(scores) / mean(scores)        │
│ ├─ Miner agreement calculation                          │
│ │   agreement = % within threshold of mean              │
│ └─ Store canonical ensemble results                      │
│                                                           │
│ 02:00 UTC - Step 4: Serve ML Service                     │
│ ├─ Expose ensemble API                                   │
│ │   GET /ensemble/scores/alerts/2025-10-30             │
│ └─ Consumers query for canonical scores                  │
│                                                           │
│ Day T+τ - Ground Truth Validation                        │
│ ├─ Load ground truth for historical date                 │
│ ├─ Re-evaluate each miner's historical scores           │
│ ├─ Compute ground_truth_score (0-0.5)                   │
│ ├─ final_score = immediate + ground_truth (1.0 total)   │
│ └─ Update permanent miner weights                        │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Ensemble Calculation Logic

```
┌───────────────────────────────────────────────────────────┐
│           VALIDATOR ENSEMBLE CALCULATION                   │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  Input: Individual Miner Scores                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ alert_001:                                          │  │
│  │   miner_1: 0.85 (weight: 0.12)                     │  │
│  │   miner_2: 0.78 (weight: 0.10)                     │  │
│  │   miner_3: 0.91 (weight: 0.15) ← highest quality   │  │
│  │   miner_4: 0.82 (weight: 0.08)                     │  │
│  │   ...                                               │  │
│  │   miner_15: 0.88 (weight: 0.11)                    │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Ensemble Calculation                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ ensemble_score = Σ(weight[i] * score[i])          │  │
│  │                = 0.12*0.85 + 0.10*0.78 + ...      │  │
│  │                = 0.8523                             │  │
│  │                                                     │  │
│  │ confidence = 1 - std(scores) / mean(scores)       │  │
│  │            = 1 - 0.048 / 0.848                     │  │
│  │            = 0.943                                  │  │
│  │                                                     │  │
│  │ miner_agreement = % of miners within 0.1 of mean  │  │
│  │                 = 13/15 = 0.867                    │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Output: Canonical Score                                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │ {                                                   │  │
│  │   "alert_id": "alert_001",                         │  │
│  │   "ensemble_score": 0.8523,                        │  │
│  │   "confidence": 0.943,                             │  │
│  │   "miner_count": 15,                               │  │
│  │   "miner_agreement": 0.867,                        │  │
│  │   "top_contributors": [...]                        │  │
│  │ }                                                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

### Validator ML Service API

**Endpoints:**

```python
# Ensemble Scores (Canonical)
GET  /ensemble/scores/alerts/{processing_date}     # Aggregated scores
GET  /ensemble/rankings/alerts/{processing_date}   # Aggregated rankings
GET  /ensemble/scores/clusters/{processing_date}   # Cluster scores
GET  /ensemble/scores/alerts/latest                # Latest scores
GET  /ensemble/rankings/alerts/latest              # Latest rankings

# Miner Submissions (Transparency)
GET  /submissions/{processing_date}                # All miner submissions
GET  /submissions/{processing_date}/{miner_id}     # Specific miner

# Validation Results
GET  /validation/{processing_date}                 # All validation results
GET  /validation/{processing_date}/{miner_id}      # Specific miner validation

# Miner Performance
GET  /miners/rankings                              # Current miner leaderboard
GET  /miners/{miner_id}/performance                # Historical metrics
```

**Ensemble Response Format:**

```json
{
  "processing_date": "2025-10-30",
  "aggregation_method": "weighted_ensemble",
  "total_miners": 15,
  "ensemble_scores": [
    {
      "alert_id": "alert_001",
      "ensemble_score": 0.8523,
      "confidence": 0.943,
      "miner_count": 15,
      "miner_agreement": 0.867,
      "score_std": 0.048,
      "top_contributors": [
        {"miner_id": "miner_3", "weight": 0.15, "score": 0.91},
        {"miner_id": "miner_1", "weight": 0.12, "score": 0.85}
      ]
    }
  ],
  "metadata": {
    "aggregated_at": "2025-10-30T01:30:00Z",
    "immediate_validation_complete": true,
    "ground_truth_validation_pending": true,
    "expected_ground_truth_date": "2025-11-06"
  }
}
```

---

## Component 3: ML Service Deployment

### Consumer Integration

**SOT System Integration:**

```python
# SOT queries validator for canonical scores
import requests

def get_daily_risk_scores(processing_date: str):
    """Query validator's ensemble API"""
    response = requests.get(
        f"http://validator:9000/ensemble/scores/alerts/{processing_date}"
    )
    ensemble_data = response.json()
    
    # Store in SOT database
    for score in ensemble_data['ensemble_scores']:
        db.insert(
            'risk_scores',
            alert_id=score['alert_id'],
            risk_score=score['ensemble_score'],
            confidence=score['confidence'],
            processing_date=processing_date
        )
```

**Alert App Integration:**

```python
# Alert app displays top-risk alerts
def get_top_risk_alerts(date: str, limit: int = 100):
    """Get highest-risk alerts from validator"""
    response = requests.get(
        f"http://validator:9000/ensemble/rankings/alerts/{date}"
    )
    rankings = response.json()
    
    # Display to analysts
    return rankings['ensemble_scores'][:limit]
```

### Complete Daily Flow

```
TIME: 00:00 UTC
┌───────────────────────────────────────────────────────────┐
│                   SOT PUBLISHES                            │
│  s3://sot-data/torus/2025-10-30/                          │
│  ├─ alerts.parquet (10,000 alerts)                        │
│  ├─ features.parquet (50,000 addresses)                   │
│  ├─ clusters.parquet (500 clusters)                       │
│  └─ money_flows.parquet (1M transactions)                 │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  │ Download (parallel)
                  ├──────────┬──────────┬──────────┐
                  ▼          ▼          ▼          ▼
TIME: 00:30 UTC
┌──────────┐  ┌──────────┐  ┌──────────┐    ┌──────────┐
│ Miner 1  │  │ Miner 2  │  │ Miner 3  │ ...│ Miner 15 │
│          │  │          │  │          │    │          │
│ Ingest   │  │ Ingest   │  │ Ingest   │    │ Ingest   │
│ ↓        │  │ ↓        │  │ ↓        │    │ ↓        │
│ Score    │  │ Score    │  │ Score    │    │ Score    │
│ ↓        │  │ ↓        │  │ ↓        │    │ ↓        │
│ Store    │  │ Store    │  │ Store    │    │ Store    │
│          │  │          │  │          │    │          │
│ API:8000 │  │ API:8000 │  │ API:8000 │    │ API:8000 │
└────┬─────┘  └────┬─────┘  └────┬─────┘    └────┬─────┘
     │             │             │               │
     │ Individual  │ Individual  │ Individual    │ Individual
     │ Scores      │ Scores      │ Scores        │ Scores
     │             │             │               │
TIME: 01:00 UTC    │             │               │
     └─────────────┼─────────────┼───────────────┘
                   ▼
            ┌──────────────┐
            │  VALIDATOR   │
            │              │
            │ 1. Collect   │──┐
            │ 2. Validate  │  │ Immediate Validation
            │ 3. Aggregate │◄─┘ (Tiers 1 & 2)
            │              │
            │ Ensemble API │
            │   Port:9000  │
            └──────┬───────┘
                   │
TIME: 02:00 UTC    │ Canonical Scores
                   │
     ┌─────────────┼─────────────┬───────────────┐
     ▼             ▼             ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   SOT    │  │ Alert    │  │Compliance│  │ Other    │
│  System  │  │   App    │  │   Tools  │  │ Consumers│
│          │  │          │  │          │  │          │
│ Stores   │  │ Displays │  │ Reports  │  │ ...      │
│ canonical│  │ top risks│  │ analysis │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## Component 4: Model Training Strategy

### Weekly Training Schedule

```
┌───────────────────────────────────────────────────────────┐
│              WEEKLY TRAINING SCHEDULE                      │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  Sunday 03:00 UTC - All Miners Train                      │
│                                                            │
│  Training Data Window:                                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Date Range: Last 30 days                           │  │
│  │ Start: 2025-10-01                                  │  │
│  │ End:   2025-10-30                                  │  │
│  │                                                     │  │
│  │ Window: 195 days for feature calculation           │  │
│  │ ├─ Lookback from each processing_date              │  │
│  │ └─ Captures long-term patterns                     │  │
│  │                                                     │  │
│  │ Ground Truth: address_labels table                 │  │
│  │ ├─ High/Critical risk → label = 1                  │  │
│  │ └─ Low/Medium risk → label = 0                     │  │
│  │                                                     │  │
│  │ Features: 56 features per alert                    │  │
│  │ ├─ Alert-level (8)                                 │  │
│  │ ├─ Address-level (23)                              │  │
│  │ ├─ Temporal (5)                                    │  │
│  │ ├─ Statistical (7)                                 │  │
│  │ ├─ Cluster (3)                                     │  │
│  │ ├─ Network (8)                                     │  │
│  │ └─ Label (2)                                       │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Training Process:                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. Feature Extraction (5-10 min)                   │  │
│  │ 2. Feature Engineering (2-5 min)                   │  │
│  │ 3. XGBoost Training (30-40 min)                    │  │
│  │    ├─ alert_scorer                                 │  │
│  │    ├─ alert_ranker                                 │  │
│  │    └─ cluster_scorer                               │  │
│  │ 4. Model Validation (5 min)                        │  │
│  │ 5. Save Models (1 min)                             │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Output: New Models                                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │ data/trained_models/torus/                         │  │
│  │ ├─ alert_scorer_torus_v1.0.0_20251030_030000.txt  │  │
│  │ ├─ alert_ranker_torus_v1.0.0_20251030_030000.txt  │  │
│  │ └─ cluster_scorer_torus_v1.0.0_20251030_030000.txt│  │
│  │                                                     │  │
│  │ Used starting Monday 00:00 UTC                     │  │
│  └────────────────────────────────────────────────────┘  │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

### Training Command

```bash
# Weekly training (Sunday 03:00 UTC)
python scripts/train_model.py \
    --network torus \
    --start-date 2025-10-01 \
    --end-date 2025-10-30 \
    --window-days 195 \
    --model-types alert_scorer alert_ranker cluster_scorer
```

---

## Implementation Roadmap

### Phase 1: Miner API (Week 1)

**Objective:** Enable miners to serve scores to validator

**Tasks:**
1. Create `packages/api/` directory structure
2. Implement FastAPI server with ClickHouse integration
3. Implement required endpoints
4. Add health checks and monitoring
5. Deploy and test

**Deliverables:**
```
packages/api/
├─ __init__.py
├─ server.py          # FastAPI application
├─ routes.py          # API endpoints
├─ models.py          # Pydantic response schemas
└─ database.py        # ClickHouse queries
```

**Endpoints:**
- `GET /health`
- `GET /scores/alerts/{processing_date}`
- `GET /rankings/alerts/{processing_date}`
- `GET /scores/clusters/{processing_date}`
- `GET /dates/available`
- `GET /dates/latest`

### Phase 2: Validator Service (Week 2-3)

**Objective:** Build validator that collects, validates, and aggregates miner scores

**Tasks:**
1. Create validator repository/service
2. Implement miner collection module
3. Implement validation modules (Tier 1, 2, 3)
4. Implement ensemble aggregation
5. Implement ML service API
6. Deploy and test

**Deliverables:**
```
validator/
├─ collector.py       # Query all miners
├─ validator.py       # Multi-tier validation
├─ aggregator.py      # Ensemble calculation
├─ storage.py         # Database for submissions
├─ api/
│   ├─ server.py      # ML Service API
│   └─ routes.py      # Ensemble endpoints
└─ config.py          # Configuration
```

### Phase 3: Integration Testing (Week 4)

**Objective:** Validate end-to-end system

**Tasks:**
1. Deploy 3-5 test miners
2. Deploy validator
3. Test daily workflow
4. Validate ensemble quality
5. Performance benchmarking
6. Documentation

**Test Scenarios:**
- Single miner failure (ensemble still works)
- Quality degradation detection
- Ground truth validation
- API performance under load

### Phase 4: Production Deployment (Week 5+)

**Objective:** Go live with Bittensor subnet

**Tasks:**
1. Deploy to Bittensor testnet
2. Monitor miner performance
3. Tune ensemble weights
4. Iterate based on feedback
5. Graduate to mainnet

---

## Validation Framework

### Multi-Tier Validation

**Tier 1: Integrity Validation (0-0.2 points)**
- Schema compliance
- Completeness check
- Score range validation [0, 1]
- Latency requirements
- Determinism check

**Tier 2: Behavior Validation (0-0.3 points)**
- Pattern trap detection
- Plagiarism check
- Gaming detection
- Consistency validation

**Tier 3: Ground Truth Validation (0-0.5 points)**
- AUC-ROC
- AUC-PR
- Precision@K
- Brier score
- Calibration

**Total Score:** immediate_score (0.5) + ground_truth_score (0.5) = 1.0

---

## Cron Schedule Summary

### Miner Cron Jobs

```bash
# Daily data ingestion
30 0 * * * python scripts/ingest_data.py --network torus --processing-date $(date +%Y-%m-%d)

# Daily scoring
0 1 * * * python scripts/score_batch.py --network torus --processing-date $(date +%Y-%m-%d) --window-days 195

# Weekly training (Sunday 03:00 UTC)
0 3 * * 0 python scripts/train_model.py --network torus --start-date $(date -d '30 days ago' +%Y-%m-%d) --end-date $(date +%Y-%m-%d) --window-days 195
```

### Validator Cron Jobs

```bash
# Daily collection and aggregation
0 1 * * * python validator/main.py collect-and-aggregate --processing-date $(date +%Y-%m-%d)

# Weekly ground truth validation (re-validate past week)
0 2 * * 0 python validator/main.py ground-truth-validation --start-date $(date -d '7 days ago' +%Y-%m-%d) --end-date $(date -d '1 day ago' +%Y-%m-%d)
```

---

## Benefits of This Architecture

### For SOT/Consumers
✅ **Single API endpoint** - Query one validator instead of many miners  
✅ **Robust scores** - Ensemble is more accurate than any single miner  
✅ **Quality assurance** - Validator ensures miners maintain standards  
✅ **Transparency** - Can see individual miner contributions  
✅ **Reliability** - System works even if some miners fail  

### For Miners
✅ **Focus on ML** - No need to manage infrastructure for serving ensemble  
✅ **Fair competition** - Quality-based weights, not just availability  
✅ **Continuous improvement** - Weekly retraining keeps models fresh  
✅ **Decentralized** - No single point of failure  

### For the Ecosystem
✅ **Innovation** - Miners compete on model quality  
✅ **Scalability** - Easy to add more miners  
✅ **Maintainability** - Clear separation of concerns  
✅ **Observability** - All components expose metrics  

---

## Next Steps

1. **Immediate (Week 1):** Build miner FastAPI server
2. **Short-term (Week 2-3):** Build validator service
3. **Medium-term (Week 4):** Integration testing
4. **Long-term (Week 5+):** Production deployment to Bittensor

---

## Appendix: Reference Commands

### Miner Operations

```bash
# Data ingestion
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-30

# Scoring
python scripts/score_batch.py \
    --network torus \
    --processing-date 2025-10-30 \
    --window-days 195

# Training
python scripts/train_model.py \
    --network torus \
    --start-date 2025-10-01 \
    --end-date 2025-10-30 \
    --window-days 195

# Start API server
python -m packages.api.server --port 8000
```

### Validator Operations

```bash
# Collect scores from all miners
python validator/collector.py --processing-date 2025-10-30

# Run validation
python validator/validator.py --processing-date 2025-10-30

# Compute ensemble
python validator/aggregator.py --processing-date 2025-10-30

# Start ML service API
python validator/api/server.py --port 9000
```

### Consumer Queries

```bash
# Get ensemble scores
curl http://validator:9000/ensemble/scores/alerts/2025-10-30

# Get latest rankings
curl http://validator:9000/ensemble/rankings/alerts/latest

# Get miner performance
curl http://validator:9000/miners/rankings
```

---

**End of Architecture Document**