# Validation Flow and API Design
## Immediate vs Delayed Validation Timeline

**Date**: 2025-10-26  
**Purpose**: Clarify validation timeline and API endpoint design for Day 0 and T+τ validation

---

## The Validation Timeline Problem

### Your Concerns (Correctly Identified):

1. **Day 0 Validation**: How does validator get today's scores immediately?
2. **Day T+τ Validation**: How does validator re-evaluate historical scores after τ days?
3. **API Design**: Why do endpoints have `{processing_date}` parameter?
4. **Alert App Consumption**: Should alert app get miner scores or validator's ensemble?

---

## Complete Validation Flow

### Day 0: Immediate Validation

```
Morning (00:00 UTC):
└─ SOT publishes today's batch
   └─ processing_date = 2025-10-26
   └─ Files: alerts.parquet, features.parquet, clusters.parquet

Miners (00:30 UTC):
└─ Download batch for 2025-10-26
└─ Process: scripts/process_batch.py --processing-date 2025-10-26
└─ Output: output/2025-10-26/alert_scores.parquet
└─ API serves via: GET /scores/alerts/2025-10-26

Validator (01:00 UTC):
└─ Query all miners: GET /scores/alerts/2025-10-26
└─ Collect responses from N miners
└─ Run immediate validation:
   ├─ IntegrityValidator (schema, completeness, latency)
   ├─ BehaviorValidator (pattern traps, plagiarism)
   └─ Compute immediate_score (0-0.5 range)
└─ Aggregate all miners' scores → Ensemble
└─ Publish ensemble to SOT or validator API
└─ Set temporary miner weights based on immediate_score

Alert App (01:30 UTC):
└─ Query validator's ensemble scores for 2025-10-26
└─ Display top-risk alerts to analysts
```

### Day 0+τ: Delayed Validation (Ground Truth)

```
T+τ days later (e.g., T+7 or T+30):
└─ Ground truth for 2025-10-26 becomes available
   └─ SAR filings, exchange labels, etc.
   └─ SOT publishes: ground_truth_2025-10-26.parquet

Validator:
└─ Load ground truth for 2025-10-26
└─ Load each miner's historical scores for 2025-10-26
   └─ GET /scores/alerts/2025-10-26 (still available)
└─ Run ground truth validation:
   ├─ GroundTruthValidator (AUC-ROC, AUC-PR, F1)
   └─ Compute ground_truth_score (0-0.5 range)
└─ Update miner weights:
   └─ final_score = immediate_score (0.5) + ground_truth_score (0.5) = 1.0
└─ Set updated weights on Bittensor chain
```

---

## API Design: Three Options Analyzed

### Option A: Date-Based Only (Current Implementation)

```python
GET /scores/alerts/{processing_date}
GET /rankings/alerts/{processing_date}
GET /scores/clusters/{processing_date}
```

**Day 0 Usage:**
```bash
# Validator knows today is 2025-10-26
curl http://miner:8000/scores/alerts/2025-10-26
```

**Day T+τ Usage:**
```bash
# Validator requests historical batch
curl http://miner:8000/scores/alerts/2025-10-26
```

**Pros:**
- ✅ Explicit - validator specifies exact batch
- ✅ Works for both Day 0 and T+τ
- ✅ Supports historical queries

**Cons:**
- ❌ Validator must know what date to request on Day 0
- ❌ What if miner hasn't processed today's batch yet?

---

### Option B: Latest-Based Only

```python
GET /scores/alerts/latest
GET /rankings/alerts/latest
GET /scores/clusters/latest
```

**Day 0 Usage:**
```bash
# Validator gets whatever miner has processed most recently
curl http://miner:8000/scores/alerts/latest
```

**Day T+τ Usage:**
```bash
# Problem: Can't request specific historical date
# Would need separate endpoint
```

**Pros:**
- ✅ Simple - validator doesn't need to know date
- ✅ Always gets most recent available

**Cons:**
- ❌ Can't request specific historical batches for T+τ
- ❌ What if different miners processed different dates?

---

### Option C: Hybrid (Recommended)

```python
# For Day 0: Get latest available batch
GET /scores/alerts/latest
GET /rankings/alerts/latest
GET /scores/clusters/latest

# For Day 0 or T+τ: Get specific batch
GET /scores/alerts/{processing_date}
GET /rankings/alerts/{processing_date}
GET /scores/clusters/{processing_date}

# Helper endpoints
GET /dates/available           # List all processed dates
GET /dates/latest              # Get latest processed date
```

**Day 0 Usage:**
```bash
# Option 1: Get latest (miner decides which date)
curl http://miner:8000/scores/alerts/latest
# Returns: {processing_date: "2025-10-26", scores: [...]}

# Option 2: Request specific date (validator decides)
curl http://miner:8000/dates/latest  # Returns: "2025-10-26"
curl http://miner:8000/scores/alerts/2025-10-26
```

**Day T+τ Usage:**
```bash
# Request historical batch
curl http://miner:8000/scores/alerts/2025-10-26
```

**Pros:**
- ✅ Flexible - supports both latest and historical
- ✅ Validator can discover what dates are available
- ✅ Works for both Day 0 and T+τ

**Cons:**
- ❌ More endpoints to implement

---

## Recommended API Design (Option C Enhanced)

### Miner API Endpoints

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

# Metadata
GET  /metadata/{processing_date}      # Processing metadata
GET  /metadata/latest                 # Latest processing metadata

# Utility
POST /refresh                         # Refresh DuckDB views
```

### Response Format

All score endpoints return:
```json
{
  "processing_date": "2025-10-26",
  "scores": [
    {
      "alert_id": "alert_001",
      "score": 0.87,
      "model_version": "1.0.0",
      "latency_ms": 12.5,
      "explain_json": "{...}"
    }
  ],
  "metadata": {
    "total_alerts": 1000,
    "model_version": "1.0.0",
    "processed_at": "2025-10-26T01:00:00Z"
  }
}
```

---

## Validator's Role

### What Validator Does

**Day 0 (Immediate Validation):**
```python
# 1. Query all miners for latest batch
for miner in active_miners:
    response = miner.get('/scores/alerts/latest')
    
    # 2. Validate each submission
    integrity_result = IntegrityValidator.validate(response.scores, input_alerts)
    behavior_result = BehaviorValidator.validate(response.scores, pattern_traps)
    
    # 3. Compute immediate score (0-0.5)
    immediate_score = (0.2 * integrity_result.score) + (0.3 * behavior_result.score)
    
    # 4. Store for later
    miner_submissions[miner.id] = {
        'processing_date': response.processing_date,
        'scores': response.scores,
        'immediate_score': immediate_score
    }

# 5. Aggregate all miners' scores → Ensemble
ensemble_scores = WeightedEnsembleAggregator.aggregate(
    submissions=miner_submissions,
    weights=current_miner_weights  # Based on historical performance
)

# 6. Publish ensemble to SOT or validator API
validator.publish_ensemble(
    processing_date='2025-10-26',
    ensemble_scores=ensemble_scores
)

# 7. Set temporary weights
validator.set_weights(immediate_scores)
```

**Day T+τ (Ground Truth Validation):**
```python
# 1. Load ground truth
ground_truth = load_ground_truth('2025-10-26')

# 2. Re-evaluate each miner's historical submission
for miner in miner_submissions:
    # Get historical scores (miner still serves them)
    historical_scores = miner.get('/scores/alerts/2025-10-26')
    
    # Validate against ground truth
    gt_result = GroundTruthValidator.validate(historical_scores, ground_truth)
    
    # Compute ground truth score (0-0.5)
    ground_truth_score = (0.3 * gt_result.auc_roc) + (0.2 * gt_result.auc_pr)
    
    # Final score
    final_score = immediate_score + ground_truth_score  # 0.5 + 0.5 = 1.0

# 3. Update weights based on final scores
validator.set_weights(final_scores)
```

---

## Alert App Consumption

### Option 1: Alert App Queries Validator (Recommended)

```
Alert App → Validator API → Ensemble Scores
```

**Validator exposes API:**
```python
GET /ensemble/scores/{processing_date}
GET /ensemble/rankings/{processing_date}
```

**Alert App Usage:**
```bash
# Get canonical (aggregated) scores for display
curl http://validator:9000/ensemble/scores/2025-10-26
```

**Pros:**
- 
✅ Single source of truth (validator's ensemble)
- ✅ Consistent across all consumers
- ✅ Alert app doesn't need to aggregate miners itself

**Cons:**
- ❌ Validator becomes single point of failure
- ❌ Validator needs to expose public API

---

### Option 2: Alert App Queries Miners Directly

```
Alert App → Miners → Individual Scores → Alert App Aggregates
```

**Alert App Logic:**
```python
# Query all miners
all_scores = []
for miner in top_miners:
    scores = miner.get('/scores/alerts/latest')
    all_scores.append(scores)

# Aggregate locally
ensemble = aggregate_with_weights(all_scores, miner_weights)
display_top_alerts(ensemble)
```

**Pros:**
- ✅ No dependency on validator
- ✅ Alert app can experiment with different aggregation

**Cons:**
- ❌ Alert app needs to know about miners
- ❌ Inconsistent results if different aggregation used
- ❌ Higher latency (multiple queries)

---

### Recommended: Hybrid Approach

**Primary**: Alert App queries Validator's ensemble
**Fallback**: Alert App can query individual miners

```
┌─────────────┐
│  Alert App  │
└──────┬──────┘
       │
       ├─ Primary: GET validator/ensemble/scores/latest
       │           └─ Returns canonical scores
       │
       └─ Fallback: GET miner_1/scores/alerts/latest
                    GET miner_2/scores/alerts/latest
                    └─ Aggregate locally if validator down
```

---

## Complete Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         DAY 0 FLOW                              │
└────────────────────────────────────────────────────────────────┘

00:00 UTC - SOT Publishes Batch
├─ Files: alerts.parquet, features.parquet, clusters.parquet
└─ processing_date: 2025-10-26

00:30 UTC - Miners Process
├─ Download: scripts/download_batch.py --processing-date 2025-10-26
├─ Process:  scripts/process_batch.py --processing-date 2025-10-26
├─ Output:   output/2025-10-26/alert_scores.parquet
└─ API:      GET /scores/alerts/2025-10-26
             GET /scores/alerts/latest  (same result on Day 0)

01:00 UTC - Validator Collects & Validates
├─ Query miners: GET /scores/alerts/latest
│  └─ All miners return: processing_date=2025-10-26, scores=[...]
│
├─ Immediate Validation (Integrity + Behavior)
│  ├─ IntegrityValidator → 0-0.2 score
│  ├─ BehaviorValidator  → 0-0.3 score
│  └─ immediate_score = 0-0.5 total
│
├─ Aggregate → Ensemble
│  └─ Weighted average of all miners' scores
│
├─ Publish Ensemble
│  └─ validator API: GET /ensemble/scores/2025-10-26
│
└─ Set Temporary Weights
   └─ Bittensor chain weight update based on immediate_score

01:30 UTC - Alert App Displays
└─ Query: GET validator/ensemble/scores/latest
   └─ Shows top-risk alerts to analysts


┌────────────────────────────────────────────────────────────────┐
│                      DAY T+τ FLOW                               │
│                    (e.g., 7 days later)                         │
└────────────────────────────────────────────────────────────────┘

Day 0+7 - Ground Truth Available
├─ SAR filings completed
├─ Exchange labels confirmed
└─ SOT publishes: ground_truth_2025-10-26.parquet

Validator Re-evaluates Historical Scores
├─ Load ground truth for 2025-10-26
│
├─ Query miners' historical scores
│  └─ GET /scores/alerts/2025-10-26
│     └─ Miners still serve this (kept in output/ directory)
│
├─ Ground Truth Validation
│  ├─ GroundTruthValidator → 0-0.5 score
│  └─ Metrics: AUC-ROC, AUC-PR, F1
│
├─ Compute Final Scores
│  └─ final_score = immediate_score (0.5) + ground_truth_score (0.5)
│
└─ Update Weights
   └─ Bittensor chain weight update based on final_score
```

---

## Why Date Parameter Makes Sense

### On Day 0:
```bash
# Validator knows today's date from SOT publication
# Queries miners for today's batch
GET /scores/alerts/2025-10-26

# OR uses latest endpoint (miner returns same date)
GET /scores/alerts/latest
# Response: {processing_date: "2025-10-26", ...}
```

### On Day T+τ:
```bash
# Validator needs historical scores for comparison with ground truth
# Must specify exact date
GET /scores/alerts/2025-10-26

# Latest won't work here - miner has processed newer dates
GET /scores/alerts/latest
# Response: {processing_date: "2025-11-02", ...} ← Wrong date!
```

**Conclusion**: Date parameter is essential for T+τ validation.

---

## Miner Data Retention Policy

### How Long Must Miners Keep Scores?

**Minimum**: τ + buffer days
- If τ = 7 days, keep at least 10 days
- If τ = 30 days, keep at least 35 days

**Recommended**: 90 days
- Allows re-validation if ground truth is delayed
- Enables audit trails
- Supports A/B testing across time

**Implementation**:
```python
# In process_batch.py
RETENTION_DAYS = 90

# Cleanup old scores
import shutil
from datetime import datetime, timedelta

def cleanup_old_scores(output_dir: Path, retention_days: int = 90):
    cutoff = datetime.now() - timedelta(days=retention_days)
    
    for date_dir in output_dir.iterdir():
        if date_dir.is_dir():
            try:
                date_str = date_dir.name
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                
                if date_obj < cutoff:
                    logger.info(f"Removing old scores: {date_dir}")
                    shutil.rmtree(date_dir)
            except ValueError:
                continue
```

---

## Updated API Implementation

### Add Latest Endpoints

```python
# aml_miner/api/routes.py

@router.get("/scores/alerts/latest", response_model=Dict)
def get_latest_alert_scores():
    latest_date = db.get_latest_date()
    
    if not latest_date:
        raise HTTPException(404, "No scores available")
    
    scores_df = db.get_alert_scores(latest_date)
    metadata = db.get_date_metadata(latest_date)
    
    return {
        "processing_date": latest_date,
        "scores": scores_df.to_dict(orient='records'),
        "metadata": metadata
    }

@router.get("/rankings/alerts/latest", response_model=Dict)
def get_latest_alert_rankings():
    latest_date = db.get_latest_date()
    
    if not latest_date:
        raise HTTPException(404, "No rankings available")
    
    rankings_df = db.get_alert_rankings(latest_date)
    metadata = db.get_date_metadata(latest_date)
    
    return {
        "processing_date": latest_date,
        "rankings": rankings_df.to_dict(orient='records'),
        "metadata": metadata
    }
```

---

## Validator API (Separate Service)

### Purpose
Validator needs its own API to serve ensemble scores to Alert App and other consumers.

### Endpoints

```python
# Ensemble Scores (Canonical)
GET  /ensemble/scores/{processing_date}     # Aggregated scores for date
GET  /ensemble/rankings/{processing_date}   # Aggregated rankings for date
GET  /ensemble/scores/latest                # Latest aggregated scores
GET  /ensemble/rankings/latest              # Latest aggregated rankings

# Miner Submissions (For Transparency)
GET  /submissions/{processing_date}         # All miner submissions for date
GET  /submissions/{processing_date}/{miner_id}  # Specific miner's submission

# Validation Results
GET  /validation/{processing_date}          # Validation results for all miners
GET  /validation/{processing_date}/{miner_id}   # Specific miner's validation

# Ground Truth
GET  /ground-truth/{processing_date}        # Ground truth labels (when available)

# Miner Performance
GET  /miners/rankings                       # Current miner rankings
GET  /miners/{miner_id}/performance         # Historical performance metrics
```

### Example Response

```json
GET /ensemble/scores/2025-10-26

{
  "processing_date": "2025-10-26",
  "aggregation_method": "weighted_ensemble",
  "total_miners": 25,
  "ensemble_scores": [
    {
      "alert_id": "alert_001",
      "ensemble_score": 0.89,
      "confidence": 0.95,
      "miner_agreement": 0.87,
      "top_contributors": [
        {"miner_id": "miner_5", "weight": 0.15, "score": 0.91},
        {"miner_id": "miner_12", "weight": 0.12, "score": 0.88}
      ]
    }
  ],
  "metadata": {
    "aggregated_at": "2025-10-26T01:15:00Z",
    "immediate_validation_complete": true,
    "ground_truth_validation_pending": true,
    "expected_ground_truth_date": "2025-11-02"
  }
}
```

---

## Summary

### API Design Answers:

1. **Day 0 Validation**: Validator uses `GET /scores/alerts/latest` to get today's scores
2. **Day T+τ Validation**: Validator uses `GET /scores/alerts/{processing_date}` to get historical scores
3. **Date Parameter**: Essential for T+τ validation, optional for Day 0 (can use `/latest`)
4. **Alert App**: Should query Validator's ensemble API for canonical scores

### Recommended Endpoints:

**Miner API:**
```
GET /scores/alerts/latest                   ← Day 0
GET /scores/alerts/{processing_date}        ← Day T+τ
GET /dates/available                        ← Discovery
GET /dates/latest                           ← Discovery
```

**Validator API (New Service):**
```
GET /ensemble/scores/latest                 ← Alert App uses this
GET /ensemble/scores/{processing_date}      ← Historical ensemble
GET /validation/{processing_date}           ← Validation results
GET /miners/rankings                        ← Miner leaderboard
```

### Data Flow:

```
Day 0:
SOT → Miners → Validator → Ensemble → Alert App

Day T+τ:
Ground Truth → Validator → Re-evaluate Miners → Update Weights
```