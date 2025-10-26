# Score Outputs and A/B Validation
## Detailed Examples and Validation Process

**Date**: 2025-10-26  
**Repository Name**: **`alert-scoring`** ⭐  
**Purpose**: Show exact score outputs and A/B validation methodology

---

## Expected Score Outputs

### 1. Alert Scores

**Input**: Raw alert from SOT
```json
{
  "alert_id": "alert_20251026_001",
  "address": "0xabc123...",
  "typology_type": "layering",
  "severity": "HIGH",
  "volume_usd": 150000.00,
  "alert_confidence_score": 0.85
}
```

**Output**: ML-scored alert
```json
{
  "alert_id": "alert_20251026_001",
  "score": 0.8734,
  "model_version": "v1.2.0",
  "latency_ms": 12.5,
  "explain_json": {
    "top_features": [
      {"name": "volume_usd", "contribution": 0.35},
      {"name": "alert_confidence_score", "contribution": 0.28},
      {"name": "address_pagerank", "contribution": 0.22},
      {"name": "typology_risk_weight", "contribution": 0.15}
    ],
    "shap_values": {
      "volume_usd": 0.25,
      "alert_confidence_score": 0.18,
      "address_pagerank": 0.15
    }
  }
}
```

**Score Interpretation:**
- `0.0 - 0.3` = Low risk
- `0.3 - 0.6` = Medium risk
- `0.6 - 0.8` = High risk
- `0.8 - 1.0` = Critical risk

---

### 2. Alert Rankings

**Input**: All scored alerts for processing_date
```json
[
  {"alert_id": "alert_001", "score": 0.8734},
  {"alert_id": "alert_002", "score": 0.9201},
  {"alert_id": "alert_003", "score": 0.7543},
  {"alert_id": "alert_004", "score": 0.8901}
]
```

**Output**: Ranked by priority
```json
[
  {
    "alert_id": "alert_002",
    "rank": 1,
    "score": 0.9201,
    "model_version": "v1.2.0"
  },
  {
    "alert_id": "alert_004",
    "rank": 2,
    "score": 0.8901,
    "model_version": "v1.2.0"
  },
  {
    "alert_id": "alert_001",
    "rank": 3,
    "score": 0.8734,
    "model_version": "v1.2.0"
  },
  {
    "alert_id": "alert_003",
    "rank": 4,
    "score": 0.7543,
    "model_version": "v1.2.0"
  }
]
```

---

### 3. Cluster Scores

**Input**: Alert cluster from SOT
```json
{
  "cluster_id": "cluster_001",
  "primary_alert_id": "alert_001",
  "related_alert_ids": ["alert_001", "alert_002", "alert_003"],
  "total_alerts": 5,
  "total_volume_usd": 450000.00,
  "severity_max": "CRITICAL",
  "confidence_avg": 0.82
}
```

**Output**: ML-scored cluster
```json
{
  "cluster_id": "cluster_001",
  "score": 0.9156,
  "model_version": "v1.0.0",
  "explain_json": {
    "top_features": [
      {"name": "total_volume_usd", "contribution": 0.38},
      {"name": "severity_max", "contribution": 0.31},
      {"name": "total_alerts", "contribution": 0.19},
      {"name": "confidence_avg", "contribution": 0.12}
    ],
    "cluster_coherence": 0.87,
    "alert_score_avg": 0.85
  }
}
```

---

## A/B Validation Process

### Scenario: Comparing Two Models

**Model A**: `alert_scorer_v1.0.0.txt` (current production)  
**Model B**: `alert_scorer_v1.1.0.txt` (new candidate)

### Step 1: Score Same Batch with Both Models

```bash
# Score with Model A
python scripts/process_batch.py \
    --processing-date 2025-10-26 \
    --network ethereum \
    --alert-scorer trained_models/alert_scorer_v1.0.0.txt \
    --output-table alert_scores_model_a

# Score with Model B
python scripts/process_batch.py \
    --processing-date 2025-10-26 \
    --network ethereum \
    --alert-scorer trained_models/alert_scorer_v1.1.0.txt \
    --output-table alert_scores_model_b
```

### Step 2: Load Ground Truth

**Ground Truth** = T+τ labels (e.g., 7 days later)
- SAR filings
- Exchange labels
- Confirmed illicit addresses

```sql
-- Ground truth table
SELECT 
    alert_id,
    is_sar_filed,        -- Did this alert lead to SAR filing?
    is_confirmed_illicit -- Was address confirmed illicit?
FROM ground_truth
WHERE processing_date = '2025-10-26'
```

Example ground truth:
```json
[
  {
    "alert_id": "alert_001",
    "is_sar_filed": true,
    "is_confirmed_illicit": true,
    "label": 1  // Positive (actual money laundering)
  },
  {
    "alert_id": "alert_002",
    "is_sar_filed": false,
    "is_confirmed_illicit": false,
    "label": 0  // Negative (false positive)
  }
]
```

### Step 3: Run A/B Validation

```bash
python scripts/validate_models.py \
    --model-a-scores alert_scores_model_a \
    --model-b-scores alert_scores_model_b \
    --processing-date 2025-10-26 \
    --ground-truth ground_truth_2025-10-26.parquet \
    --output validation_results/v1.0.0_vs_v1.1.0.json
```

### Step 4: Validation Results

**Model A Results:**
```json
{
  "model": "v1.0.0",
  "integrity_validation": {
    "passed": true,
    "score": 0.20,
    "checks": {
      "completeness": true,
      "schema": true,
      "score_range": true,
      "latency": true,
      "determinism": true
    }
  },
  "behavior_validation": {
    "passed": true,
    "score": 0.28,
    "traps_detected": [],
    "variance": 0.145,
    "median_score": 0.52
  },
  "ground_truth_validation": {
    "passed": true,
    "score": 0.42,
    "metrics": {
      "auc_roc": 0.8421,
      "auc_pr": 0.7834,
      "best_f1": 0.7612,
      "best_threshold": 0.62,
      "n_samples": 1000,
      "n_positive": 234,
      "positive_rate": 0.234
    }
  },
  "final_score": 0.90  // 0.20 + 0.28 + 0.42
}
```

**Model B Results:**
```json
{
  "model": "v1.1.0",
  "integrity_validation": {
    "passed": true,
    "score": 0.20,
    "checks": {
      "completeness": true,
      "schema": true,
      "score_range": true,
      "latency": true,
      "determinism": true
    }
  },
  "behavior_validation": {
    "passed": true,
    "score": 0.30,
    "traps_detected": [],
    "variance": 0.162,
    "median_score": 0.55
  },
  "ground_truth_validation": {
    "passed": true,
    "score": 0.46,
    "metrics": {
      "auc_roc": 0.8756,      // +0.0335 improvement
      "auc_pr": 0.8123,        // +0.0289 improvement
      "best_f1": 0.7891,       // +0.0279 improvement
      "best_threshold": 0.58,
      "n_samples": 1000,
      "n_positive": 234,
      "positive_rate": 0.234
    }
  },
  "final_score": 0.96  // 0.20 + 0.30 + 0.46 = BETTER!
}
```

**Comparison Summary:**
```json
{
  "winner": "Model B (v1.1.0)",
  "improvements": {
    "auc_roc": +0.0335,
    "auc_pr": +0.0289,
    "f1": +0.0279,
    "final_score": +0.06
  },
  "recommendation": "Deploy Model B to production",
  "confidence": "High - statistically significant improvement across all metrics"
}
```

---

## Detailed Validation Metrics

### Integrity Score (0-0.2 range)

**What it validates:**
1. **Completeness**: All alerts scored
2. **Schema**: Correct columns and types
3. **Score Range**: All scores in [0, 1]
4. **Latency**: Average < 100ms per alert
5. **Determinism**: Same input → same output

**How it's calculated:**
```python
if all_checks_pass:
    integrity_score = 0.20
else:
    integrity_score = 0.00  # Fail fast
```

### Behavior Score (0-0.3 range)

**What it validates:**
1. **Variance**: Not all scores identical (gaming detection)
2. **Pattern Traps**: Doesn't match known bad patterns
3. **Median**: Reasonable middle value
4. **Plagiarism**: Not copying other miners

**How it's calculated:**
```python
base_score = 0.30

# Penalize issues
if variance < 0.001:
    base_score -= 0.10  # Constant scores (gaming)

if pattern_traps_detected > 0:
    base_score -= (0.05 * pattern_traps_detected)

if median < 0.1 or median > 0.9:
    base_score -= 0.05  # Extreme median

behavior_score = max(0.0, base_score)
```

### Ground Truth Score (0-0.5 range)

**What it validates:**
- Accuracy against real-world outcomes
- AUC-ROC (overall discrimination)
- AUC-PR (precision-recall tradeoff)

**How it's calculated:**
```python
auc_roc = compute_auc_roc(y_true, y_pred)
auc_pr = compute_auc_pr(y_true, y_pred)

# Weighted combination
ground_truth_score = (0.3 * auc_roc) + (0.2 * auc_pr)

# Max is 0.5 (perfect score on both metrics)
```

---

## Complete Example: End-to-End

### Day 0 (2025-10-26)

**1. Miner processes batch:**
```bash
python scripts/process_batch.py \
    --processing-date 2025-10-26 \
    --network ethereum
```

**2. Stores results in ClickHouse:**
```sql
INSERT INTO alert_scores VALUES
('2025-10-26', 'ethereum', 'alert_001', 0.8734, 'v1.2.0', 12.5, '{"top_features":[...]}'),
('2025-10-26', 'ethereum', 'alert_002', 0.9201, 'v1.2.0', 11.8, '{"top_features":[...]}'),
...
```

**3. Validator queries miner:**
```bash
curl "http://miner:8000/scores/alerts/latest?network=ethereum"
```

**4. Validator runs immediate validation:**
```python
# Integrity check
integrity_result = IntegrityValidator().validate(miner_scores, input_alerts)
# Score: 0.20 (passed all checks)

# Behavior check
behavior_result = BehaviorValidator().validate(miner_scores, pattern_traps)
# Score: 0.28 (good variance, no traps)

# Immediate score
immediate_score = 0.20 + 0.28 = 0.48 (out of 0.5 possible)
```

### Day T+7 (2025-11-02)

**1. Ground truth available:**
```sql
-- SAR filings confirmed
INSERT INTO ground_truth VALUES
('alert_001', '2025-10-26', true, 1),   -- Was money laundering
('alert_002', '2025-10-26', false, 0),  -- False positive
...
```

**2. Validator re-evaluates:**
```bash
curl "http://miner:8000/scores/alerts/2025-10-26?network=ethereum"
```

**3. Validator runs ground truth validation:**
```python
# Load historical scores from miner
historical_scores = miner.get_scores('2025-10-26', 'ethereum')

# Load ground truth
ground_truth = load_ground_truth('2025-10-26')

# Validate
gt_result = GroundTruthValidator().validate(historical_scores, ground_truth)
# Metrics: AUC-ROC=0.8756, AUC-PR=0.8123
# Score: 0.46 (out of 0.5 possible)

# Final score
final_score = immediate_score (0.48) + ground_truth_score (0.46) = 0.94
```

**4. Validator updates weights:**
```python
# Miner performed well!
validator.set_weights(miner_id, weight=0.94)
```

---

## Summary

### Repository Name
**`alert-scoring`** ✅

### Score Outputs
1. **Alert Scores**: `{alert_id, score, model_version, latency_ms, explain_json}`
2. **Rankings**: `{alert_id, rank, model_version}`
3. **Cluster Scores**: `{cluster_id, score, model_version, explain_json}`

### A/B Validation
1. Score same batch with Model A and Model B
2. Compare against ground truth (T+τ labels)
3. Compute metrics: AUC-ROC, AUC-PR, F1
4. Winner = highest final_score (integrity + behavior + ground_truth)

### Validation Scores
- **Integrity**: 0-0.2 (pass/fail checks)
- **Behavior**: 0-0.3 (gaming detection)
- **Ground Truth**: 0-0.5 (accuracy metrics)
- **Total**: 0-1.0 (final miner score)

This validation ensures miners produce **accurate, honest, deterministic** scores!