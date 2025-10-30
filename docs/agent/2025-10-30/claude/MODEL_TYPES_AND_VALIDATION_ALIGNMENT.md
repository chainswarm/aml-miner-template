# Model Types and Validation Alignment

**Date**: 2025-10-30  
**Purpose**: Clarify differences between alert_scorer, alert_ranker, and cluster_scorer, and their alignment with validation algorithms

---

## Overview of Model Types

We have **three distinct model types**, each serving a different purpose in the risk assessment pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                   RISK ASSESSMENT PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. ALERT SCORER    →  Score individual alerts [0-1]        │
│     Input: Single alert + features                          │
│     Output: P(illicit within τ days)                        │
│     Purpose: Quantify individual alert risk                 │
│                                                              │
│  2. ALERT RANKER    →  Rank alerts by priority              │
│     Input: Batch of scored alerts                           │
│     Output: Ranked list (1, 2, 3, ...)                     │
│     Purpose: Prioritize investigation order                 │
│                                                              │
│  3. CLUSTER SCORER  →  Score alert clusters [0-1]           │
│     Input: Cluster of related alerts + features             │
│     Output: P(cluster represents coordinated illicit)       │
│     Purpose: Assess coordinated/organized risk              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Alert Scorer

### What It Does
Assigns a risk score to **each individual alert**, representing the probability that the alert corresponds to illicit activity.

### Input
```python
{
  "alert_id": "alert_001",
  "address": "0xabc123...",
  "typology_type": "layering",
  "severity": "HIGH",
  "volume_usd": 150000.00,
  "alert_confidence_score": 0.85
}
```

### Output
```python
{
  "alert_id": "alert_001",
  "score": 0.8734,  # P(illicit)
  "model_version": "v1.2.0"
}
```

### Purpose
- **Risk Quantification**: How risky is THIS specific alert?
- **Absolute Scoring**: Each alert scored independently
- **Validation Target**: Primary validation against ground truth

### Training Objective
**Binary Classification**
```python
# Predict: Is this alert illicit? (0 = benign, 1 = illicit)
model.fit(X_alerts, y_illicit)
```

### Validation Alignment
✅ **Directly validated** against ground truth (T+τ):
- AUC-ROC: Ability to distinguish illicit vs benign
- AUC-PR: Precision-recall tradeoff
- Brier Score: Calibration quality

---

## 2. Alert Ranker

### What It Does
**Orders alerts by investigation priority** - determines which alerts should be investigated first.

### Input
```python
# Batch of alerts with scores
[
  {"alert_id": "alert_001", "score": 0.87},
  {"alert_id": "alert_002", "score": 0.92},
  {"alert_id": "alert_003", "score": 0.75},
  {"alert_id": "alert_004", "score": 0.89}
]
```

### Output
```python
# Ranked by priority
[
  {"alert_id": "alert_002", "rank": 1, "score": 0.92},
  {"alert_id": "alert_004", "rank": 2, "score": 0.89},
  {"alert_id": "alert_001", "rank": 3, "score": 0.87},
  {"alert_id": "alert_003", "rank": 4, "score": 0.75}
]
```

### Purpose
- **Investigation Prioritization**: Which alerts should analysts review first?
- **Relative Ordering**: Ranks alerts against each other
- **Resource Optimization**: Top-K alerts should have highest confirmed illicit rate

### Training Objective
**Learning-to-Rank (LambdaMART)**
```python
# Predict: What's the best order for investigating these alerts?
# Optimize NDCG@K (quality of top-K results)
ranker.fit(X_alerts, y_illicit, groups=batch_groups)
```

### Validation Alignment
✅ **Validated** via ranking metrics:
- **NDCG@K**: Quality of top-K rankings
- **Precision@K**: Fraction of top-K that are illicit
- **Coverage**: Top-K covers most illicit alerts
- **Stability**: Kendall-τ correlation with baseline

### Relationship to Alert Scorer
**Ranker uses scorer output + additional features**:
```python
# Ranker considers:
- alert_score (from scorer)
- severity_weight
- cluster_importance
- freshness_decay
- investigation_history
```

---

## 3. Cluster Scorer

### What It Does
Scores **groups of related alerts** (clusters) to assess coordinated/organized illicit activity.

### What is a Cluster?
A cluster represents **related alerts** that may indicate:
- Coordinated money laundering
- Organized criminal network
- Multi-account fraud scheme
- Layering pattern across addresses

### Input
```python
{
  "cluster_id": "cluster_001",
  "primary_alert_id": "alert_001",
  "related_alert_ids": ["alert_001", "alert_002", "alert_003"],
  "total_alerts": 5,
  "addresses_involved": ["0xabc...", "0xdef...", "0x123..."],
  "total_volume_usd": 450000.00,
  "severity_max": "CRITICAL",
  "confidence_avg": 0.82
}
```

### Output
```python
{
  "cluster_id": "cluster_001",
  "score": 0.9156,  # P(coordinated illicit activity)
  "model_version": "v1.0.0"
}
```

### Purpose
- **Coordinated Risk Assessment**: Is this a coordinated illicit operation?
- **Network Analysis**: Evaluate relationships between addresses
- **Higher-Level Patterns**: Detect organized crime vs individual bad actors

### Training Objective
**Binary Classification (at cluster level)**
```python
# Predict: Is this cluster a coordinated illicit operation?
model.fit(X_clusters, y_cluster_illicit)
```

### Validation Alignment
✅ **Validated** against cluster-level ground truth:
- AUC-ROC: Distinguish illicit vs benign clusters
- Cluster coherence: Internal consistency
- Pattern validation: Structural pattern confirmation

---

## Key Differences Summary

| Aspect | Alert Scorer | Alert Ranker | Cluster Scorer |
|--------|-------------|--------------|----------------|
| **Granularity** | Individual alert | Alert ordering | Alert group |
| **Output Type** | Risk score [0-1] | Rank (1, 2, 3...) | Risk score [0-1] |
| **Question** | "How risky is THIS alert?" | "Which alerts first?" | "Coordinated activity?" |
| **Algorithm** | Binary classification | Learning-to-rank | Binary classification |
| **Validation** | AUC-ROC, AUC-PR | NDCG@K, Precision@K | AUC-ROC (cluster) |
| **Dependencies** | None | Uses alert scores | Uses alert info |

---

## Validation Alignment

### Tier 1: Integrity Validation (Immediate)
**Validates**: All three model types

```python
# Check for all model outputs:
- Schema compliance (correct columns, types)
- Completeness (all alerts/clusters scored)
- Score range [0, 1] for scorers
- Rank uniqueness for ranker
- Latency constraints
```

### Tier 2: Behavior Validation (Immediate)
**Validates**: All three model types

```python
# Pattern trap detection:
- Alert Scorer: Detects gaming via constant scores
- Alert Ranker: Detects manipulation via ranking patterns
- Cluster Scorer: Detects cluster-level gaming
```

### Tier 3: Ground Truth Validation (T+τ)
**Primary**: Alert Scorer  
**Secondary**: Cluster Scorer  
**Not Directly**: Alert Ranker (validated via ranked alert outcomes)

```python
# T+τ days later (e.g., 7-30 days)
ground_truth = {
  'alert_001': {'confirmed_illicit': True},   # SAR filed
  'alert_002': {'confirmed_illicit': False},  # Benign
  'cluster_001': {'confirmed_illicit': True}  # Network confirmed
}

# Validate Alert Scorer
alert_auc = roc_auc_score(
  y_true=[gt['confirmed_illicit'] for gt in ground_truth.values()],
  y_pred=[scores['alert_001'], scores['alert_002'], ...]
)

# Validate Cluster Scorer
cluster_auc = roc_auc_score(
  y_true=[cluster_gt['confirmed_illicit']],
  y_pred=[cluster_scores['cluster_001'], ...]
)

# Validate Alert Ranker (indirectly)
# Check if top-ranked alerts have higher confirmed illicit rate
top_k_precision = sum(
  ground_truth[alert_id]['confirmed_illicit'] 
  for alert_id in top_k_ranked
) / k
```

---

## Should We Score Both Alerts AND Clusters?

### Answer: **Yes, both are valuable**

### Alert Scoring (PRIMARY)
✅ **Required** for:
- Individual alert risk assessment
- Alert app display (show risk per alert)
- Investigation prioritization base
- Direct ground truth validation

### Cluster Scoring (COMPLEMENTARY)
✅ **Valuable** for:
- Detecting organized/coordinated activity
- Network-level risk assessment
- Identifying sophisticated schemes
- Complementary signal to individual alerts

### Why Both Matter

**Example Scenario**:
```
Individual Alerts:
├─ alert_001: score=0.65 (medium risk)
├─ alert_002: score=0.62 (medium risk)
└─ alert_003: score=0.68 (medium risk)

Cluster Analysis:
└─ cluster_001 (alerts 001-003): score=0.92 (HIGH RISK)
   └─ Reason: Coordinated layering pattern
   └─ All three alerts from same criminal network
```

**Insight**: Individual alerts appear medium risk, but **together** they form high-risk coordinated scheme.

---

## Recommended Implementation Priority

### Phase 1: Core (Highest Priority)
1. ✅ **Alert Scorer** - Individual risk quantification
   - Direct ground truth validation
   - Primary validation metric (AUC-ROC)
   - Required for all downstream use cases

### Phase 2: Investigation Optimization
2. ✅ **Alert Ranker** - Investigation prioritization
   - Uses alert scorer output
   - Optimizes analyst workflow
   - Validated via ranking quality metrics

### Phase 3: Advanced Analysis
3. ✅ **Cluster Scorer** - Coordinated activity detection
   - Complementary to alert scoring
   - Detects organized crime patterns
   - Validated against cluster-level ground truth

---

## Current Implementation Status

### ✅ Completed
All three models are **implemented and ready**:

1. **Alert Scorer** ([`model_training.py`](../../packages/training/model_training.py))
   - LightGBM binary classifier
   - 40+ features
   - Cross-validation support

2. **Alert Ranker** ([`model_training.py`](../../packages/training/model_training.py))
   - LambdaMART ranker
   - Uses alert scores + ranking features
   - NDCG@K optimization

3. **Cluster Scorer** ([`model_training.py`](../../packages/training/model_training.py))
   - LightGBM binary classifier
   - Cluster-level features
   - Network analysis

### Training Usage

```bash
# Train Alert Scorer (PRIMARY)
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer \
    --window-days 7

# Train Alert Ranker
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_ranker \
    --window-days 7

# Train Cluster Scorer
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type cluster_scorer \
    --window-days 7
```

---

## Validation Alignment Summary

```
┌─────────────────────────────────────────────────────────┐
│              VALIDATION FRAMEWORK                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  T+0 (Immediate Validation)                             │
│  ├─ Integrity: ALL three models                         │
│  │  └─ Schema, completeness, latency                    │
│  └─ Behavior: ALL three models                          │
│     └─ Gaming detection, pattern traps                  │
│                                                          │
│  T+τ (Ground Truth Validation)                          │
│  ├─ Alert Scorer: DIRECT validation                     │
│  │  └─ AUC-ROC against confirmed illicit (0.5 points)  │
│  ├─ Cluster Scorer: DIRECT validation                   │
│  │  └─ AUC-ROC against confirmed clusters              │
│  └─ Alert Ranker: INDIRECT validation                   │
│     └─ Top-K precision, NDCG@K                          │
│                                                          │
│  Final Miner Score = 0.5 (immediate) + 0.5 (ground truth)│
└─────────────────────────────────────────────────────────┘
```

---

## Recommendations

### For Validation
1. ✅ **Primary focus**: Alert Scorer validation (direct ground truth)
2. ✅ **Secondary**: Cluster Scorer validation (cluster-level ground truth)
3. ✅ **Tertiary**: Alert Ranker validation (ranking quality metrics)

### For Training Priority
1. **Start with Alert Scorer** - Most critical for validation
2. **Add Alert Ranker** - Improves investigation efficiency
3. **Add Cluster Scorer** - Detects coordinated schemes

### For Ground Truth Labeling
Collect labels at **both levels**:
```python
ground_truth = {
  # Alert-level labels (REQUIRED)
  'alerts': {
    'alert_001': {'confirmed_illicit': True, 'sar_filed': True},
    'alert_002': {'confirmed_illicit': False}
  },
  
  # Cluster-level labels (OPTIONAL but valuable)
  'clusters': {
    'cluster_001': {'confirmed_illicit': True, 'network_type': 'layering'}
  }
}
```

---

## Conclusion

All three model types serve distinct purposes and are **aligned with validation**:

- **Alert Scorer**: Core risk assessment, directly validated against ground truth
- **Alert Ranker**: Investigation prioritization, validated via ranking metrics
- **Cluster Scorer**: Coordinated activity detection, validated at cluster level

The current implementation supports **all three**, allowing miners to:
1. Score individual alerts (required for primary validation)
2. Rank alerts by priority (optional, for better investigation workflow)
3. Score clusters (optional, for detecting organized schemes)

**All three contribute to the final miner score**, with Alert Scorer being the most heavily weighted due to direct ground truth validation.