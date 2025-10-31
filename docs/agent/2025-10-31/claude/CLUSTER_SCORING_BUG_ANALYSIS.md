# Cluster Scoring Bug Analysis and Solution Plan

**Date:** 2025-10-31  
**Issue:** ValueError - All arrays must be of the same length

## Root Cause Analysis

### The Problem

The error occurs in [`score_generator.py:87`](packages/scoring/score_generator.py:87) when creating a DataFrame for cluster scores:

```python
result = pd.DataFrame({
    'cluster_id': cluster_ids.values,  # 25 cluster IDs
    'score': scores                     # 86 scores (from 86 alerts)
})
```

### Why This Happens

1. **In [`risk_scoring.py:212`](packages/scoring/risk_scoring.py:212)**, cluster scoring builds features from ALL alerts:
   ```python
   builder = FeatureBuilder()
   X_clusters = builder.build_inference_features(data)  # Uses all 86 alerts
   ```

2. **In [`risk_scoring.py:214`](packages/scoring/risk_scoring.py:214)**, only cluster IDs are extracted:
   ```python
   cluster_ids = data['clusters']['cluster_id']  # Only 25 clusters
   ```

3. **The mismatch:** We score 86 alert features but try to match them with 25 cluster IDs.

### The Core Issue

**Cluster scoring requires CLUSTER-LEVEL features, not ALERT-LEVEL features.**

Currently, [`feature_builder.py`](packages/training/feature_builder.py) has only one method:
- [`build_inference_features()`](packages/training/feature_builder.py:10) - builds features from alerts

We need a separate method to build features AT THE CLUSTER LEVEL by aggregating alert and address features.

## Architecture Design

### Current Flow (Broken)

```
Alerts (86) → build_inference_features() → Alert Features (86×56)
                                                ↓
                                          score_clusters()
                                                ↓
                                    TRY to match with Clusters (25)
                                                ↓
                                            ERROR ❌
```

### Correct Flow (Needed)

```
Alerts (86) → build_cluster_features() → Cluster Features (25×N)
Clusters (25) ↗                              ↓
                                      score_clusters()
                                             ↓
                                    Cluster Scores (25) ✓
```

## Solution Architecture

### 1. New Method: `build_cluster_features()`

Add to [`FeatureBuilder`](packages/training/feature_builder.py:8):

```python
def build_cluster_features(
    self,
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build cluster-level features by aggregating alert and address features.
    
    Returns one row per cluster with aggregated features.
    """
```

### 2. Cluster Feature Aggregation Strategy

For each cluster, aggregate features from:

#### A. Alert-level aggregations
- Count: number of alerts in cluster
- Sum: total volume across alerts
- Mean: average volume, confidence, severity
- Max/Min: maximum/minimum values
- Std: variability within cluster

#### B. Address-level aggregations  
- Unique addresses count
- Address type distribution
- Network degree statistics
- Label distribution

#### C. Temporal patterns
- Time span of alerts
- Alert frequency
- Day-of-week distribution

#### D. Cluster-specific features
- Cluster size (from clusters table)
- Total cluster volume (from clusters table)
- Network connectivity metrics

### 3. Feature Matrix Structure

**Alert Features (56 columns):**
- Used for: alert scoring, alert ranking
- One row per alert

**Cluster Features (TBD columns):**
- Used for: cluster scoring
- One row per cluster
- Aggregated from alert + address data

## Implementation Plan

### Phase 1: Add `build_cluster_features()` Method

**File:** [`packages/training/feature_builder.py`](packages/training/feature_builder.py)

```python
def build_cluster_features(
    self,
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build cluster-level features by aggregating alert features."""
    
    logger.info("Building cluster features")
    
    # Step 1: Build alert-level features first
    alert_features = self.build_inference_features(data)
    
    # Step 2: Map alerts to clusters
    cluster_map = self._build_cluster_alert_map(data['clusters'], data['alerts'])
    
    # Step 3: Aggregate features per cluster
    cluster_features = self._aggregate_cluster_features(
        alert_features, 
        cluster_map,
        data
    )
    
    logger.success(
        "Cluster feature building completed",
        extra={
            "num_clusters": len(cluster_features),
            "num_features": len(cluster_features.columns)
        }
    )
    
    return cluster_features
```

### Phase 2: Helper Methods

#### A. Build cluster-alert mapping
```python
def _build_cluster_alert_map(
    self,
    clusters_df: pd.DataFrame,
    alerts_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Map cluster_id to list of alert_ids."""
```

#### B. Aggregate features per cluster
```python
def _aggregate_cluster_features(
    self,
    alert_features: pd.DataFrame,
    cluster_map: Dict[str, List[str]],
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Aggregate alert features to cluster level."""
```

### Phase 3: Update Risk Scoring Pipeline

**File:** [`packages/scoring/risk_scoring.py`](packages/scoring/risk_scoring.py)

Change line 212 from:
```python
X_clusters = builder.build_inference_features(data)
```

To:
```python
X_clusters = builder.build_cluster_features(data)
```

### Phase 4: Training Pipeline Updates

**File:** [`packages/training/model_training.py`](packages/training/model_training.py)

Update cluster training to also use `build_cluster_features()` for consistency.

## Feature Aggregation Specifications

### Numerical Features (from alerts)
- **Aggregations:** mean, std, min, max, median, sum
- **Examples:** volume_usd, confidence_score, severity_encoded

### Categorical Features
- **Aggregations:** mode (most common), unique count, distribution
- **Examples:** address_type, risk_level

### Binary Features
- **Aggregations:** sum (count of True), mean (percentage)
- **Examples:** is_exchange_flag, is_mixer_flag, in_cluster

### Temporal Features
- **Aggregations:** min, max, range
- **Examples:** processing_date, day_of_week

## Canonical Cluster Feature Order

Similar to alert features, define canonical order for reproducibility:

```python
canonical_cluster_features = [
    'cluster_id',
    'cluster_size',
    'cluster_total_volume',
    'alert_count',
    'unique_addresses',
    'avg_volume_usd',
    'std_volume_usd',
    'max_volume_usd',
    'min_volume_usd',
    'avg_confidence',
    'max_severity',
    'exchange_address_count',
    'mixer_address_count',
    'high_risk_count',
    'avg_tx_count',
    'total_network_degree',
    # ... more aggregated features
]
```

## Testing Strategy

### Unit Tests
1. Test `build_cluster_features()` with known data
2. Verify aggregation correctness
3. Check feature matrix dimensions match cluster count

### Integration Tests
1. End-to-end cluster scoring with real data
2. Verify scores DataFrame has correct length
3. Validate all clusters get scored

### Data Validation
1. Check cluster_ids align between features and scores
2. Ensure no data leakage between training/inference
3. Validate feature consistency across train/score

## Migration Notes

### No Database Migration Needed
- Schema remains unchanged
- Only code logic changes

### Backward Compatibility
- Alert scoring unchanged
- Alert ranking unchanged
- Only cluster scoring affected

## Success Criteria

1. ✅ Cluster scoring completes without errors
2. ✅ Score count equals cluster count (25 = 25)
3. ✅ Features properly aggregated at cluster level
4. ✅ Training pipeline uses same feature building
5. ✅ Model can load and predict successfully

## Next Steps

1. Implement `build_cluster_features()` method
2. Add helper methods for aggregation
3. Update risk scoring to use new method
4. Update training pipeline
5. Test end-to-end
6. Switch to code mode for implementation