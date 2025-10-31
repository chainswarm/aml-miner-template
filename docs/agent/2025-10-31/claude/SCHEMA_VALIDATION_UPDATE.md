# Schema Validation Update - Cluster Scoring Fix

**Date:** 2025-10-31  
**Status:** Plan validated and enhanced with schema insights

## Schema Review Findings

### Key Schema Elements

#### 1. raw_clusters Table Structure
From [`raw_clusters.sql`](packages/storage/schema/raw_clusters.sql):

```sql
cluster_id String,
cluster_type String,
primary_alert_id String,
related_alert_ids Array(String),      -- ✓ Maps cluster → alerts
addresses_involved Array(String),     -- ✓ Maps cluster → addresses
total_alerts UInt32,                  -- ✓ Pre-aggregated feature
total_volume_usd Decimal128(18),      -- ✓ Pre-aggregated feature
severity_max String,                  -- ✓ Pre-aggregated feature
confidence_avg Float32,               -- ✓ Pre-aggregated feature
```

#### 2. cluster_scores Table Structure
From [`cluster_scores.sql`](packages/storage/schema/cluster_scores.sql):

```sql
processing_date Date,
cluster_id String,                    -- ✓ One row per cluster
score Float64,                        -- ✓ Score output
model_version String
```

### Critical Insight: `related_alert_ids`

The schema reveals that clusters contain **`related_alert_ids Array(String)`** - this is the direct mapping we need for aggregation!

**Current broken code** in [`feature_builder.py:265`](packages/training/feature_builder.py:265):
```python
if 'addresses_involved' in row and row['addresses_involved']:
    for addr in row['addresses_involved']:
        cluster_map[addr] = {...}  # Maps address → cluster
```

**Correct approach** should be:
```python
if 'related_alert_ids' in row and row['related_alert_ids']:
    for alert_id in row['related_alert_ids']:
        cluster_map[alert_id] = {...}  # Maps alert_id → cluster
```

## Validation Results

### ✅ Plan Remains Valid

The original architecture plan is **fundamentally correct**:
1. Need to build cluster-level features (not alert-level)
2. Must aggregate alert features to cluster level
3. Output should be one row per cluster

### ✓ Enhanced with Schema Insights

The schema provides additional guidance:

#### A. Pre-aggregated Features Available
The clusters table already contains useful features:
- `total_alerts` - cluster size
- `total_volume_usd` - total cluster volume
- `severity_max` - maximum severity in cluster
- `confidence_avg` - average confidence

**Recommendation:** Use these as base features, then add computed aggregations.

#### B. Proper Alert-Cluster Mapping
Use `related_alert_ids` array to map clusters to their constituent alerts for feature aggregation.

#### C. Address-based Features
Use `addresses_involved` array to aggregate address-level features from `raw_features` table.

## Updated Implementation Strategy

### Phase 1: Build Cluster-Alert Mapping

```python
def _build_cluster_alert_map(
    self,
    clusters_df: pd.DataFrame,
    alerts_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Map cluster_id to list of alert_ids using related_alert_ids."""
    
    cluster_to_alerts = {}
    
    for _, cluster_row in clusters_df.iterrows():
        cluster_id = cluster_row['cluster_id']
        
        # Use related_alert_ids from schema
        if 'related_alert_ids' in cluster_row:
            alert_ids = cluster_row['related_alert_ids']
            if alert_ids:  # Check not empty
                cluster_to_alerts[cluster_id] = alert_ids
    
    return cluster_to_alerts
```

### Phase 2: Feature Aggregation Categories

#### A. Direct from Cluster Table (No Aggregation Needed)
- `total_alerts` → `cluster_size`
- `total_volume_usd` → `cluster_total_volume`
- `severity_max` → `cluster_max_severity_encoded`
- `confidence_avg` → `cluster_avg_confidence`
- `cluster_type` → `cluster_type_encoded`

#### B. Aggregated from Alerts (via related_alert_ids)
For each cluster, aggregate features from its related alerts:
- **Count features:** number of alerts per severity, type, etc.
- **Sum features:** total volumes, transaction counts
- **Mean features:** average confidence, volume, etc.
- **Min/Max features:** range of values
- **Std features:** variability within cluster

#### C. Aggregated from Addresses (via addresses_involved)
For each cluster, aggregate features from involved addresses:
- **Network features:** degree statistics
- **Behavioral features:** anomaly scores
- **Label features:** risk level distribution

#### D. Temporal Features
From alert timestamps:
- Time span of cluster activity
- Alert frequency
- Temporal distribution

### Phase 3: Canonical Cluster Feature Order

```python
canonical_cluster_features = [
    # Base cluster features (from clusters table)
    'cluster_size',                    # total_alerts
    'cluster_total_volume',            # total_volume_usd
    'cluster_max_severity',            # severity_max encoded
    'cluster_avg_confidence',          # confidence_avg
    'cluster_type_encoded',            # cluster_type encoded
    
    # Aggregated alert features
    'alert_count_high_severity',
    'alert_count_critical_severity',
    'alert_volume_sum',
    'alert_volume_mean',
    'alert_volume_std',
    'alert_volume_max',
    'alert_confidence_mean',
    'alert_confidence_std',
    
    # Aggregated address features
    'unique_address_count',            # len(addresses_involved)
    'address_degree_total_sum',
    'address_degree_total_mean',
    'address_anomaly_behavioral_mean',
    'address_anomaly_graph_mean',
    'address_exchange_count',
    'address_mixer_count',
    
    # Temporal features
    'time_span_days',
    'alert_frequency',
    'earliest_timestamp',
    'latest_timestamp',
    
    # Network features
    'total_network_degree',
    'avg_network_degree',
    'network_density',
    
    # Label features
    'high_risk_address_count',
    'labeled_address_ratio'
]
```

## Schema Alignment Validation

### Input Data Flow

```
raw_alerts (86 rows)
    ↓
alerts_df in feature_builder
    ↓
build_inference_features() → Alert Features (86×56)

raw_clusters (25 rows)
    ↓
clusters_df with related_alert_ids
    ↓
build_cluster_features() → Cluster Features (25×N)
    ↑
Uses related_alert_ids to aggregate alert features
```

### Output Data Flow

```
Cluster Features (25×N)
    ↓
score_clusters() → Predictions (25 scores)
    ↓
DataFrame with cluster_id + score (25 rows)
    ↓
cluster_scores table (25 rows inserted)
```

## Updated Implementation Checklist

- [ ] Update `_build_cluster_alert_map()` to use `related_alert_ids`
- [ ] Add helper to extract pre-aggregated features from clusters table
- [ ] Implement alert feature aggregation using cluster-alert mapping
- [ ] Implement address feature aggregation using addresses_involved
- [ ] Add temporal feature extraction
- [ ] Define canonical cluster feature order
- [ ] Update [`risk_scoring.py:212`](packages/scoring/risk_scoring.py:212) to use `build_cluster_features()`
- [ ] Update training pipeline for consistency
- [ ] Validate output dimensions (25 clusters → 25 scores)

## Success Validation

### Data Flow Verification

1. **Cluster table has 25 rows** ✓
2. **related_alert_ids contains alert IDs** ✓
3. **Build cluster features → 25 rows** ✓
4. **Score clusters → 25 scores** ✓
5. **Write to cluster_scores → 25 rows** ✓

### Schema Compliance

- ✓ Uses `related_alert_ids` for alert aggregation
- ✓ Uses `addresses_involved` for address aggregation
- ✓ Uses pre-aggregated cluster features
- ✓ Outputs match `cluster_scores` table schema
- ✓ One row per cluster_id maintained throughout

## Conclusion

The original plan is **valid and enhanced** by schema insights:

1. **Core approach confirmed:** Build cluster-level features by aggregation
2. **Key schema element identified:** `related_alert_ids` for mapping
3. **Pre-aggregated features discovered:** Can use existing cluster table features
4. **Implementation strategy refined:** Use both related_alert_ids and addresses_involved

The plan is **ready for implementation** with these schema-aware enhancements.