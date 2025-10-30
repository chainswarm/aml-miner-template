# SOT Time-Series Validation Strategy

**Date**: 2025-10-30
**Purpose**: Validate miners using SOT's data (including existing address_labels as ground truth!)

---

## Key Insight: SOT Already Has Ground Truth!

### What SOT Contains

```
SOT Database:
├─ raw_features: 98 features per address (time-series)
├─ raw_alerts: Alerts per address (time-series)
├─ raw_clusters: Clusters (time-series)
├─ raw_money_flows: Flows (time-series)
└─ raw_address_labels: CONFIRMED LABELS ⭐ ← WE HAVE THIS!

Example data points for address 0xABC:
├─ 2025-08-01 (195d window): features snapshot
├─ 2025-08-08 (195d window): features snapshot
├─ 2025-08-15 (195d window): features snapshot
└─ 2025-08-22 (195d window): features snapshot
```

**This is a gold mine for validation!** We can track how **98 features** evolve over time!

---

## Critical Question: Should SOT Track Confirmations?

### Current State: NO Confirmation Tracking

```sql
-- raw_alerts schema (current)
CREATE TABLE raw_alerts (
    alert_id String,
    address String,
    severity String,
    -- NO confirmation fields ❌
)
```

### Proposed: ADD Confirmation Tracking

```sql
-- Enhanced raw_alerts schema (proposed)
CREATE TABLE raw_alerts (
    alert_id String,
    address String,
    severity String,
    
    -- NEW: Confirmation tracking ✨
    confirmed_illicit Bool DEFAULT NULL,        -- NULL = unknown, True/False = confirmed
    confirmation_date Date DEFAULT NULL,         -- When confirmed
    confirmation_source String DEFAULT '',       -- How confirmed (SAR, exchange, forensic)
    confirmation_confidence Float32 DEFAULT 0.5, -- Confidence in confirmation
    confirmation_evidence_json String DEFAULT '' -- Evidence details
)
```

### Pros of Adding Confirmation Tracking

**1. Single Source of Truth**
```
✅ All validation data in one place (SOT)
✅ No separate ground_truth tables
✅ Easy to join alerts with confirmations
✅ Historical confirmations preserved
```

**2. Progressive Confirmation**
```sql
-- Day 0: Alert created
INSERT INTO raw_alerts (..., confirmed_illicit=NULL)

-- Day 30: Exchange labels address
UPDATE raw_alerts 
SET confirmed_illicit=True, 
    confirmation_date='2025-09-01',
    confirmation_source='binance_label'
WHERE alert_id='alert_001'

-- Day 90: Additional evidence
UPDATE confirmation_evidence_json
WHERE alert_id='alert_001'
```

**3. Multi-Source Confirmations**
```sql
-- Track multiple confirmations
confirmation_evidence_json = {
  "confirmations": [
    {"source": "exchange_label", "date": "2025-09-01", "confidence": 0.9},
    {"source": "blockchain_forensics", "date": "2025-09-15", "confidence": 0.85},
    {"source": "ofac_sanction", "date": "2025-10-01", "confidence": 1.0}
  ],
  "consensus_confidence": 0.95
}
```

### Cons of Adding Confirmation Tracking

**1. Data Management Complexity**
```
⚠️ Need UPDATE operations (ClickHouse optimized for INSERT)
⚠️ Mutable data (confirmations change over time)
⚠️ Schema evolution (adding fields as sources grow)
```

**2. Alternative: Separate Table**
```sql
-- Keep alerts immutable, add ground_truth table
CREATE TABLE ground_truth (
    alert_id String,
    original_processing_date Date,
    confirmed_illicit Bool,
    confirmation_date Date,
    source String,
    confidence Float32,
    evidence_json String
) ENGINE = ReplacingMergeTree(confirmation_date)
ORDER BY (alert_id, confirmation_date)
```

---

## Recommendation: Hybrid Approach

### Use ReplacingMergeTree for Confirmations

```sql
CREATE TABLE alert_confirmations (
    alert_id String,
    processing_date Date,
    
    -- Confirmation data (updateable)
    confirmed_illicit Bool,
    confirmation_date Date,
    confirmation_source String,
    confirmation_confidence Float32,
    confirmation_evidence_json String,
    
    -- Versioning
    version UInt32,
    updated_at DateTime
)
ENGINE = ReplacingMergeTree(version)
PARTITION BY toYYYYMM(processing_date)
ORDER BY (alert_id, processing_date)
SETTINGS index_granularity = 8192;

-- Query latest confirmations
SELECT alert_id, confirmed_illicit, confirmation_source
FROM alert_confirmations
FINAL  -- Get latest version
WHERE processing_date = '2025-08-01'
```

**Benefits:**
- ✅ Keeps raw_alerts immutable
- ✅ Supports updates via ReplacingMergeTree
- ✅ Historical confirmation tracking
- ✅ Clean separation of concerns

---

## Validation Using SOT Time-Series Features

### Proposal 1: Feature Delta Analysis (BEST FOR SOT!)

**Core Idea**: Track how features change between snapshots for alerted addresses

#### 1.1 Network Growth Metrics

```sql
-- Query feature evolution for address with alert
WITH 
  baseline AS (
    SELECT * FROM raw_features
    WHERE address = '0xABC' 
      AND processing_date = '2025-08-01'  -- Alert date
      AND window_days = 195
  ),
  week_1 AS (
    SELECT * FROM raw_features
    WHERE address = '0xABC'
      AND processing_date = '2025-08-08'  -- +7 days
      AND window_days = 195
  ),
  week_4 AS (
    SELECT * FROM raw_features
    WHERE address = '0xABC'
      AND processing_date = '2025-08-29'  -- +28 days
      AND window_days = 195
  )
SELECT 
  -- Network expansion
  (week_4.degree_total - baseline.degree_total) as degree_delta,
  (week_4.unique_counterparties - baseline.unique_counterparties) as counterparty_delta,
  (week_4.khop1_count - baseline.khop1_count) as khop1_delta,
  
  -- Activity changes
  (week_4.total_volume_usd - baseline.total_volume_usd) as volume_delta,
  (week_4.tx_total_count - baseline.tx_total_count) as tx_delta,
  (week_4.activity_days - baseline.activity_days) as activity_delta,
  
  -- Behavioral changes
  (week_4.behavioral_anomaly_score - baseline.behavioral_anomaly_score) as anomaly_delta,
  (week_4.structuring_score - baseline.structuring_score) as structuring_delta,
  (week_4.suspicious_pattern_score - baseline.suspicious_pattern_score) as pattern_delta,
  
  -- Classification flags changes
  CASE WHEN week_4.is_mixer_like AND NOT baseline.is_mixer_like THEN 1 ELSE 0 END as became_mixer_like,
  CASE WHEN week_4.is_dormant_reactivated THEN 1 ELSE 0 END as became_dormant
FROM baseline, week_1, week_4
```

#### 1.2 Evolution Classification

```python
def classify_feature_evolution(feature_deltas):
    """
    Classify evolution pattern from feature deltas
    """
    
    # Illicit Pattern: Expanding & Escalating
    expanding_illicit = (
        feature_deltas['degree_delta'] > 10 and
        feature_deltas['counterparty_delta'] > 5 and
        feature_deltas['structuring_delta'] > 0.2 and
        feature_deltas['suspicious_pattern_delta'] > 0.15
    )
    
    # Illicit Pattern: Going Dark (detected and stopped)
    going_dark = (
        feature_deltas['activity_delta'] < -5 and
        feature_deltas['tx_delta'] < -10 and
        feature_deltas['became_dormant'] == 1
    )
    
    # Benign Pattern: Stable Normal Activity
    stable_benign = (
        abs(feature_deltas['degree_delta']) < 3 and
        abs(feature_deltas['anomaly_delta']) < 0.1 and
        feature_deltas['structuring_delta'] < 0.05
    )
    
    # Benign Pattern: Legitimate Growth
    legitimate_growth = (
        feature_deltas['volume_delta'] > 0 and
        feature_deltas['tx_delta'] > 0 and
        feature_deltas['structuring_delta'] < 0.1 and
        feature_deltas['became_mixer_like'] == 0
    )
    
    if expanding_illicit:
        return {'pattern': 'expanding_illicit', 'confidence': 0.85}
    elif going_dark:
        return {'pattern': 'going_dark', 'confidence': 0.70}  # Ambiguous
    elif stable_benign or legitimate_growth:
        return {'pattern': 'benign', 'confidence': 0.75}
    else:
        return {'pattern': 'uncertain', 'confidence': 0.50}
```

#### 1.3 Validation Logic

```python
def validate_via_feature_evolution(miner_score, evolution_pattern):
    """
    Validate miner prediction against feature evolution
    """
    
    # Case 1: High prediction + expanding illicit pattern
    if miner_score > 0.7 and evolution_pattern['pattern'] == 'expanding_illicit':
        return {
            'validated': True,
            'score': 0.4,
            'confidence': evolution_pattern['confidence'],
            'reason': 'correctly_predicted_illicit_expansion'
        }
    
    # Case 2: High prediction + going dark
    elif miner_score > 0.7 and evolution_pattern['pattern'] == 'going_dark':
        return {
            'validated': True,
            'score': 0.35,
            'confidence': 0.70,
            'reason': 'possibly_caught_early'  # Could be why it went dark
        }
    
    # Case 3: Low prediction + benign pattern
    elif miner_score < 0.3 and evolution_pattern['pattern'] == 'benign':
        return {
            'validated': True,
            'score': 0.3,
            'confidence': evolution_pattern['confidence'],
            'reason': 'correctly_identified_benign'
        }
    
    # Case 4: Mismatch
    else:
        return {
            'validated': False,
            'score': 0.0,
            'confidence': 0.0,
            'reason': f"mismatch_{evolution_pattern['pattern']}"
        }
```

**Pros:**
- ✅ Uses SOT's 98 features (richest data available!)
- ✅ No external dependencies
- ✅ Quantitative (not subjective)
- ✅ Can validate immediately after 7-30 days
- ✅ Validates private intelligence

**Cons:**
- ⚠️ Requires baseline feature snapshot
- ⚠️ "Going dark" is ambiguous (caught OR naturally ended)
- ⚠️ Need to define thresholds for each metric

---

## Proposal 2: Alert Cluster Evolution

**Track how clusters grow/shrink over time**

### 2.1 Cluster Lifecycle Tracking

```sql
-- Track same cluster across processing dates
WITH cluster_timeline AS (
  SELECT 
    cluster_id,
    processing_date,
    total_alerts,
    total_volume_usd,
    arrayLength(addresses_involved) as num_addresses,
    arrayLength(related_alert_ids) as num_related_alerts,
    severity_max,
    confidence_avg
  FROM raw_clusters
  WHERE cluster_id = 'cluster_001'
    AND window_days = 195
  ORDER BY processing_date
)
SELECT 
  *,
  num_addresses - LAG(num_addresses) OVER (ORDER BY processing_date) as address_growth,
  total_volume_usd - LAG(total_volume_usd) OVER (ORDER BY processing_date) as volume_growth,
  num_related_alerts - LAG(num_related_alerts) OVER (ORDER BY processing_date) as alert_growth
FROM cluster_timeline
```

### 2.2 Cluster Evolution Patterns

```python
def classify_cluster_evolution(cluster_timeline):
    # Growing criminal network
    if (cluster_timeline[-1]['num_addresses'] > cluster_timeline[0]['num_addresses'] * 1.5 and
        cluster_timeline[-1]['total_volume_usd'] > cluster_timeline[0]['total_volume_usd'] * 2.0):
        return {'pattern': 'expanding_network', 'illicit_indicator': 0.9}
    
    # Dormant/Dissolved
    elif (cluster_timeline[-1]['num_addresses'] < cluster_timeline[0]['num_addresses'] * 0.5):
        return {'pattern': 'dissolving', 'illicit_indicator': 0.4}
    
    # Stable cluster
    else:
        return {'pattern': 'stable', 'illicit_indicator': 0.5}
```

**Pros:**
- ✅ Cluster-level validation
- ✅ Uses SOT data only
- ✅ Tracks coordinated activity evolution

**Cons:**
- ⚠️ Requires cluster IDs to persist across dates
- ⚠️ Lower coverage (fewer clusters than alerts)

---

## Proposal 3: Money Flow Pattern Changes

**Track how money flow patterns evolve**

### 3.1 Flow Evolution Query

```sql
-- Compare money flows before and after alert
WITH 
  pre_alert AS (
    SELECT 
      from_address,
      COUNT(DISTINCT to_address) as unique_destinations,
      SUM(amount_usd_sum) as total_outflow,
      AVG(reciprocity_ratio) as avg_reciprocity,
      SUM(CASE WHEN is_bidirectional THEN 1 ELSE 0 END) as bidirectional_count
    FROM raw_money_flows
    WHERE from_address = '0xABC'
      AND processing_date = '2025-08-01'  -- Pre-alert
      AND window_days = 195
    GROUP BY from_address
  ),
  post_alert AS (
    SELECT 
      from_address,
      COUNT(DISTINCT to_address) as unique_destinations,
      SUM(amount_usd_sum) as total_outflow,
      AVG(reciprocity_ratio) as avg_reciprocity,
      SUM(CASE WHEN is_bidirectional THEN 1 ELSE 0 END) as bidirectional_count
    FROM raw_money_flows
    WHERE from_address = '0xABC'
      AND processing_date = '2025-08-29'  -- 28 days post-alert
      AND window_days = 195
    GROUP BY from_address
  )
SELECT 
  post.unique_destinations - pre.unique_destinations as destination_growth,
  post.total_outflow - pre.total_outflow as outflow_increase,
  post.avg_reciprocity - pre.avg_reciprocity as reciprocity_change
FROM pre_alert pre
JOIN post_alert post ON pre.from_address = post.from_address
```

### 3.2 Flow Pattern Classification

```python
def classify_flow_evolution(flow_deltas):
    # Layering signature: Rapid destination increase
    if (flow_deltas['destination_growth'] > 20 and
        flow_deltas['outflow_increase'] > 100000):
        return {'pattern': 'active_layering', 'illicit_indicator': 0.90}
    
    # Structuring: Many small flows
    elif (flow_deltas['destination_growth'] > 10 and
          flow_deltas['avg_amount'] < 1000):
        return {'pattern': 'structuring', 'illicit_indicator': 0.85}
    
    # Normal business
    elif (flow_deltas['destination_growth'] < 5 and
          flow_deltas['reciprocity_change'] > 0):
        return {'pattern': 'legitimate_business', 'illicit_indicator': 0.2}
```

**Pros:**
- ✅ Specific to money laundering patterns
- ✅ Uses money_flows table (detailed flow data)
- ✅ Clear illicit vs benign signatures

**Cons:**
- ⚠️ Requires same addresses across dates
- ⚠️ May miss sophisticated mixing

---

## Proposal 4: Anomaly Score Trajectory

**Track how SOT's anomaly scores evolve**

### 4.1 Anomaly Trend Analysis

```sql
-- Track anomaly score trends
SELECT 
  address,
  processing_date,
  behavioral_anomaly_score,
  graph_anomaly_score,
  neighborhood_anomaly_score,
  global_anomaly_score,
  
  -- Compute trends
  behavioral_anomaly_score - LAG(behavioral_anomaly_score, 1) 
    OVER (PARTITION BY address ORDER BY processing_date) as behavioral_trend_1w,
  
  behavioral_anomaly_score - LAG(behavioral_anomaly_score, 4) 
    OVER (PARTITION BY address ORDER BY processing_date) as behavioral_trend_4w
FROM raw_features
WHERE address IN (SELECT address FROM raw_alerts WHERE processing_date = '2025-08-01')
  AND window_days = 195
ORDER BY address, processing_date
```

### 4.2 Trend-Based Validation

```python
def validate_via_anomaly_trends(miner_score, anomaly_trends):
    # Increasing anomaly = Illicit behavior escalating
    if anomaly_trends['behavioral_trend_4w'] > 0.2:
        if miner_score > 0.7:
            return {'validated': True, 'score': 0.35}
    
    # Decreasing anomaly = Behavior normalizing
    elif anomaly_trends['behavioral_trend_4w'] < -0.2:
        if miner_score < 0.3:
            return {'validated': True, 'score': 0.30}
    
    # Stable anomaly
    else:
        return {'validated': False, 'score': 0.0}
```

**Pros:**
- ✅ Uses SOT's own anomaly detection
- ✅ 4 different anomaly types to track
- ✅ Trend is objective measurement

**Cons:**
- ⚠️ Circular dependency (validating with SOT's own scores)
- ⚠️ May not detect what SOT misses

---

## Proposal 5: New Alert Generation Rate

**Track if address generates MORE alerts over time**

### 5.1 Alert Accumulation Query

```sql
-- Count alerts per address over time
WITH alert_counts AS (
  SELECT 
    address,
    processing_date,
    COUNT(*) as num_alerts,
    SUM(CASE WHEN severity IN ('high', 'critical') THEN 1 ELSE 0 END) as high_severity_count
  FROM raw_alerts
  WHERE window_days = 195
  GROUP BY address, processing_date
)
SELECT 
  address,
  processing_date,
  num_alerts,
  num_alerts - LAG(num_alerts, 1) OVER (PARTITION BY address ORDER BY processing_date) as new_alerts_1w,
  num_alerts - LAG(num_alerts, 4) OVER (PARTITION BY address ORDER BY processing_date) as new_alerts_4w
FROM alert_counts
WHERE address IN (SELECT address FROM raw_alerts WHERE processing_date = '2025-08-01')
ORDER BY address, processing_date
```

### 5.2 Validation

```python
def validate_via_alert_accumulation(miner_score, alert_accumulation):
    # More alerts = Escalating risk
    if alert_accumulation['new_alerts_4w'] > 3:
        if miner_score > 0.7:
            return {'validated': True, 'score': 0.4, 
                   'reason': 'correctly_predicted_escalation'}
    
    # No new alerts = Maybe was false positive OR caught
    elif alert_accumulation['new_alerts_4w'] == 0:
        if miner_score < 0.3:
            return {'validated': True, 'score': 0.25,
                   'reason': 'correctly_identified_isolated'}
        elif miner_score > 0.7:
            return {'validated': True, 'score': 0.20,
                   'reason': 'possibly_caught_early'}
```

**Pros:**
- ✅ Simple query
- ✅ Clear signal (more alerts = worse)
- ✅ Addresses SOT already tracking

**Cons:**
- ⚠️ Depends on SOT generating new alerts
- ⚠️ No new alerts could mean caught OR benign

---

## Proposal 6: Classification Flag Transitions

**Track binary classification changes**

### 6.1 Flag Evolution Query

```sql
-- Track when addresses become classified as risky types
SELECT 
  address,
  processing_date,
  
  -- Classification flags
  is_mixer_like,
  is_whale,
  is_exchange_like,
  is_contract_like,
  is_dormant_reactivated,
  
  -- Detect transitions
  CASE WHEN is_mixer_like AND NOT LAG(is_mixer_like) OVER w THEN 1 ELSE 0 END as became_mixer,
  CASE WHEN NOT is_whale AND LAG(is_whale) OVER w THEN 1 ELSE 0 END as stopped_being_whale,
  CASE WHEN is_dormant_reactivated THEN 1 ELSE 0 END as reactivated
FROM raw_features
WHERE address IN (SELECT address FROM raw_alerts WHERE processing_date = '2025-08-01')
  AND window_days = 195
WINDOW w AS (PARTITION BY address ORDER BY processing_date)
ORDER BY address, processing_date
```

### 6.2 Validation

```python
def validate_via_flag_transitions(miner_score, flag_changes):
    # Became mixer-like = Very suspicious
    if flag_changes['became_mixer'] == 1:
        if miner_score > 0.8:
            return {'validated': True, 'score': 0.45}  # Strong validation
    
    # Dormant reactivation = Suspicious
    elif flag_changes['reactivated'] == 1:
        if miner_score > 0.6:
            return {'validated': True, 'score': 0.35}
    
    # No suspicious transitions
    elif sum(flag_changes.values()) == 0:
        if miner_score < 0.3:
            return {'validated': True, 'score': 0.30}
```

**Pros:**
- ✅ Clear binary signals
- ✅ Based on SOT's own classifications
- ✅ Easy to interpret

**Cons:**
- ⚠️ Limited to SOT's classification ability
- ⚠️ May miss subtle changes

---

## Proposal 7: Multi-Feature Composite Score

**Create composite evolution score from multiple features**

### 7.1 Weighted Feature Changes

```python
def compute_evolution_risk_score(feature_deltas):
    """
    Weighted combination of feature changes
    """
    
    weights = {
        # Network expansion (20%)
        'degree_delta': 0.08,
        'counterparty_delta': 0.07,
        'khop1_delta': 0.05,
        
        # Activity changes (20%)
        'volume_delta': 0.10,
        'tx_delta': 0.10,
        
        # Behavioral changes (30%)
        'anomaly_delta': 0.10,
        'structuring_delta': 0.10,
        'suspicious_pattern_delta': 0.10,
        
        # Classification changes (30%)
        'became_mixer_like': 0.15,
        'became_dormant': -0.15  # Negative = less risky
    }
    
    # Normalize deltas and apply weights
    evolution_score = sum(
        weights[feature] * normalize(feature_deltas[feature])
        for feature in weights
    )
    
    return evolution_score  # Range: [0, 1]
```

### 7.2 Validation

```python
def validate_via_composite_score(miner_score, evolution_score):
    # Compute agreement
    agreement = 1.0 - abs(miner_score - evolution_score)
    
    # High agreement = Good prediction
    if agreement > 0.7:
        return {'validated': True, 'score': 0.4 * agreement}
    else:
        return {'validated': False, 'score': 0.0}
```

**Pros:**
- ✅ Comprehensive (uses all 98 features)
- ✅ Continuous validation score
- ✅ Captures subtle changes

**Cons:**
- ⚠️ Complex weight tuning
- ⚠️ May overfit to specific patterns

---

## Proposal 8: Cross-Address Comparison

**Compare alerted address evolution vs similar non-alerted addresses**

### 8.1 Control Group Matching

```python
def find_control_group(alerted_address, all_addresses):
    """
    Find similar addresses that were NOT alerted
    """
    # Get baseline features of alerted address
    baseline = get_features(alerted_address, alert_date)
    
    # Find non-alerted addresses with similar profiles
    similar_addresses = []
    for addr in all_addresses:
        if addr not in alerted_addresses:
            addr_features = get_features(addr, alert_date)
            
            # Compute similarity
            similarity = cosine_similarity(baseline, addr_features)
            
            if similarity > 0.9:
                similar_addresses.append(addr)
    
    return similar_addresses[:10]  # Top 10 similar
```

### 8.2 Comparative Evolution

```python
def validate_via_control_comparison(alerted_address, control_group):
    # Track evolution for both
    alerted_evolution = track_evolution(alerted_address, days=30)
    control_evolution = [track_evolution(addr, days=30) for addr in control_group]
    
    # Compute relative change
    alerted_risk_increase = alerted_evolution['composite_risk_score']
    control_avg_risk_increase = mean([e['composite_risk_score'] for e in control_evolution])
    
    # Alerted address should diverge from control group if truly risky
    if alerted_risk_increase > control_avg_risk_increase + 0.3:
        return {'validated': True, 'score': 0.40, 'reason': 'diverged_from_control'}
    
    # Alerted address similar to control group = Maybe false positive
    elif abs(alerted_risk_increase - control_avg_risk_increase) < 0.1:
        return {'validated': False, 'score': 0.0, 'reason': 'similar_to_control'}
```

**Pros:**
- ✅ Scientific approach (control group)
- ✅ Differentiates real risk from noise
- ✅ Uses only SOT data

**Cons:**
- ⚠️ Requires many addresses for matching
- ⚠️ Complex to find good controls
- ⚠️ Assumes similar baseline = similar evolution

---

## Proposal 9: Feature Velocity Analysis

**Measure RATE of change, not just change**

### 9.1 Velocity Computation

```python
def compute_feature_velocities(address, dates):
    """
    Compute rate of change for key features
    """
    features_timeline = [get_features(address, date) for date in dates]
    
    velocities = {}
    for feature in ['degree_total', 'total_volume_usd', 'behavioral_anomaly_score']:
        values = [f[feature] for f in features_timeline]
        
        # Compute velocity (change per week)
        deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
        velocity = mean(deltas)
        
        # Compute acceleration (change in velocity)
        if len(deltas) > 1:
            delta_deltas = [deltas[i+1] - deltas[i] for i in range(len(deltas)-1)]
            acceleration = mean(delta_deltas)
        else:
            acceleration = 0
        
        velocities[feature] = {
            'velocity': velocity,
            'acceleration': acceleration
        }
    
    return velocities
```

### 9.2 Validation

```python
def validate_via_velocities(miner_score, velocities):
    # Accelerating risk (not just growing, but growing FASTER)
    degree_accel = velocities['degree_total']['acceleration']
    anomaly_accel = velocities['behavioral_anomaly_score']['acceleration']
    
    if degree_accel > 0 and anomaly_accel > 0:
        # Accelerating network + anomaly = Escalating illicit
        if miner_score > 0.8:
            return {'validated': True, 'score': 0.45}
    
    # Decelerating (slowing down)
    elif degree_accel < 0 and anomaly_accel < 0:
        if miner_score < 0.4:
            return {'validated': True, 'score': 0.30}
```

**Pros:**
- ✅ Captures acceleration (more sophisticated)
- ✅ Differentiates explosive growth from gradual
- ✅ Uses time-series nature of SOT

**Cons:**
- ⚠️ Requires 3+ data points
- ⚠️ Noisy on short timescales

---

## Proposal 10: Ensemble Consistency Validation

**Use SOT feature ensemble to validate predictions**

### 10.1 SOT's Own Risk Indicators

```python
def compute_sot_ensemble_score(features):
    """
    Use SOT's existing risk indicators as ensemble
    """
    # SOT has multiple anomaly scores
    ensemble = {
        'behavioral_anomaly_score': 0.25,  # Weight
        'graph_anomaly_score': 0.25,
        'neighborhood_anomaly_score': 0.20,
        'global_anomaly_score': 0.15,
        'suspicious_pattern_score': 0.15
    }
    
    sot_score = sum(
        features[metric] * weight 
        for metric, weight in ensemble.items()
    )
    
    return sot_score
```

### 10.2 Validation Logic

```python
def validate_via_sot_ensemble(miner_score, sot_ensemble_score):
    # Track if SOT's own ensemble changes
    sot_delta = sot_ensemble_score['week_4'] - sot_ensemble_score['baseline']
    
    # Miner prediction aligns with SOT evolution
    if miner_score > 0.7 and sot_delta > 0.2:
        return {'validated': True, 'score': 0.35}
    
    elif miner_score < 0.3 and sot_delta < -0.1:
        return {'validated': True, 'score': 0.30}
```

**Pros:**
- ✅ Uses SOT's multi-score consensus
- ✅ Leverages SOT's sophisticated detection
- ✅ Objective (SOT's own evolution)

**Cons:**
- ⚠️ Limited to what SOT can detect
- ⚠️ Won't validate superior private intelligence

---

## Recommended SOT-Based Validation Stack

### Tier 1: Immediate (Day 0) - 0.20 pts
```python
# No SOT evolution needed
- Integrity checks (schema, completeness)
- Pattern traps (synthetic tests)
```

### Tier 2: Short-Term Evolution (Day 7-14) - 0.30 pts
```python
# Use 1-2 week SOT feature evolution
validation = {
  'flag_transitions': 0.15,        # is_mixer_like, etc.
  'anomaly_trends': 0.15          # Anomaly score direction
}
```

### Tier 3: Long-Term Evolution (Day 30) - 0.50 pts
```python
# Use 4+ week SOT feature evolution
validation = {
  'feature_delta_analysis': 0.20,    # 98 features ⭐ RICHEST
  'cluster_evolution': 0.15,         # Cluster growth/shrink
  'money_flow_patterns': 0.15        # Flow pattern changes
}
```

---

## Answering Your Question: Should SOT Track Confirmations?

### YES - Add Confirmation Tracking

**Recommended Schema Addition:**

```sql
CREATE TABLE alert_confirmations (
    alert_id String,
    address String,
    processing_date Date,
    
    -- Confirmation data
    confirmed_illicit Bool DEFAULT NULL,      -- NULL=unknown, True/False=confirmed
    confirmation_date Date DEFAULT NULL,       -- When confirmed
    confirmation_source String DEFAULT '',     -- Source (SAR, exchange, forensic, evolution)
    confirmation_method String DEFAULT '',     -- How (external_gt, feature_evolution, network_propagation)
    confirmation_confidence Float32 DEFAULT 0.5,
    confirmation_evidence_json String DEFAULT '',
    
    -- Allow updates
    version UInt32,
    updated_at DateTime
)
ENGINE = ReplacingMergeTree(version)
PARTITION BY toYYYYMM(processing_date)
ORDER BY (alert_id, processing_date, version)
SETTINGS index_granularity = 8192;
```

### Why Track Confirmations in SOT?

**1. Multiple Confirmation Sources**
```python
# Confirmation can come from:
confirmations = {
  'external': ['SAR', 'exchange_label', 'ofac_sanction'],
  'internal': ['feature_evolution', 'cluster_growth', 'network_propagation'],
  'computed': ['anomaly_trend', 'flow_pattern', 'flag_transition']
}

# Store ALL confirmations:
alert_confirmations = [
  {'alert_001', confirmed=True, source='feature_evolution', confidence=0.80, date='2025-08-29'},
  {'alert_001', confirmed=True, source='exchange_label', confidence=0.95, date='2025-09-15'},
  {'alert_001', confirmed=True, source='sar_filing', confidence=1.00, date='2025-10-01'}
]

# Consensus confidence increases over time!
```

**2. Progressive Validation**
```
Week 1: Use feature evolution (confidence=0.75)
Week 2: Add cluster growth (confidence=0.80)
Week 4: Add network propagation (confidence=0.85)
Week 8: Add exchange label (confidence=0.95)
Week 12: Add SAR filing (confidence=1.00)
```

**3. Validation Independence**
```
SOT can confirm alerts EVEN IF no external GT arrives:
├─ Feature evolution shows expanding network
├─ Cluster grows from 5 to 40 addresses
├─ Flow patterns show layering
└─ Confirmed internally with 0.85 confidence

No SAR needed! ✅
```

---

## Practical Implementation

### Query Templates for Validation

#### Template 1: Feature Evolution Validation
```sql
-- Get feature deltas for alert validation
WITH 
  alert_addresses AS (
    SELECT DISTINCT address, processing_date as alert_date
    FROM raw_alerts
    WHERE processing_date = '2025-08-01' AND window_days = 195
  ),
  baseline_features AS (
    SELECT f.*
    FROM raw_features f
    JOIN alert_addresses a ON f.address = a.address 
      AND f.processing_date = a.alert_date
      AND f.window_days = 195
  ),
  evolved_features AS (
    SELECT f.*
    FROM raw_features f
    JOIN alert_addresses a ON f.address = a.address
      AND f.processing_date = toDate(a.alert_date + INTERVAL 28 DAY)
      AND f.window_days = 195
  )
SELECT 
  b.address,
  
  -- Network deltas
  e.degree_total - b.degree_total as degree_delta,
  e.unique_counterparties - b.unique_counterparties as counterparty_delta,
  
  -- Activity deltas
  e.total_volume_usd - b.total_volume_usd as volume_delta,
  e.tx_total_count - b.tx_total_count as tx_delta,
  
  -- Anomaly deltas
  e.behavioral_anomaly_score - b.behavioral_anomaly_score as anomaly_delta,
  e.structuring_score - b.structuring_score as structuring_delta,
  
  -- Classification changes
  CASE WHEN e.is_mixer_like AND NOT b.is_mixer_like THEN 1 ELSE 0 END as became_mixer
  
FROM baseline_features b
JOIN evolved_features e ON b.address = e.address
```

#### Template 2: Cluster Evolution Validation
```sql
-- Track cluster evolution
SELECT 
  cluster_id,
  processing_date,
  total_alerts,
  total_volume_usd,
  arrayLength(addresses_involved) as cluster_size,
  
  -- Compute growth
  cluster_size - LAG(cluster_size, 1) OVER (
    PARTITION BY cluster_id ORDER BY processing_date
  ) as size_delta_1w,
  
  total_volume_usd - LAG(total_volume_usd, 1) OVER (
    PARTITION BY cluster_id ORDER BY processing_date
  ) as volume_delta_1w
  
FROM raw_clusters
WHERE window_days = 195
  AND cluster_id IN (
    SELECT cluster_id FROM raw_clusters 
    WHERE processing_date = '2025-08-01'
  )
ORDER BY cluster_id, processing_date
```

---

## Final Recommendations

### For Subnet Cold Start

**Phase 1 (Months 1-3): SOT Evolution Validation**
```python
validation_score = {
  'immediate': 0.20,                    # Integrity + traps
  'feature_evolution': 0.40,            # 98 features ⭐ PRIMARY
  'cluster_evolution': 0.20,            # Cluster growth
  'flow_pattern_evolution': 0.20        # Money flow changes
}
# Total: 1.0 - No external GT needed!
```

**Phase 2 (Months 4+): Add External GT When Available**
```python
validation_score = {
  'immediate': 0.10,
  'sot_evolution': 0.40,              # Keep using SOT
  'external_confirmations': 0.50      # SAR, exchange labels (when available)
}
```

### Schema Changes Needed

**Add to SOT:**
```sql
CREATE TABLE alert_confirmations (
    alert_id String,
    processing_date Date,
    confirmed_illicit Bool DEFAULT NULL,
    confirmation_date Date DEFAULT NULL,
    confirmation_source String DEFAULT '',  -- 'feature_evolution', 'sar', 'exchange', etc.
    confirmation_method String DEFAULT '',  -- 'automated', 'manual', 'external'
    confidence Float32 DEFAULT 0.5,
    evidence_json String DEFAULT '',
    version UInt32,
    updated_at DateTime
) ENGINE = ReplacingMergeTree(version)
ORDER BY (alert_id, processing_date, version);
```

**Benefits:**
- ✅ Track confirmations from ANY source
- ✅ Distinguish internal (evolution) vs external (SAR) confirmations
- ✅ Support progressive confidence increases
- ✅ Enable validation even without external GT

---

## Summary

### Key Insights

1. **SOT has 98 time-series features** - richest validation data possible!
2. **Feature evolution reveals truth** - illicit behavior escalates, benign stabilizes
3. **No external GT needed** - SOT's own evolution is ground truth
4. **Add confirmation tracking** - Store ALL confirmations (internal + external)

### Best Validation Approach

**Primary (0.40 pts): Feature Delta Analysis**
- Track 98 features across processing_dates
- Compute deltas (degree, volume, anomaly, etc.)
- Classify evolution pattern (expanding/dormant/stable)
- Validate miner prediction vs pattern

**Secondary (0.20 pts): Cluster Evolution**
- Track cluster size/volume changes
- Growing clusters = illicit confirmation

**Tertiary (0.20 pts): Flow Pattern Evolution**
- Track money_flows changes
- Layering/mixing patterns = illicit confirmation

**Bonus (0.20 pts): External GT**
- When SAR/labels arrive, add confirmation
- Highest confidence source

### Answer to "Do We Confirm Alerts in SOT?"

**YES - We should!** Create `alert_confirmations` table to track:
- Internal confirmations (feature evolution, cluster growth) - confidence 0.70-0.85
- External confirmations (SAR, exchange, OFAC) - confidence 0.90-1.00
- Progressive updates as more evidence accumulates

**This makes SOT the single source of truth for both alerts AND their outcomes!**
# ⭐ CRITICAL: SOT Already Has Ground Truth via Address Labels!

**We don't need to wait for SAR or build ground truth - we have `raw_address_labels` table!**

```sql
-- SOT's existing ground truth
SELECT address, label, risk_level, confidence_score, source
FROM raw_address_labels
WHERE processing_date = '2025-08-01'
  AND window_days = 195

-- Coverage: Typically 5-15% of addresses
-- Sources: Exchanges, Chainalysis, OFAC, community labels
-- Confidence: 0.70-0.95 depending on source
```

**Immediate validation possible:**
```python
# Join alerts with address_labels for instant ground truth
labeled_alerts = alerts.merge(address_labels, on='address')

# Binary ground truth from risk_level
labeled_alerts['confirmed_illicit'] = labeled_alerts['risk_level'].isin(['high', 'critical'])

# Validate miners immediately
auc_roc = roc_auc_score(
    y_true=labeled_alerts['confirmed_illicit'],
    y_pred=miner_scores[labeled_alerts.alert_id]
)

# Ground truth validation works from DAY 0! ✅
```

---
