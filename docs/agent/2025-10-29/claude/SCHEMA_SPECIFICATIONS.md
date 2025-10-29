# Schema Specifications for Raw Tables

**Date:** 2025-10-29  
**Purpose:** Define ClickHouse schema for raw data tables aligned with source system

## Overview

These schemas are based on the source system's `analyzers_*` and `core_money_flows_view` tables, with modifications for our ingestion system:

1. **Remove `_version`** - ClickHouse handles versioning differently
2. **Use MergeTree** - Instead of ReplacingMergeTree  
3. **Simplify indexes** - Only essential indexes for our use case
4. **String severity** - Use String instead of Enum for flexibility

## Schema Files to Create/Update

### 1. raw_features.sql

**Location:** `packages/storage/schema/raw_features.sql`  
**Replace existing file completely**

```sql
CREATE TABLE IF NOT EXISTS raw_features (
    window_days UInt16,
    processing_date Date,
    address String,
    
    degree_in UInt32,
    degree_out UInt32,
    degree_total UInt32,
    unique_counterparties UInt32,
    
    total_in_usd Decimal128(18),
    total_out_usd Decimal128(18),
    net_flow_usd Decimal128(18),
    total_volume_usd Decimal128(18),
    avg_tx_in_usd Decimal128(18),
    avg_tx_out_usd Decimal128(18),
    median_tx_in_usd Decimal128(18),
    median_tx_out_usd Decimal128(18),
    max_tx_usd Decimal128(18),
    min_tx_usd Decimal128(18),
    
    amount_variance Float64,
    amount_skewness Float64,
    amount_kurtosis Float64,
    volume_std Float64,
    volume_cv Float64,
    flow_concentration Float64,
    
    tx_in_count UInt64,
    tx_out_count UInt64,
    tx_total_count UInt64,
    
    activity_days UInt32,
    activity_span_days UInt32,
    avg_daily_volume_usd Decimal128(18),
    peak_hour UInt8,
    peak_day UInt8,
    regularity_score Float32,
    burst_factor Float32,
    
    reciprocity_ratio Float32,
    flow_diversity Float32,
    counterparty_concentration Float32,
    concentration_ratio Float32,
    velocity_score Float32,
    structuring_score Float32,
    
    unique_assets_in UInt32,
    unique_assets_out UInt32,
    dominant_asset_in String,
    dominant_asset_out String,
    asset_diversity_score Float32,
    
    hourly_activity Array(UInt16),
    daily_activity Array(UInt16),
    peak_activity_hour UInt8,
    peak_activity_day UInt8,
    hourly_entropy Float32,
    daily_entropy Float32,
    weekend_transaction_ratio Float32,
    night_transaction_ratio Float32,
    small_transaction_ratio Float32,
    consistency_score Float32,
    
    pagerank Float32,
    betweenness Float32,
    closeness Float32,
    clustering_coefficient Float32,
    kcore UInt32,
    community_id UInt32,
    centrality_score Float32,
    
    khop1_count UInt32,
    khop2_count UInt32,
    khop3_count UInt32,
    khop1_volume_usd Decimal128(18),
    khop2_volume_usd Decimal128(18),
    khop3_volume_usd Decimal128(18),
    
    flow_reciprocity_entropy Float32,
    counterparty_stability Float32,
    flow_burstiness Float32,
    transaction_regularity Float32,
    amount_predictability Float32,
    
    behavioral_anomaly_score Float32,
    graph_anomaly_score Float32,
    neighborhood_anomaly_score Float32,
    global_anomaly_score Float32,
    outlier_transactions UInt32,
    suspicious_pattern_score Float32,
    
    is_exchange_like Boolean,
    is_whale Boolean,
    is_mixer_like Boolean,
    is_contract_like Boolean,
    is_new_address Boolean,
    is_dormant_reactivated Boolean,
    is_high_volume_trader Boolean,
    is_hub_address Boolean,
    is_retail_active Boolean,
    is_whale_inactive Boolean,
    is_retail_inactive Boolean,
    is_regular_user Boolean,
    
    unique_recipients_count UInt32,
    unique_senders_count UInt32,
    
    completeness_score Float32,
    quality_score Float32,
    outlier_score Float32,
    confidence_score Float32,
    
    first_activity_timestamp UInt64,
    last_activity_timestamp UInt64,
    
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY (window_days, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, address)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_features(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_features(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_address ON raw_features(address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_total_volume_usd ON raw_features(total_volume_usd) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_degree_total ON raw_features(degree_total) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_is_whale ON raw_features(is_whale) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_is_exchange_like ON raw_features(is_exchange_like) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_behavioral_anomaly ON raw_features(behavioral_anomaly_score) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_graph_anomaly ON raw_features(graph_anomaly_score) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_neighborhood_anomaly ON raw_features(neighborhood_anomaly_score) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_global_anomaly ON raw_features(global_anomaly_score) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_quality_score ON raw_features(quality_score) TYPE minmax GRANULARITY 4;
```

**Changes from original:**
- Changed from long format (4 cols) to wide format (96+ cols)
- Added all feature columns from source system
- Removed `_version` column
- Changed engine from ReplacingMergeTree to MergeTree
- Added `created_at` for audit trail

---

### 2. raw_alerts.sql

**Location:** `packages/storage/schema/raw_alerts.sql`  
**Update existing file**

```sql
CREATE TABLE IF NOT EXISTS raw_alerts (
    window_days UInt16,
    processing_date Date,
    alert_id String,
    address String,
    
    typology_type String,
    pattern_id String DEFAULT '',
    pattern_type String DEFAULT '',
    
    severity String DEFAULT 'medium',
    suspected_address_type String DEFAULT 'unknown',
    suspected_address_subtype String DEFAULT '',
    alert_confidence_score Float32,
    description String,
    volume_usd Decimal128(18) DEFAULT 0,
    
    evidence_json String,
    risk_indicators Array(String),
    
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (window_days, processing_date, alert_id, typology_type)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_alerts(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_alerts(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_address ON raw_alerts(address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_severity ON raw_alerts(severity) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_typology_type ON raw_alerts(typology_type) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_suspected_address_type ON raw_alerts(suspected_address_type) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_suspected_address_subtype ON raw_alerts(suspected_address_subtype) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_alert_id ON raw_alerts(alert_id) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_pattern_id ON raw_alerts(pattern_id) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_pattern_type ON raw_alerts(pattern_type) TYPE bloom_filter(0.01) GRANULARITY 4;
```

**Changes from original:**
- Changed `severity` from `Enum8` to `String`
- Changed data types to match source (Float32 for confidence, Decimal128 for volume)
- Removed `_version` column
- Changed engine from ReplacingMergeTree to MergeTree
- Updated ORDER BY to match source system

---

### 3. raw_clusters.sql

**Location:** `packages/storage/schema/raw_clusters.sql`  
**Update existing file**

```sql
CREATE TABLE IF NOT EXISTS raw_clusters (
    window_days UInt16,
    processing_date Date,
    cluster_id String,
    cluster_type String,
    
    primary_address String DEFAULT '',
    pattern_id String DEFAULT '',
    
    primary_alert_id String,
    related_alert_ids Array(String),
    addresses_involved Array(String),
    
    total_alerts UInt32,
    total_volume_usd Decimal128(18),
    severity_max String DEFAULT 'medium',
    confidence_avg Float32,
    
    earliest_alert_timestamp UInt64,
    latest_alert_timestamp UInt64,
    
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY (cluster_type, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, cluster_type, cluster_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_clusters(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_clusters(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_cluster_type ON raw_clusters(cluster_type) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_severity_max ON raw_clusters(severity_max) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_primary_alert_id ON raw_clusters(primary_alert_id) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_cluster_id ON raw_clusters(cluster_id) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_primary_address ON raw_clusters(primary_address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_pattern_id ON raw_clusters(pattern_id) TYPE bloom_filter(0.01) GRANULARITY 4;
```

**Changes from original:**
- Changed `severity_max` from `Enum8` to `String`
- Changed data types to match source (Decimal128 for volume, Float32 for confidence)
- Removed `_version` column
- Changed engine from ReplacingMergeTree to MergeTree

---

### 4. raw_money_flows.sql

**Location:** `packages/storage/schema/raw_money_flows.sql`  
**New file - create this**

```sql
CREATE TABLE IF NOT EXISTS raw_money_flows (
    window_days UInt16,
    processing_date Date,
    
    from_address String,
    to_address String,
    
    tx_count UInt64,
    amount_sum Decimal128(18),
    amount_usd_sum Decimal128(18),
    
    first_seen_timestamp UInt64,
    last_seen_timestamp UInt64,
    active_days UInt32,
    
    avg_tx_size_usd Decimal128(18),
    unique_assets UInt32,
    dominant_asset String,
    
    hourly_pattern Array(UInt16),
    weekly_pattern Array(UInt16),
    
    reciprocity_ratio Float32,
    is_bidirectional Boolean,
    
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY (window_days, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, from_address, to_address)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_money_flows(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_money_flows(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_from_address ON raw_money_flows(from_address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_to_address ON raw_money_flows(to_address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_amount_usd_sum ON raw_money_flows(amount_usd_sum) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_tx_count ON raw_money_flows(tx_count) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_is_bidirectional ON raw_money_flows(is_bidirectional) TYPE minmax GRANULARITY 4;
```

**Notes:**
- New table based on `core_money_flows_view` from source
- Source system will export view data to parquet
- Added `window_days` and `processing_date` for consistency
- Stores aggregated money flow data between address pairs

---

## Migration Guide

### For Existing Databases

```sql
-- Drop existing tables (data will be lost - acceptable for new system)
DROP TABLE IF EXISTS raw_features;
DROP TABLE IF EXISTS raw_alerts;
DROP TABLE IF EXISTS raw_clusters;

-- Create new tables using updated schemas above
-- Then run ingestion
```

### For New Deployments

```bash
# Schema files will be automatically applied by MigrateSchema
# No manual intervention needed
```

## Validation Checklist

After creating/updating schema files, verify:

- [ ] All column names match source system (except `_version`)
- [ ] All data types match source system
- [ ] `severity` fields use String not Enum
- [ ] All tables have `window_days` and `processing_date`
- [ ] All tables have `created_at` for audit
- [ ] Indexes created for commonly queried fields
- [ ] Partition keys appropriate for query patterns

## Expected Ingestion Volumes

| Table | Rows (Example) | Columns | Size |
|-------|----------------|---------|------|
| raw_alerts | 86 | 16 | ~0.02 MB |
| raw_features | 715 | 96 | ~0.24 MB |
| raw_clusters | 25 | 17 | ~0.02 MB |
| raw_money_flows | 842 | 19 | ~0.09 MB |

**Total:** ~0.37 MB per batch (will scale with network activity)

## Next Steps

1. **Code Mode:** Create/update the 4 schema files
2. **Code Mode:** Update ingestion script to handle 4 tables
3. **Code Mode:** Add money_flows to validation logic
4. **Test:** Verify schemas work with real S3 data

---

**Important:** These schemas must be created before the next ingestion run. The source system will generate parquet files matching these schemas.