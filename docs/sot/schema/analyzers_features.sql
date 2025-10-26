-- Phase 3A Feature Tables - ML Feature Engineering Infrastructure
-- Generated for Phase 3A: Foundation Infrastructure for ML Feature Builder

-- =============================================================================
-- analyzers_features: Comprehensive ML features for graph analysis
-- =============================================================================

CREATE OR REPLACE TABLE analyzers_features (
    -- Time series dimensions
    window_days UInt16,
    processing_date Date,

    -- Primary identifier
    address String,

    -- Node topology features (from address panel data)
    degree_in UInt32,                   -- Number of unique senders
    degree_out UInt32,                  -- Number of unique receivers
    degree_total UInt32,                -- Total degree (in + out)
    unique_counterparties UInt32,       -- Distinct addresses interacted with

    -- Volume features (USD normalized)
    total_in_usd Decimal128(18),        -- Total incoming value
    total_out_usd Decimal128(18),       -- Total outgoing value
    net_flow_usd Decimal128(18),        -- Net flow (in - out)
    total_volume_usd Decimal128(18),    -- Total volume (in + out)
    avg_tx_in_usd Decimal128(18),       -- Average incoming transaction size
    avg_tx_out_usd Decimal128(18),      -- Average outgoing transaction size
    median_tx_in_usd Decimal128(18),    -- Median incoming transaction size
    median_tx_out_usd Decimal128(18),   -- Median outgoing transaction size
    max_tx_usd Decimal128(18),          -- Maximum single transaction value
    min_tx_usd Decimal128(18),          -- Minimum single transaction value

    -- Statistical distribution features
    amount_variance Float64,            -- Variance of transaction amounts
    amount_skewness Float64,            -- Skewness of transaction amounts
    amount_kurtosis Float64,            -- Kurtosis of transaction amounts
    volume_std Float64,                 -- Standard deviation of volumes
    volume_cv Float64,                  -- Coefficient of variation
    flow_concentration Float64,         -- Gini coefficient of flow distribution

    -- Transaction count features
    tx_in_count UInt64,                 -- Total incoming transactions
    tx_out_count UInt64,                -- Total outgoing transactions
    tx_total_count UInt64,              -- Total transaction count

    -- Temporal features
    activity_days UInt32,               -- Days with activity in window
    activity_span_days UInt32,          -- Days between first and last activity
    avg_daily_volume_usd Decimal128(18), -- Average daily volume
    peak_hour UInt8,                    -- Most active hour (0-23)
    peak_day UInt8,                     -- Most active day (0-6, Mon=0)
    regularity_score Float32,           -- Activity regularity score (0-1)
    burst_factor Float32,               -- Ratio of peak to average activity

    -- Flow characteristics (enhanced with ADDRESS_PANEL data)
    reciprocity_ratio Float32,          -- Bidirectional flow ratio (0-1)
    flow_diversity Float32,             -- Shannon entropy of flow distribution
    counterparty_concentration Float32, -- Concentration of counterparty interactions
    concentration_ratio Float32,        -- Gini coefficient of counterparty distribution
    velocity_score Float32,             -- Transaction frequency score (0-1)
    structuring_score Float32,          -- Small transaction clustering score (0-1)

    -- Asset diversity features (enhanced with ADDRESS_PANEL data)
    unique_assets_in UInt32,            -- Different assets received
    unique_assets_out UInt32,           -- Different assets sent
    dominant_asset_in String,           -- Most received asset (by USD)
    dominant_asset_out String,          -- Most sent asset (by USD)
    asset_diversity_score Float32,      -- Asset interaction diversity

    -- Behavioral pattern features (enhanced with ADDRESS_PANEL data)
    hourly_activity Array(UInt16),      -- 24-hour activity histogram [0-23]
    daily_activity Array(UInt16),       -- 7-day activity histogram [0-6]
    peak_activity_hour UInt8,           -- Most active hour (0-23)
    peak_activity_day UInt8,            -- Most active day (0-6, Mon=0)
    hourly_entropy Float32,             -- Entropy of hourly activity pattern
    daily_entropy Float32,              -- Entropy of daily activity pattern
    weekend_transaction_ratio Float32,  -- Weekend vs weekday activity ratio
    night_transaction_ratio Float32,    -- Night vs day activity ratio
    small_transaction_ratio Float32,    -- Small transaction clustering ratio
    consistency_score Float32,          -- Transaction timing consistency

    -- Graph algorithm features (placeholders for Phase 4)
    pagerank Float32,                   -- PageRank centrality score
    betweenness Float32,                -- Betweenness centrality
    closeness Float32,                  -- Closeness centrality
    clustering_coefficient Float32,     -- Local clustering coefficient
    kcore UInt32,                       -- K-core decomposition result
    community_id UInt32,                -- Community detection result
    centrality_score Float32,           -- Combined centrality measure

    -- k-hop neighborhood features
    khop1_count UInt32,                 -- 1-hop neighbors count
    khop2_count UInt32,                 -- 2-hop neighbors count
    khop3_count UInt32,                 -- 3-hop neighbors count
    khop1_volume_usd Decimal128(18),    -- 1-hop neighborhood volume
    khop2_volume_usd Decimal128(18),    -- 2-hop neighborhood volume
    khop3_volume_usd Decimal128(18),    -- 3-hop neighborhood volume

    -- Advanced flow features
    flow_reciprocity_entropy Float32,   -- Entropy of reciprocal flows
    counterparty_stability Float32,     -- Stability of counterparty relationships
    flow_burstiness Float32,            -- Temporal burstiness of flows
    transaction_regularity Float32,     -- Regularity of transaction timing
    amount_predictability Float32,      -- Predictability of transaction amounts

    -- Risk and anomaly features
    behavioral_anomaly_score Float32,   -- Behavioral/temporal anomaly score from IsolationForest
    graph_anomaly_score Float32,        -- Graph structure anomaly score from IsolationForest
    neighborhood_anomaly_score Float32, -- Neighborhood interaction anomaly score from IsolationForest
    global_anomaly_score Float32,       -- Global anomaly score across all feature dimensions
    outlier_transactions UInt32,        -- Count of outlier transactions
    suspicious_pattern_score Float32,   -- Suspicious activity pattern score

    -- Classification features
    is_exchange_like Boolean,           -- High degree, regular patterns
    is_whale Boolean,                   -- High volume transactions
    is_mixer_like Boolean,              -- Many small mixed transactions
    is_contract_like Boolean,           -- Contract-like interaction patterns
    is_new_address Boolean,             -- First seen in current window
    is_dormant_reactivated Boolean,     -- Previously inactive, now active

    -- Extended classification features (from transfers.sql address types)
    is_high_volume_trader Boolean,      -- >= $10K volume + >= 1000 transactions
    is_hub_address Boolean,             -- >= 50 recipients + >= 50 senders
    is_retail_active Boolean,           -- >= 100 transactions + < $1K volume
    is_whale_inactive Boolean,          -- < 10 transactions + >= $10K volume
    is_retail_inactive Boolean,         -- < 10 transactions + < $100 volume
    is_regular_user Boolean,            -- Default classification

    -- Supporting metrics for classification
    unique_recipients_count UInt32,     -- Number of unique addresses received funds from
    unique_senders_count UInt32,        -- Number of unique addresses sent funds to

    -- Feature quality and metadata
    completeness_score Float32,         -- Feature completeness score (0-1)
    quality_score Float32,              -- Overall feature quality score (0-1)
    outlier_score Float32,              -- Statistical outlier score (0-1)
    confidence_score Float32,           -- Confidence in feature calculations

    -- Temporal metadata
    first_activity_timestamp UInt64,    -- First transaction in window
    last_activity_timestamp UInt64,     -- Last transaction in window

    _version UInt64                     -- For ReplacingMergeTree
)
ENGINE = ReplacingMergeTree(_version)
PARTITION BY (window_days, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, address)
SETTINGS index_granularity = 8192;

-- Indexes for ML feature queries
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_processing_date processing_date TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_window_days window_days TYPE set(0) GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_address address TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_total_volume_usd total_volume_usd TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_degree_total degree_total TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_is_whale is_whale TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_is_exchange_like is_exchange_like TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_behavioral_anomaly behavioral_anomaly_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_graph_anomaly graph_anomaly_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_neighborhood_anomaly neighborhood_anomaly_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_global_anomaly global_anomaly_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_features ADD INDEX IF NOT EXISTS idx_quality_score quality_score TYPE minmax GRANULARITY 4;

-- =============================================================================
-- analyzers_feature_stats: Statistical distributions for ML normalization
-- =============================================================================

CREATE OR REPLACE TABLE analyzers_feature_stats (
    -- Time series dimensions
    window_days UInt16,
    processing_date Date,

    -- Feature identification
    feature_name String,
    feature_category String,

    -- Statistical distribution metrics
    count UInt64,                       -- Number of non-null observations
    mean Float64,                       -- Mean value
    std Float64,                        -- Standard deviation
    variance Float64,                   -- Variance
    min_value Float64,                  -- Minimum value
    max_value Float64,                  -- Maximum value

    -- Percentile statistics
    percentile_1 Float64,               -- 1st percentile
    percentile_5 Float64,               -- 5th percentile
    percentile_10 Float64,              -- 10th percentile
    percentile_25 Float64,              -- 25th percentile (Q1)
    percentile_50 Float64,              -- 50th percentile (median)
    percentile_75 Float64,              -- 75th percentile (Q3)
    percentile_90 Float64,              -- 90th percentile
    percentile_95 Float64,              -- 95th percentile
    percentile_99 Float64,              -- 99th percentile

    -- Distribution shape metrics
    skewness Float64,                   -- Distribution skewness
    kurtosis Float64,                   -- Distribution kurtosis
    entropy Float64,                    -- Shannon entropy

    -- Outlier detection thresholds
    iqr Float64,                        -- Interquartile range
    lower_fence Float64,                -- Lower outlier fence (Q1 - 1.5*IQR)
    upper_fence Float64,                -- Upper outlier fence (Q3 + 1.5*IQR)
    outlier_count UInt32,               -- Number of outliers detected
    outlier_percentage Float32,         -- Percentage of outliers

    -- ML normalization parameters
    z_score_threshold Float64,          -- Z-score threshold for outliers
    mad Float64,                        -- Median Absolute Deviation
    robust_mean Float64,                -- Robust mean (outliers excluded)
    robust_std Float64,                 -- Robust standard deviation

    -- Data quality metrics
    null_count UInt64,                  -- Number of null values
    null_percentage Float32,            -- Percentage of nulls
    zero_count UInt64,                  -- Number of zero values
    unique_count UInt64,                -- Number of unique values
    cardinality_ratio Float32,          -- unique_count / total_count

    _version UInt64                     -- For ReplacingMergeTree
)
ENGINE = ReplacingMergeTree(_version)
PARTITION BY (window_days, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, feature_name)
SETTINGS index_granularity = 8192;

-- Indexes for feature statistics queries
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_processing_date processing_date TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_window_days window_days TYPE set(0) GRANULARITY 4;
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_feature_name feature_name TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_feature_category feature_category TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_outlier_percentage outlier_percentage TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_feature_stats ADD INDEX IF NOT EXISTS idx_null_percentage null_percentage TYPE minmax GRANULARITY 4;