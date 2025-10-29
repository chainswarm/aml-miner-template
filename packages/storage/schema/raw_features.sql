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