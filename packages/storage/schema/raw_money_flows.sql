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