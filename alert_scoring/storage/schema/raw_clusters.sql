CREATE TABLE IF NOT EXISTS raw_clusters (
    window_days Int32,
    processing_date Date,
    network String,
    cluster_id String,
    cluster_type String,
    primary_address String DEFAULT '',
    pattern_id String DEFAULT '',
    primary_alert_id String,
    related_alert_ids Array(String),
    addresses_involved Array(String),
    total_alerts Int32,
    total_volume_usd Decimal(18, 2),
    severity_max Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    confidence_avg Float64,
    earliest_alert_timestamp Int64,
    latest_alert_timestamp Int64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;