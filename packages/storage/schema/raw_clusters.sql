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