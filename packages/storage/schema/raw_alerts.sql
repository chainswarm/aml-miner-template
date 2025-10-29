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