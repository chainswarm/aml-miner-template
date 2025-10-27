CREATE TABLE IF NOT EXISTS raw_alerts (
    window_days Int32,
    processing_date Date,
    alert_id String,
    address String,
    typology_type String,
    pattern_id String DEFAULT '',
    pattern_type String DEFAULT '',
    severity Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    suspected_address_type String DEFAULT 'unknown',
    suspected_address_subtype String DEFAULT '',
    alert_confidence_score Float64,
    description String,
    volume_usd Decimal(18, 2) DEFAULT 0,
    evidence_json String,
    risk_indicators Array(String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date))
ORDER BY (processing_date, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_alert_id ON raw_alerts(alert_id) TYPE bloom_filter GRANULARITY 1;