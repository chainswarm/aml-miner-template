CREATE TABLE IF NOT EXISTS raw_address_labels (
    processing_date Date,
    window_days UInt16,
    network String,
    address String,
    label String,
    network_type String DEFAULT '',
    address_type String DEFAULT 'unknown',
    address_subtype String DEFAULT '',
    risk_level String DEFAULT 'medium',
    confidence_score Float32 DEFAULT 0.5,
    trust_level String DEFAULT '',
    source String DEFAULT ''
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date), network)
ORDER BY (processing_date, window_days, network, address, label)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_address_labels(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_address_labels(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_address ON raw_address_labels(address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_network ON raw_address_labels(network) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_risk_level ON raw_address_labels(risk_level) TYPE bloom_filter(0.01) GRANULARITY 4;