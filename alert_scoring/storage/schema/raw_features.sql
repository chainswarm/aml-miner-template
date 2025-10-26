CREATE TABLE IF NOT EXISTS raw_features (
    processing_date Date,
    network String,
    address String,
    feature_name String,
    feature_value Float64,
    feature_metadata String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, address, feature_name)
SETTINGS index_granularity = 8192;