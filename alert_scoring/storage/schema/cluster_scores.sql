CREATE TABLE IF NOT EXISTS cluster_scores (
    processing_date Date,
    network String,
    cluster_id String,
    score Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;