CREATE TABLE IF NOT EXISTS risk_scoring_cluster_scores (
    processing_date Date,
    cluster_id String,
    score Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date))
ORDER BY (processing_date, cluster_id)
SETTINGS index_granularity = 8192;