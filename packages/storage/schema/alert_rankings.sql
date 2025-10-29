CREATE TABLE IF NOT EXISTS alert_rankings (
    processing_date Date,
    alert_id String,
    rank Int32,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date))
ORDER BY (processing_date, rank)
SETTINGS index_granularity = 8192;