CREATE TABLE IF NOT EXISTS alert_scores (
    processing_date Date,
    network String,
    alert_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    explain_json String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_score ON alert_scores(score) TYPE minmax GRANULARITY 4;