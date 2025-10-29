CREATE TABLE IF NOT EXISTS batch_metadata (
    processing_date Date,
    processed_at DateTime,
    input_counts_alerts Int32,
    input_counts_features Int32,
    input_counts_clusters Int32,
    output_counts_alert_scores Int32,
    output_counts_alert_rankings Int32,
    output_counts_cluster_scores Int32,
    latencies_ms_alert_scoring Int32,
    latencies_ms_alert_ranking Int32,
    latencies_ms_cluster_scoring Int32,
    latencies_ms_total Int32,
    model_versions_alert_scorer String,
    model_versions_alert_ranker String,
    model_versions_cluster_scorer String,
    status Enum8('PROCESSING' = 1, 'COMPLETED' = 2, 'FAILED' = 3) DEFAULT 'PROCESSING',
    error_message String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date))
ORDER BY (processing_date)
SETTINGS index_granularity = 8192;