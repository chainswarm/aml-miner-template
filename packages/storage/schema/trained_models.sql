CREATE TABLE IF NOT EXISTS trained_models (
    model_id String,
    model_type String,
    version String,
    network String,
    training_start_date Date,
    training_end_date Date,
    created_at DateTime,
    model_path String,
    metrics_json String,
    hyperparameters_json String,
    feature_names Array(String),
    num_samples UInt32,
    num_features UInt16,
    test_auc Float32,
    cv_auc_mean Float32,
    cv_auc_std Float32
)
ENGINE = MergeTree()
ORDER BY (network, model_type, created_at)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_model_type ON trained_models(model_type) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_network ON trained_models(network) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_version ON trained_models(version) TYPE bloom_filter(0.01) GRANULARITY 4;