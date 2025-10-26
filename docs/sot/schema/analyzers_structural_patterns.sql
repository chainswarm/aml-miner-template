-- =============================================================================
-- analyzers_structural_patterns: Structural patterns for AML detection
-- =============================================================================
-- Stores graph-based patterns that cannot fit in single-row features table
-- =============================================================================

CREATE OR REPLACE TABLE analyzers_structural_patterns (
    -- Time series dimensions
    window_days UInt16,
    processing_date Date,

    -- Primary identifiers
    pattern_id String,
    address String,

    -- Pattern classification (using String like alerts.sql)
    pattern_type String,                -- Type: cycle, layering_path, smurfing_network, etc.
    pattern_subtype String,             -- Optional subtype for specific variations

    -- Pattern metrics
    severity_score Float32,             -- Pattern severity (0.0-1.0)
    confidence_score Float32,           -- Detection confidence (0.0-1.0)
    risk_score Float32,                 -- Risk assessment (0.0-1.0)

    -- Pattern-specific data (JSON for flexibility)
    pattern_data String,                -- JSON containing pattern-specific details

    -- Cycle-specific fields (for cycle patterns)
    cycle_length UInt32,                -- Length of detected cycle (0 if not cycle)
    cycle_volume_usd Decimal128(18),    -- Total volume in cycle
    cycle_addresses Array(String),      -- Addresses in the cycle

    -- Path-specific fields (for layering patterns)
    path_depth UInt32,                  -- Depth of layering path (0 if not path)
    path_volume_usd Decimal128(18),     -- Volume through the path
    path_addresses Array(String),       -- Addresses in the path
    source_address String,              -- Origin address of the path
    destination_address String,         -- Final address of the path

    -- Network-specific fields (for smurfing/proximity patterns)
    network_size UInt32,                -- Size of detected network
    network_density Float32,            -- Density of the network (0.0-1.0)
    network_addresses Array(String),    -- Addresses in the network
    hub_addresses Array(String),        -- Central hub addresses

    -- Proximity fields (for proximity-to-risk patterns)
    risk_source_address String,         -- Address that is source of risk
    distance_to_risk UInt32,            -- Hop distance to risky address
    risk_propagation_score Float32,     -- How risk propagates (0.0-1.0)

    -- Motif fields (for structural motifs)
    motif_type String,                  -- Type of motif (fanin, fanout, etc.)
    motif_center_address String,        -- Central address in motif
    motif_participant_count UInt32,     -- Number of participating addresses

    -- Temporal information
    detection_timestamp UInt64,         -- When pattern was detected
    pattern_start_time UInt64,          -- When pattern activity started
    pattern_end_time UInt64,            -- When pattern activity ended
    pattern_duration_hours UInt32,      -- Duration of pattern activity

    -- Evidence and metadata
    evidence_transaction_count UInt32,  -- Number of supporting transactions
    evidence_volume_usd Decimal128(18), -- Total volume evidence
    detection_method String,            -- Algorithm used for detection

    -- Quality metrics
    anomaly_score Float32,              -- Statistical anomaly strength

    -- Administrative fields
    _version UInt64                     -- For ReplacingMergeTree
)
ENGINE = ReplacingMergeTree(_version)
PARTITION BY (pattern_type, toYYYYMM(processing_date))
ORDER BY (window_days, processing_date, pattern_type, address, pattern_id)
SETTINGS index_granularity = 8192;

-- Indexes for structural pattern queries
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_processing_date processing_date TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_window_days window_days TYPE set(0) GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_address address TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_pattern_type pattern_type TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_severity_score severity_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_risk_score risk_score TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_detection_timestamp detection_timestamp TYPE minmax GRANULARITY 4;
ALTER TABLE analyzers_structural_patterns ADD INDEX IF NOT EXISTS idx_evidence_volume evidence_volume_usd TYPE minmax GRANULARITY 4;