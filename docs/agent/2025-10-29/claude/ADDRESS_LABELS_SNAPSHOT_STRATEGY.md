# Address Labels Snapshot Strategy

**Date**: 2025-10-29
**Purpose**: Design snapshot strategy for raw address labels reference data

---

## Problem Statement

Address labels data needs to be:
1. Downloaded from S3 (`s3://{bucket}/address-labels/{network}/address_labels_{processing_date}.parquet`)
2. Stored as `raw_address_labels` (reference data, not modified by risk scoring)
3. **Stored as snapshots** per `processing_date` for training consistency
4. Used for ML training with proper labels (risk_level: low/medium/high/critical)

---

## Proposed Solution: Processing Date Snapshots

### ✅ Recommended Approach

**Add `processing_date` field to track when labels were ingested**

This follows the same pattern as `raw_alerts`, `raw_features`, etc.

### Schema: `raw_address_labels`

Following the `raw_` naming convention for reference data:

```sql
CREATE TABLE IF NOT EXISTS raw_address_labels (
    -- Snapshot tracking
    processing_date Date,
    window_days UInt16,
    
    -- Primary identifiers
    network String,
    address String,
    
    -- Label details
    label String,
    address_type String DEFAULT 'unknown',
    address_subtype String DEFAULT '',
    
    -- Risk classification (for ML training)
    risk_level String DEFAULT 'medium',  -- low/medium/high/critical
    confidence_score Float32 DEFAULT 0.5,
    
    source String DEFAULT ''
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(processing_date), network)
ORDER BY (processing_date, window_days, network, address, label)
SETTINGS index_granularity = 8192;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_processing_date ON raw_address_labels(processing_date) TYPE minmax GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_window_days ON raw_address_labels(window_days) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_address ON raw_address_labels(address) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_network ON raw_address_labels(network) TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_risk_level ON raw_address_labels(risk_level) TYPE bloom_filter(0.01) GRANULARITY 4;
```

---

## S3 Structure

### Format: `s3://{bucket}/address-labels/{network}/address_labels_{processing_date}.parquet`

```
s3://{bucket}/address-labels/
├── ethereum/
│   ├── address_labels_2024-01-15.parquet
│   ├── address_labels_2024-01-14.parquet
│   └── address_labels_2024-01-13.parquet
├── bitcoin/
│   ├── address_labels_2024-01-15.parquet
│   └── address_labels_2024-01-14.parquet
└── polygon/
    └── address_labels_2024-01-15.parquet
```

**Ingestion**: Download `address_labels_{processing_date}.parquet` matching current batch date

---

## Ingestion Workflow

### Modified SOTDataIngestion

```python
class SOTDataIngestion(ABC):
    
    def run(self):
        # ... existing code ...
        
        # NEW: Download and ingest address labels
        logger.info("Downloading address labels")
        self._download_address_labels()
        
        logger.info("Ingesting address labels")
        self._ingest_address_labels()
        
        # Continue with alerts, features, etc.
        # ...
    
    def _download_address_labels(self):
        """Download address labels matching processing_date from S3"""
        
        # S3 path: address-labels/{network}/address_labels_{processing_date}.parquet
        s3_key = f"address-labels/{self.network}/address_labels_{self.processing_date}.parquet"
        local_path = self.local_dir / "address_labels.parquet"
        
        logger.info(f"Downloading address labels: s3://{self.bucket}/{s3_key}")
        
        try:
            self.s3_client.download_file(
                self.bucket,
                s3_key,
                str(local_path)
            )
            logger.success(f"Downloaded address labels to {local_path}")
        except ClientError as e:
            logger.warning(f"No address labels found for {self.network} on {self.processing_date}: {e}")
            # Optional: Try previous day or skip
            return False
        
        return True
    
    def _ingest_address_labels(self):
        """Ingest address labels with processing_date snapshot"""
        
        file_path = self.local_dir / "address_labels.parquet"
        
        if not file_path.exists():
            logger.warning("No address labels file found")
            return
        
        df = pd.read_parquet(file_path)
        
        if df.empty:
            logger.warning("Address labels file is empty")
            return
        
        # Add processing_date and window_days
        df['processing_date'] = pd.to_datetime(self.processing_date)
        df['window_days'] = self.days
        
        logger.info(f"Ingesting {len(df):,} address labels")
        
        self.client.insert_df(table='raw_address_labels', df=df)
        
        logger.success(f"Ingested {len(df):,} address labels")
```

---

## Data Flow

```
Day 1 (2024-01-01):
S3: address-labels/ethereum/address_labels_2024-01-01.parquet (10K addresses)
    ↓
Download & Insert with processing_date='2024-01-01', window_days=7
    ↓
ClickHouse: 10K records with processing_date='2024-01-01'

Day 2 (2024-01-02):
S3: address-labels/ethereum/address_labels_2024-01-02.parquet (10.5K addresses)
    ↓
Download & Insert with processing_date='2024-01-02'
    ↓
ClickHouse:
  - 10K records with processing_date='2024-01-01'
  - 10.5K records with processing_date='2024-01-02'

Training on Day 1 data:
  SELECT * FROM raw_address_labels
  WHERE processing_date = '2024-01-01'
  AND window_days = 7
  AND network = 'ethereum'
  → Gets Day 1 snapshot (10K addresses)

Training on Day 2 data:
  SELECT * FROM raw_address_labels
  WHERE processing_date = '2024-01-02'
  AND window_days = 7
  AND network = 'ethereum'
  → Gets Day 2 snapshot (10.5K addresses)
```

---

## Benefits

### ✅ Snapshot Consistency
- Each `processing_date` has its own label snapshot
- Training on historical data uses historical labels
- Reproducible training results

### ✅ Temporal Analysis
- Track label changes over time
- Analyze label drift
- Monitor labeling quality

### ✅ Pattern Consistency
- Same approach as `raw_alerts`, `raw_features`, etc.
- Familiar query patterns
- Easy to understand

### ✅ Storage Efficiency
- ReplacingMergeTree deduplicates if same address/label
- Only stores changes between snapshots
- Automatic cleanup with TTL (optional)

---

## Training Integration

### Extract Labels with Features

```python
# In FeatureExtractor
def extract_training_data(
    self,
    start_date: str,
    end_date: str,
    window_days: int = 7
) -> Dict[str, pd.DataFrame]:
    
    data = {
        'alerts': self._extract_alerts(...),
        'features': self._extract_features(...),
        'clusters': self._extract_clusters(...),
        'money_flows': self._extract_money_flows(...),
        'address_labels': self._extract_address_labels(...)  # NEW
    }
    
    return data

def _extract_address_labels(
    self,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Extract address labels for training period"""
    
    query = f"""
        SELECT DISTINCT ON (address)
            address,
            label,
            address_type,
            risk_level,
            confidence_score,
            processing_date
        FROM core_address_labels
        WHERE processing_date >= '{start_date}'
          AND processing_date <= '{end_date}'
        ORDER BY address, processing_date DESC
    """
    
    # Gets most recent label for each address in date range
    result = self.client.query(query)
    
    if not result.result_rows:
        logger.warning("No address labels found")
        return pd.DataFrame()
    
    df = pd.DataFrame(
        result.result_rows,
        columns=[col[0] for col in result.column_names]
    )
    
    logger.info(f"Extracted {len(df):,} address labels")
    
    return df
```

### Use Labels in Feature Building

```python
# In FeatureBuilder
def _add_label_features(
    self,
    alerts_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Add address label features"""
    
    if labels_df.empty:
        logger.warning("No labels available - using defaults")
        alerts_df['has_label'] = 0
        alerts_df['label_risk_encoded'] = 2  # medium default
        return alerts_df
    
    # Merge with labels
    merged = alerts_df.merge(
        labels_df[['address', 'risk_level', 'confidence_score']],
        on='address',
        how='left'
    )
    
    # Binary flag for labeled addresses
    merged['has_label'] = merged['risk_level'].notna().astype(int)
    
    # Encode risk level
    risk_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    merged['label_risk_encoded'] = merged['risk_level'].map(risk_map).fillna(2)
    
    # Label confidence
    merged['label_confidence'] = merged['confidence_score'].fillna(0.5)
    
    return merged
```

---

## Query Examples

### Get Labels for Specific Date

```sql
-- Labels snapshot for 2024-01-15
SELECT 
    address,
    label,
    risk_level,
    confidence_score
FROM core_address_labels
WHERE processing_date = '2024-01-15'
  AND network = 'ethereum'
```

### Get Latest Labels in Date Range

```sql
-- Most recent label for each address in Jan 2024
SELECT DISTINCT ON (address)
    address,
    label,
    risk_level,
    processing_date
FROM core_address_labels
WHERE processing_date >= '2024-01-01'
  AND processing_date <= '2024-01-31'
  AND network = 'ethereum'
ORDER BY address, processing_date DESC
```

### Count Label Changes

```sql
-- How many addresses got new labels each day?
SELECT 
    processing_date,
    COUNT(DISTINCT address) as labeled_addresses,
    COUNT(*) as total_labels
FROM core_address_labels
WHERE network = 'ethereum'
GROUP BY processing_date
ORDER BY processing_date DESC
LIMIT 30
```

---

## Storage Considerations

### Deduplication

ReplacingMergeTree automatically deduplicates:
- Same `(processing_date, network, address, label)` 
- Keeps highest `_version`
- Runs during merges

### Partitioning

```sql
PARTITION BY (toYYYYMM(processing_date), network)
```

Benefits:
- Efficient queries by date range
- Easy to drop old partitions
- Isolated by network

### TTL (Optional)

```sql
ALTER TABLE core_address_labels 
MODIFY TTL processing_date + INTERVAL 365 DAY;
```

Automatically delete labels older than 1 year.

---

## Implementation Checklist

- [ ] Create updated `core_address_labels.sql` schema
- [ ] Add to `MigrateSchema` in `packages/storage/__init__.py`
- [ ] Update `SOTDataIngestion`:
  - [ ] Add `_download_address_labels()` method
  - [ ] Add `_ingest_address_labels()` method
  - [ ] Call in `run()` workflow
- [ ] Update `FeatureExtractor`:
  - [ ] Add `_extract_address_labels()` method
  - [ ] Return in `extract_training_data()`
- [ ] Update `FeatureBuilder`:
  - [ ] Add `_add_label_features()` method
  - [ ] Call in `build_training_features()`
- [ ] Test with sample data
- [ ] Verify snapshot behavior

---

## Alternative: Separate Snapshots Table

If you want to keep a "current" table plus snapshots:

```sql
-- Current labels (no processing_date)
CREATE TABLE core_address_labels (...)

-- Historical snapshots
CREATE TABLE core_address_labels_snapshots (
    processing_date Date,
    ...
) PARTITION BY (toYYYYMM(processing_date), network)
```

**Not recommended** because:
- More complex queries
- Duplication between tables
- Inconsistent with other raw_ tables

---

## Recommendation

✅ **Use single table with `processing_date` field**

**Why:**
- Consistent with existing pattern
- Simple queries
- Automatic deduplication
- Temporal analysis built-in
- Training uses same approach as alerts/features

**Next Steps:**
1. Approve this approach
2. Implement schema changes
3. Update ingestion code
4. Update training code
5. Test with real data
