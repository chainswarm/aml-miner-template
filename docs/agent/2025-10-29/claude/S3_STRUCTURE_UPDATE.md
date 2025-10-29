# S3 Structure Update

## Overview

Updated ingestion code to match the new S3 folder structure for the risk-scoring bucket.

## New S3 Structure

```
s3://risk-scoring/
├── snapshots/{network}/{processing_date}/  # Risk assessment snapshots
│   ├── META.json
│   ├── alerts.parquet
│   ├── features.parquet
│   ├── clusters.parquet
│   └── money_flows.parquet
│
└── address-labels/                         # Address labels reference
    ├── ethereum_address_labels.parquet
    ├── bitcoin_address_labels.parquet
    └── polygon_address_labels.parquet
```

## Changes Made

### 1. Snapshot Path Update

**Before:**
```python
self.s3_prefix = f"{network}/{processing_date}/{days}d"
self.local_dir = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / network / processing_date / f'{days}d'
```

**After:**
```python
self.s3_prefix = f"snapshots/{network}/{processing_date}"
self.local_dir = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / 'snapshots' / network / processing_date
```

### 2. Address Labels Path Update

**Before:**
```python
s3_key = f"address-labels/{network}/address_labels_{processing_date}.parquet"
```

**After:**
```python
s3_key = f"address-labels/{network}_address_labels.parquet"
```

## Key Differences

### Snapshots
- **Old**: `s3://bucket/{network}/{processing_date}/{window_days}d/*.parquet`
- **New**: `s3://bucket/snapshots/{network}/{processing_date}/*.parquet`
- **Rationale**: Cleaner structure, window_days is now a data attribute not a path component

### Address Labels
- **Old**: `s3://bucket/address-labels/{network}/address_labels_{processing_date}.parquet`
- **New**: `s3://bucket/address-labels/{network}_address_labels.parquet`
- **Rationale**: Single file per network (not date-specific), simpler reference data structure

## Impact

### Ingestion
✅ Updated `packages/ingestion/sot_ingestion.py`
- Modified `__init__()` to use new paths
- Modified `_download_address_labels()` to use new S3 key format

### Training
✅ No changes needed
- Training code extracts from ClickHouse (not directly from S3)
- Ingestion handles S3 → ClickHouse transformation

### Local Storage
```
data/input/risk-scoring/
├── snapshots/
│   └── {network}/
│       └── {processing_date}/
│           ├── META.json
│           ├── alerts.parquet
│           ├── features.parquet
│           ├── clusters.parquet
│           └── money_flows.parquet
│
└── address-labels/
    ├── ethereum_address_labels.parquet
    ├── bitcoin_address_labels.parquet
    └── torus_address_labels.parquet
```

## Usage

### Ingestion Command (Unchanged)
```bash
python -m packages.ingestion.sot_ingestion \
    --network ethereum \
    --processing-date 2024-01-15 \
    --days 7
```

### Expected S3 Locations
- **Snapshots**: `s3://risk-scoring/snapshots/ethereum/2024-01-15/`
- **Address Labels**: `s3://risk-scoring/address-labels/ethereum_address_labels.parquet`

## Testing

Verify the changes work correctly:

```bash
# Test ingestion with new structure
python -m packages.ingestion.sot_ingestion \
    --network ethereum \
    --processing-date 2024-01-15 \
    --days 7

# Expected logs:
# - "S3 source: s3://bucket/snapshots/ethereum/2024-01-15"
# - "Downloading address labels from s3://bucket/address-labels/ethereum_address_labels.parquet"
```

## Migration Notes

### For Existing S3 Data
If you have data in the old structure, you can migrate it:

```bash
# Copy snapshots to new location
aws s3 sync s3://risk-scoring/ethereum/2024-01-15/7d/ \
            s3://risk-scoring/snapshots/ethereum/2024-01-15/

# Consolidate address labels (one file per network)
aws s3 cp s3://risk-scoring/address-labels/ethereum/address_labels_2024-01-15.parquet \
          s3://risk-scoring/address-labels/ethereum_address_labels.parquet
```

### Database Schema
No changes needed - ClickHouse tables remain the same:
- `raw_alerts`
- `raw_features`
- `raw_clusters`
- `raw_money_flows`
- `raw_address_labels`

## File Changes

Modified:
- [`packages/ingestion/sot_ingestion.py`](../../packages/ingestion/sot_ingestion.py) - Lines 28-29, 428

## References

- Original structure: [`S3_FOLDER_STRUCTURE_PRESERVATION.md`](S3_FOLDER_STRUCTURE_PRESERVATION.md)
- Address labels: [`ADDRESS_LABELS_SNAPSHOT_STRATEGY.md`](ADDRESS_LABELS_SNAPSHOT_STRATEGY.md)
- Implementation: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)