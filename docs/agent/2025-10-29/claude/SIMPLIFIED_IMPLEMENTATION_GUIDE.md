# Simplified Implementation Guide

**Date:** 2025-10-29  
**Architecture:** Direct Parquet Insertion (No Transformation)  
**Status:** Ready for Implementation

## Overview

Since the source system will provide complete data in parquet files, we only need to:
1. Update schema files to match source
2. Add money_flows table support
3. Add checksum validation (optional but recommended)

**No transformation layer needed!**

## Implementation Steps

### Phase 1: Update Schema Files

#### Task 1.1: Update raw_features.sql
**File:** [`packages/storage/schema/raw_features.sql`](../../../packages/storage/schema/raw_features.sql)  
**Action:** Replace entire file with schema from [`SCHEMA_SPECIFICATIONS.md`](SCHEMA_SPECIFICATIONS.md#1-raw_featuressql)

**Key Changes:**
- Change from 4 columns (long format) → 96+ columns (wide format)
- Match `analyzers_features` schema
- Remove `_version` column
- Keep `created_at` for audit

#### Task 1.2: Update raw_alerts.sql
**File:** [`packages/storage/schema/raw_alerts.sql`](../../../packages/storage/schema/raw_alerts.sql)  
**Action:** Update severity field and data types

**Key Changes:**
```sql
-- OLD:
severity Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',

-- NEW:
severity String DEFAULT 'medium',
```
- Change `Enum8` → `String`
- Update `alert_confidence_score` to `Float32`
- Update `volume_usd` to `Decimal128(18)`

#### Task 1.3: Update raw_clusters.sql
**File:** [`packages/storage/schema/raw_clusters.sql`](../../../packages/storage/schema/raw_clusters.sql)  
**Action:** Update severity_max field and data types

**Key Changes:**
```sql
-- OLD:
severity_max Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',

-- NEW:
severity_max String DEFAULT 'medium',
```
- Change `Enum8` → `String`
- Update `confidence_avg` to `Float32`
- Update `total_volume_usd` to `Decimal128(18)`

#### Task 1.4: Create raw_money_flows.sql
**File:** `packages/storage/schema/raw_money_flows.sql`  
**Action:** Create new file with schema from [`SCHEMA_SPECIFICATIONS.md`](SCHEMA_SPECIFICATIONS.md#4-raw_money_flowssql)

**This is a new table** based on `core_money_flows_view`

---

### Phase 2: Update Ingestion Script

#### Task 2.1: Add money_flows to ingestion files mapping
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** Around line 213-218

**Current code:**
```python
ingestion_files = {}
for table, base_name in [
    ('raw_alerts', 'alerts'),
    ('raw_features', 'features'),
    ('raw_clusters', 'clusters')
]:
```

**Update to:**
```python
ingestion_files = {}
for table, base_name in [
    ('raw_alerts', 'alerts'),
    ('raw_features', 'features'),
    ('raw_clusters', 'clusters'),
    ('raw_money_flows', 'money_flows')
]:
```

#### Task 2.2: Update validation count check
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** Around line 133, 162

**Current code:**
```python
validation_query = f"""
    SELECT COUNT(DISTINCT table) as tables_with_data
    FROM (
        SELECT 'raw_alerts' as table
        FROM raw_alerts
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
        
        UNION ALL
        
        SELECT 'raw_features' as table
        FROM raw_features
        WHERE processing_date = '{self.processing_date}'
        LIMIT 1
        
        UNION ALL
        
        SELECT 'raw_clusters' as table
        FROM raw_clusters
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
    )
"""
# ...
if tables_with_data == 3:
```

**Update to:**
```python
validation_query = f"""
    SELECT COUNT(DISTINCT table) as tables_with_data
    FROM (
        SELECT 'raw_alerts' as table
        FROM raw_alerts
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
        
        UNION ALL
        
        SELECT 'raw_features' as table
        FROM raw_features
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
        
        UNION ALL
        
        SELECT 'raw_clusters' as table
        FROM raw_clusters
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
        
        UNION ALL
        
        SELECT 'raw_money_flows' as table
        FROM raw_money_flows
        WHERE processing_date = '{self.processing_date}'
          AND window_days = {self.days}
        LIMIT 1
    )
"""
# ...
if tables_with_data == 4:  # Changed from 3
```

#### Task 2.3: Update cleanup queries
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** Around line 176-180

**Add money_flows to cleanup:**
```python
cleanup_queries = [
    f"ALTER TABLE raw_alerts DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
    f"ALTER TABLE raw_features DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
    f"ALTER TABLE raw_clusters DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
    f"ALTER TABLE raw_money_flows DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}"
]
```

#### Task 2.4: Update verification query
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** Around line 282-303

**Add money_flows to verification:**
```python
verify_query = f"""
    SELECT
        'raw_alerts' as table, COUNT(*) as count
    FROM raw_alerts
    WHERE processing_date = '{self.processing_date}'
      AND window_days = {self.days}
    
    UNION ALL
    
    SELECT
        'raw_features' as table, COUNT(*) as count
    FROM raw_features
    WHERE processing_date = '{self.processing_date}'
      AND window_days = {self.days}
    
    UNION ALL
    
    SELECT
        'raw_clusters' as table, COUNT(*) as count
    FROM raw_clusters
    WHERE processing_date = '{self.processing_date}'
      AND window_days = {self.days}
    
    UNION ALL
    
    SELECT
        'raw_money_flows' as table, COUNT(*) as count
    FROM raw_money_flows
    WHERE processing_date = '{self.processing_date}'
      AND window_days = {self.days}
"""
```

#### Task 2.5: Update validation requirements (Optional)
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** Around line 93-97

**Update required columns check:**
```python
required_columns = {
    'raw_alerts': ['alert_id', 'processing_date', 'window_days', 'address'],
    'raw_features': ['processing_date', 'window_days', 'address'],  # Removed feature_name, feature_value
    'raw_clusters': ['cluster_id', 'processing_date', 'window_days'],
    'raw_money_flows': ['from_address', 'to_address', 'processing_date', 'window_days']
}
```

**Note:** With source providing complete data, this validation is less critical but still useful as a sanity check.

---

### Phase 3: Optional - Add Checksum Validation

This is optional but recommended for data integrity.

#### Task 3.1: Download META.json first
**File:** [`packages/ingestion/sot_ingestion.py`](../../../packages/ingestion/sot_ingestion.py)  
**Location:** In `_download_all()` method around line 30

**Add before parquet downloads:**
```python
def _download_all(self) -> int:
    try:
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.s3_prefix
        )

        if 'Contents' not in response:
            logger.warning(f"No files found in S3 at {self.s3_prefix}")
            return 0

        # Download META.json first
        meta_key = f"{self.s3_prefix}/META.json"
        meta_found = any(obj['Key'] == meta_key for obj in response['Contents'])
        
        if not meta_found:
            logger.warning(f"META.json not found at {meta_key}, proceeding without checksum validation")
        else:
            self._download_file(meta_key)
            logger.success("Downloaded META.json")
        
        # Continue with parquet files...
```

#### Task 3.2: Add checksum validation
**Create new file (optional):** `packages/ingestion/file_validator.py`

Or add inline validation after download. This is optional since parquet files have built-in integrity checks.

---

## Testing Steps

### Step 1: Test Schema Creation
```bash
# Connect to ClickHouse
clickhouse-client

# Drop old tables (if they exist)
DROP TABLE IF EXISTS risk_scoring_torus.raw_features;
DROP TABLE IF EXISTS risk_scoring_torus.raw_alerts;
DROP TABLE IF EXISTS risk_scoring_torus.raw_clusters;

# Run migration to create new schemas
python scripts/init_database.py --network torus
```

### Step 2: Test Ingestion
```bash
# Run ingestion with updated code
python packages/ingestion/sot_ingestion.py \
  --network torus \
  --processing-date 2025-08-01 \
  --days 195
```

### Step 3: Verify Data
```sql
-- Check row counts
SELECT 'raw_alerts' as table, count(*) as rows FROM raw_alerts;
SELECT 'raw_features' as table, count(*) as rows FROM raw_features;
SELECT 'raw_clusters' as table, count(*) as rows FROM raw_clusters;
SELECT 'raw_money_flows' as table, count(*) as rows FROM raw_money_flows;

-- Verify features are wide format
SELECT count(*) as column_count 
FROM system.columns 
WHERE database = 'risk_scoring_torus' 
  AND table = 'raw_features';
-- Should return ~96+ columns

-- Sample data check
SELECT * FROM raw_features LIMIT 1;
SELECT * FROM raw_alerts LIMIT 1;
SELECT * FROM raw_clusters LIMIT 1;
SELECT * FROM raw_money_flows LIMIT 1;
```

---

## File Checklist

### Files to Update
- [x] `packages/storage/schema/raw_features.sql` - Replace entire file
- [x] `packages/storage/schema/raw_alerts.sql` - Update severity, data types
- [x] `packages/storage/schema/raw_clusters.sql` - Update severity_max, data types
- [ ] `packages/storage/schema/raw_money_flows.sql` - Create new file

### Code Changes
- [ ] `packages/ingestion/sot_ingestion.py`
  - [ ] Add money_flows to ingestion_files (line ~215)
  - [ ] Update validation count 3 → 4 (line ~162)
  - [ ] Add money_flows to cleanup queries (line ~177)
  - [ ] Add money_flows to verification query (line ~282)
  - [ ] Update required_columns for features (line ~95)
  - [ ] Optional: Add META.json download logic (line ~30)

---

## Deployment Plan

### Pre-Deployment
1. Coordinate with source system team to ensure parquet files include all columns
2. Verify META.json format includes all 4 files
3. Backup existing database (if any data exists)

### Deployment Steps
1. **Code Mode:** Update schema files (Phase 1)
2. **Code Mode:** Update ingestion script (Phase 2)
3. **Terminal:** Drop old tables
4. **Terminal:** Run init_database.py to create new schemas
5. **Terminal:** Run test ingestion
6. **Verify:** Check data with SQL queries

### Rollback Plan
If deployment fails:
1. Revert schema files to previous version
2. Revert ingestion script changes
3. Re-run init_database.py
4. Re-run ingestion with old data format

---

## Expected Results

### Before Changes
```
❌ ValueError: Parquet validation failed for: alerts.parquet, features.parquet, clusters.parquet
```

### After Changes
```
✅ Downloaded 5 files (1 META.json + 4 parquet)
✅ All parquet files validated successfully
✅ Ingested 86 rows into raw_alerts
✅ Ingested 715 rows into raw_features
✅ Ingested 25 rows into raw_clusters
✅ Ingested 842 rows into raw_money_flows
✅ Ingestion workflow completed successfully
```

### Data Verification
```sql
-- raw_alerts: 86 rows × 16 columns ✅
-- raw_features: 715 rows × 96+ columns ✅ (wide format)
-- raw_clusters: 25 rows × 17 columns ✅
-- raw_money_flows: 842 rows × 19 columns ✅
```

---

## Time Estimate

- **Schema updates:** 30 minutes (straightforward file replacements)
- **Code updates:** 1 hour (small changes, multiple locations)
- **Testing:** 1 hour (verify schemas, test ingestion, check data)
- **Total:** ~2.5 hours

**Much faster than original 2-3 days with transformation layer!**

---

## Success Criteria

- [ ] All 4 schema files exist and match source system
- [ ] Ingestion script handles 4 tables
- [ ] Test ingestion completes without errors
- [ ] raw_features has 96+ columns (wide format)
- [ ] All 4 tables have correct row counts
- [ ] Data types match source system
- [ ] Partitioning works correctly

---

## Next Steps

1. **Review this guide** - Ensure all changes are clear
2. **Switch to Code mode** - Implement schema and code changes
3. **Test locally** - Verify with real S3 data
4. **Deploy** - Update production system

---

**Summary:** This is a straightforward update - no transformation logic needed, just schema alignment and adding the 4th table (money_flows). The implementation is simple and low-risk.