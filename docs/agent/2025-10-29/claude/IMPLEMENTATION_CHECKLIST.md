# Implementation Checklist

**Date:** 2025-10-29  
**Project:** Metadata-Driven Ingestion System  
**Mode:** Code Implementation Guide

## Overview

This checklist breaks down the implementation into manageable tasks. Each task should be completed and tested before moving to the next.

## Phase 1: Core Components

### Task 1.1: Create MetadataExtractor Class
- [ ] Create file `packages/ingestion/metadata_extractor.py`
- [ ] Implement `__init__(meta_path: Path)` method
- [ ] Implement `load() -> Dict` method
- [ ] Implement `_validate_structure() -> None` method
- [ ] Implement `get_processing_date() -> str` method
- [ ] Implement `get_window_days() -> int` method
- [ ] Implement `get_network() -> str` method
- [ ] Implement `get_batch_id() -> str` method
- [ ] Implement `get_expected_count(table: str) -> int` method
- [ ] Implement `get_checksum(filename: str) -> str` method
- [ ] Implement `_ensure_loaded() -> None` helper method
- [ ] Add proper logging with loguru
- [ ] Add docstrings (documentation only, no reasoning)

**Files to modify:**
- Create: `packages/ingestion/metadata_extractor.py`

**Dependencies:**
- pathlib.Path
- typing.Dict
- json
- loguru.logger

---

### Task 1.2: Create FileValidator Class
- [ ] Create file `packages/ingestion/file_validator.py`
- [ ] Implement `__init__(expected_checksums: Dict[str, str])` method
- [ ] Implement `calculate_sha256(file_path: Path) -> str` method
- [ ] Implement `validate_checksum(file_path: Path) -> bool` method
- [ ] Implement `validate_all(file_paths: List[Path]) -> None` method
- [ ] Add proper logging with loguru
- [ ] Add error handling for file not found
- [ ] Add docstrings (documentation only, no reasoning)

**Files to modify:**
- Create: `packages/ingestion/file_validator.py`

**Dependencies:**
- pathlib.Path
- typing.Dict, List
- hashlib
- loguru.logger

---

### Task 1.3: Create DataTransformer Class
- [ ] Create file `packages/ingestion/data_transformer.py`
- [ ] Implement `__init__(processing_date: str, window_days: int)` method
- [ ] Implement `transform_alerts(file_path: Path) -> pd.DataFrame` method
- [ ] Implement `transform_clusters(file_path: Path) -> pd.DataFrame` method
- [ ] Implement `transform_features(file_path: Path) -> pd.DataFrame` method
- [ ] Implement `_validate_alerts_schema(df: pd.DataFrame) -> None` method
- [ ] Implement `_validate_clusters_schema(df: pd.DataFrame) -> None` method
- [ ] Implement `_validate_features_schema(df: pd.DataFrame) -> None` method
- [ ] Add proper logging with loguru
- [ ] Add error handling for schema mismatches
- [ ] Add docstrings (documentation only, no reasoning)

**Files to modify:**
- Create: `packages/ingestion/data_transformer.py`

**Dependencies:**
- pathlib.Path
- typing.Optional
- pandas as pd
- loguru.logger

---

## Phase 2: Integration

### Task 2.1: Update SOTDataIngestion Download Logic
- [ ] Modify [`_download_all()`](../../../packages/ingestion/sot_ingestion.py:30) method
- [ ] Add META.json download as first step
- [ ] Update file filtering to include .json files
- [ ] Add logging for META.json download
- [ ] Update downloaded count to include META.json
- [ ] Test download with real S3 data

**Files to modify:**
- Modify: `packages/ingestion/sot_ingestion.py` (lines 30-70)

**Key changes:**
```python
# Before downloading parquets, download META.json first
meta_key = f"{self.s3_prefix}/META.json"
# Validate META.json exists
# Download META.json
# Then download parquet files
```

---

### Task 2.2: Add Transformation Steps to run() Method
- [ ] Import new classes at top of file
- [ ] Add Step 4: Extract metadata (after download)
- [ ] Add Step 5: Validate checksums
- [ ] Add Step 6: Transform data
- [ ] Update Step numbering (now 1-7 instead of 1-6)
- [ ] Add error handling for each new step
- [ ] Add termination checks between steps
- [ ] Update logging messages

**Files to modify:**
- Modify: `packages/ingestion/sot_ingestion.py` (lines 1-12 for imports, 117-325 for run method)

**Key changes:**
```python
# Add imports
from packages.ingestion.metadata_extractor import MetadataExtractor
from packages.ingestion.file_validator import FileValidator
from packages.ingestion.data_transformer import DataTransformer

# In run() method, after download:
# Step 4: Extract metadata
# Step 5: Validate checksums
# Step 6: Transform data
# Step 7: Ingest (previously Step 5)
```

---

### Task 2.3: Update Ingestion Logic
- [ ] Modify ingestion loop to use transformed data instead of original files
- [ ] Save transformed DataFrames to temporary parquet files
- [ ] Update `insert_file` calls to use temporary files
- [ ] Add cleanup for temporary files after successful ingestion
- [ ] Update logging to show transformation statistics

**Files to modify:**
- Modify: `packages/ingestion/sot_ingestion.py` (lines 249-273)

**Key changes:**
```python
# Instead of ingesting from original files:
for table, df in transformed_data.items():
    temp_file = self.local_dir / f"{table}_transformed.parquet"
    df.to_parquet(temp_file, index=False)
    self.client.insert_file(table=table, file_path=str(temp_file), fmt='Parquet')
    temp_file.unlink()  # Clean up
```

---

### Task 2.4: Remove Old Validation Logic
- [ ] Remove `_validate_parquet_file()` method (lines 72-115)
- [ ] Remove validation step from old location (lines 211-243)
- [ ] Update step numbers in logging messages
- [ ] Ensure no references to removed method

**Files to modify:**
- Modify: `packages/ingestion/sot_ingestion.py` (remove lines 72-115, 211-243)

**Rationale:**
- Schema validation now happens after transformation
- Checksum validation replaces parquet validation

---

## Phase 3: Testing

### Task 3.1: Unit Tests for MetadataExtractor
- [ ] Create `tests/ingestion/test_metadata_extractor.py`
- [ ] Test `load()` with valid META.json
- [ ] Test `load()` with missing file
- [ ] Test `load()` with invalid structure
- [ ] Test `get_processing_date()`
- [ ] Test `get_window_days()`
- [ ] Test `get_checksum()` with valid filename
- [ ] Test `get_checksum()` with invalid filename
- [ ] Test `_ensure_loaded()` raises error before load

**Files to create:**
- Create: `tests/ingestion/test_metadata_extractor.py`
- Create: `tests/ingestion/fixtures/valid_META.json`
- Create: `tests/ingestion/fixtures/invalid_META.json`

---

### Task 3.2: Unit Tests for FileValidator
- [ ] Create `tests/ingestion/test_file_validator.py`
- [ ] Test `calculate_sha256()` with known file
- [ ] Test `validate_checksum()` with matching checksum
- [ ] Test `validate_checksum()` with mismatching checksum
- [ ] Test `validate_all()` with all valid files
- [ ] Test `validate_all()` with one invalid file
- [ ] Test `validate_checksum()` with missing file

**Files to create:**
- Create: `tests/ingestion/test_file_validator.py`
- Create: `tests/ingestion/fixtures/test.parquet`

---

### Task 3.3: Unit Tests for DataTransformer
- [ ] Create `tests/ingestion/test_data_transformer.py`
- [ ] Test `transform_alerts()` adds metadata columns
- [ ] Test `transform_clusters()` adds metadata columns
- [ ] Test `transform_features()` wide-to-long conversion
- [ ] Test `transform_features()` row count expansion
- [ ] Test schema validation for each table
- [ ] Test null value validation
- [ ] Test missing column validation

**Files to create:**
- Create: `tests/ingestion/test_data_transformer.py`
- Create: `tests/ingestion/fixtures/alerts.parquet`
- Create: `tests/ingestion/fixtures/features.parquet`
- Create: `tests/ingestion/fixtures/clusters.parquet`

---

### Task 3.4: Integration Test
- [ ] Create `tests/ingestion/test_sot_ingestion_integration.py`
- [ ] Test full pipeline with real S3 data structure
- [ ] Test metadata extraction → validation → transformation → ingestion
- [ ] Test error handling at each stage
- [ ] Test termination handling
- [ ] Verify final database state

**Files to create:**
- Create: `tests/ingestion/test_sot_ingestion_integration.py`

---

## Phase 4: Documentation

### Task 4.1: Update Code Documentation
- [ ] Add module docstring to `metadata_extractor.py`
- [ ] Add module docstring to `file_validator.py`
- [ ] Add module docstring to `data_transformer.py`
- [ ] Update docstring for `SOTDataIngestion` class
- [ ] Update docstring for `run()` method

**Files to modify:**
- Modify: All new files and `sot_ingestion.py`

---

### Task 4.2: Update README
- [ ] Document new transformation layer
- [ ] Add section on META.json requirements
- [ ] Update system requirements
- [ ] Add troubleshooting guide for common errors

**Files to modify:**
- Modify: `README.md`

---

## Phase 5: Deployment Validation

### Task 5.1: Local Testing
- [ ] Test with torus network data
- [ ] Verify all checksums pass
- [ ] Verify features transformation (715 → 67,925 rows)
- [ ] Verify alerts transformation (86 → 86 rows + metadata)
- [ ] Verify clusters transformation (25 → 25 rows + metadata)
- [ ] Check ClickHouse data integrity
- [ ] Verify partitions created correctly

---

### Task 5.2: Error Recovery Testing
- [ ] Test with missing META.json
- [ ] Test with corrupted parquet file
- [ ] Test with checksum mismatch
- [ ] Test with invalid metadata structure
- [ ] Test with missing required columns
- [ ] Test with null values in required fields
- [ ] Verify error messages are clear

---

### Task 5.3: Performance Validation
- [ ] Measure transformation time
- [ ] Measure memory usage during features transformation
- [ ] Verify no memory leaks
- [ ] Check temporary file cleanup
- [ ] Validate database insert performance

---

## Implementation Order

Execute tasks in this order for optimal workflow:

1. **Phase 1:** Implement all three core components (Tasks 1.1, 1.2, 1.3)
2. **Phase 3 (Early):** Write unit tests (Tasks 3.1, 3.2, 3.3) and run them
3. **Phase 2:** Integrate components (Tasks 2.1, 2.2, 2.3, 2.4)
4. **Phase 3 (Late):** Integration testing (Task 3.4)
5. **Phase 4:** Documentation (Tasks 4.1, 4.2)
6. **Phase 5:** Validation (Tasks 5.1, 5.2, 5.3)

## File Structure

After implementation, the structure should be:

```
packages/ingestion/
├── __init__.py
├── sot_ingestion.py           # Modified
├── metadata_extractor.py      # New
├── file_validator.py          # New
└── data_transformer.py        # New

tests/ingestion/
├── __init__.py
├── test_metadata_extractor.py # New
├── test_file_validator.py     # New
├── test_data_transformer.py   # New
├── test_sot_ingestion_integration.py # New
└── fixtures/
    ├── valid_META.json
    ├── invalid_META.json
    ├── alerts.parquet
    ├── features.parquet
    ├── clusters.parquet
    └── test.parquet
```

## Dependencies

Add to `requirements.txt` if not present:
```
pandas>=2.0.0
pyarrow>=12.0.0
boto3>=1.28.0
clickhouse-connect>=0.6.0
loguru>=0.7.0
python-dotenv>=1.0.0
```

## Success Criteria

Implementation is complete when:

- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Real data ingestion works (torus network)
- [ ] Features correctly transform from 96 cols → 4 cols
- [ ] Metadata columns correctly added to alerts and clusters
- [ ] Checksums validate successfully
- [ ] Database queries return expected row counts
- [ ] No temporary files left after ingestion
- [ ] Error messages are clear and actionable
- [ ] Documentation is complete and accurate

## Rollback Plan

If implementation fails:
1. Keep original `sot_ingestion.py` backed up
2. Can revert to original by removing new imports and steps
3. Database schema unchanged, so no migration needed
4. S3 data unchanged, so safe to retry

## Notes

- Use loguru for all logging
- Raise exceptions immediately, no default values
- No method/class documentation with reasoning
- Focus on domain-specific log messages
- Clean up temporary files in all code paths
- Use context managers for file operations
- Validate data at transformation boundaries