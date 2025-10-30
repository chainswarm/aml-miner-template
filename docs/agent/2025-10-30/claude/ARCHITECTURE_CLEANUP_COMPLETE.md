# Architecture Cleanup - scripts/ Pattern Implementation

## Date
2025-10-30

## Objective
Enforce the scripts/ pattern throughout the codebase for clean separation between business logic (packages/) and CLI entry points (scripts/).

## Changes Made

### 1. Removed __main__ Blocks from Packages

#### packages/training/model_training.py
- **Removed**: Lines 154-193 (argparse CLI code)
- **Reason**: Business logic should not contain CLI code
- **Impact**: Module is now pure Python class, importable without side effects

#### packages/ingestion/sot_ingestion.py
- **Removed**: Lines 481-540 (argparse CLI code)
- **Reason**: Business logic should not contain CLI code
- **Impact**: Module is now pure Python class, importable without side effects

### 2. Created New Script Wrappers

#### scripts/train_model.py (NEW)
```bash
python scripts/train_model.py \
  --network torus \
  --start-date 2025-08-01 \
  --end-date 2025-08-01 \
  --model-type alert_scorer \
  --window-days 195
```
- **Purpose**: CLI wrapper for ModelTraining class
- **Pattern**: Thin wrapper (50 lines)
- **Imports**: `from packages.training.model_training import ModelTraining`

#### scripts/ingest_data.py (NEW)
```bash
python scripts/ingest_data.py \
  --network torus \
  --processing-date 2025-08-01 \
  --days 195
```
- **Purpose**: CLI wrapper for SOTDataIngestion class
- **Pattern**: Thin wrapper (72 lines)
- **Imports**: `from packages.ingestion.sot_ingestion import SOTDataIngestion`

#### scripts/score_batch.py (ALREADY CREATED)
```bash
python scripts/score_batch.py \
  --network torus \
  --processing-date 2025-08-01 \
  --window-days 195
```
- **Purpose**: CLI wrapper for RiskScoring class
- **Pattern**: Thin wrapper (56 lines)
- **Imports**: `from packages.scoring import RiskScoring`

### 3. Updated docker-compose.yml

#### Model Paths
```yaml
# OLD
- ./trained_models:/app/trained_models:ro
- ALERT_SCORER_PATH=/app/trained_models/alert_scorer_v1.0.0.txt

# NEW
- ./data/trained_models:/app/data/trained_models:ro
- ALERT_SCORER_PATH=/app/data/trained_models/alert_scorer_v1.0.0.txt
```

#### Schema Path
```yaml
# OLD
- ./alert_scoring/storage/schema:/docker-entrypoint-initdb.d:ro

# NEW
- ./packages/storage/schema:/docker-entrypoint-initdb.d:ro
```

## Final Directory Structure

```
packages/
  ingestion/
    sot_ingestion.py              # Pure logic, no CLI
  training/
    model_training.py             # Pure logic, no CLI
    feature_builder.py
    model_storage.py
    strategies/
      xgboost_trainer.py
      address_label_strategy.py
  scoring/                        # NEW
    risk_scoring.py               # Pure logic, no CLI
    model_loader.py
    score_generator.py
    score_writer.py
  storage/
    schema/                       # ClickHouse schemas

scripts/
  ingest_data.py                  # NEW - CLI wrapper
  train_model.py                  # NEW - CLI wrapper
  score_batch.py                  # NEW - CLI wrapper
  download_batch.py               # Existing
  process_batch.py                # Existing
  validate_models.py              # Existing
  init_database.py                # Existing

data/
  trained_models/                 # NEW location
    {network}/
      alert_scorer_*.txt
      alert_ranker_*.txt
      cluster_scorer_*.txt
```

## Benefits of This Architecture

### 1. Clean Separation of Concerns ✅
- **packages/**: Reusable business logic
- **scripts/**: CLI entry points
- Each file has single responsibility

### 2. Better Reusability ✅
- Can import classes without triggering CLI code
- Multiple CLIs can use same logic:
  - `score_batch.py` - batch scoring
  - `score_realtime.py` - streaming (future)
  - Both use `RiskScoring` class

### 3. Easier Discovery ✅
- All executables in `scripts/`
- Clear what can be run vs what is library code
- No need to search for `__main__` blocks

### 4. Cleaner Testing ✅
- Test modules independently
- No CLI argparse logic in unit tests
- Mock CLI arguments easily in scripts

### 5. Better Package Distribution ✅
- Modules are pure Python library
- Scripts are optional CLI tools
- Can use modules in other projects

### 6. Docker-Compose Compatibility ✅
- Updated paths to match new structure
- Models in `data/trained_models/`
- Schema in `packages/storage/schema/`

## Migration Guide

### For Users

**OLD:**
```bash
python packages/training/model_training.py --network torus ...
python packages/ingestion/sot_ingestion.py --network torus ...
```

**NEW:**
```bash
python scripts/train_model.py --network torus ...
python scripts/ingest_data.py --network torus ...
python scripts/score_batch.py --network torus ...
```

### For Developers

**Importing for Reuse:**
```python
# Training
from packages.training.model_training import ModelTraining
training = ModelTraining(network='torus', ...)
training.run()

# Ingestion
from packages.ingestion.sot_ingestion import SOTDataIngestion
ingestion = SOTDataIngestion(network='torus', ...)
ingestion.run()

# Scoring
from packages.scoring import RiskScoring
scoring = RiskScoring(network='torus', ...)
scoring.run()
```

## Verification

### 1. Check No __main__ Blocks in packages/
```bash
grep -r "if __name__ == \"__main__\":" packages/
# Should return nothing
```

### 2. All scripts/ Files Work
```bash
ls scripts/*.py
# Should show all CLI entry points
```

### 3. Docker-Compose Paths Correct
```bash
docker-compose config | grep trained_models
# Should show data/trained_models paths
```

## Files Modified

### Modified
- `packages/training/model_training.py` - Removed __main__ block
- `packages/ingestion/sot_ingestion.py` - Removed __main__ block
- `docker-compose.yml` - Updated paths

### Created
- `scripts/train_model.py` - New CLI wrapper
- `scripts/ingest_data.py` - New CLI wrapper
- `scripts/score_batch.py` - Already created for risk scoring

## Consistency Check

All Python files with `if __main__` blocks:
```
scripts/validate_models.py       ✅ Correct location
scripts/score_batch.py           ✅ Correct location
scripts/process_batch.py         ✅ Correct location
scripts/init_database.py         ✅ Correct location
scripts/download_from_sot.py     ✅ Correct location
scripts/download_batch.py        ✅ Correct location
scripts/train_model.py           ✅ NEW - Correct location
scripts/ingest_data.py           ✅ NEW - Correct location
```

No `__main__` blocks in packages/ anymore! ✅

## Success Criteria

✅ **All CLI code in scripts/**
- No `__main__` blocks in `packages/`
- All CLI entry points in `scripts/`

✅ **Business Logic Separation**
- `packages/` contains pure Python classes
- `scripts/` contains thin wrappers (50-70 lines)

✅ **Reusability**
- Can import classes without side effects
- Multiple scripts can use same classes

✅ **Docker-Compose Updated**
- Correct paths for models and schemas
- Compatible with new structure

✅ **Documentation**
- Clear migration guide
- Usage examples provided

## Status

🎉 **ARCHITECTURE CLEANUP COMPLETE** 🎉

The codebase now follows a clean, consistent architecture pattern with proper separation between business logic and CLI entry points.