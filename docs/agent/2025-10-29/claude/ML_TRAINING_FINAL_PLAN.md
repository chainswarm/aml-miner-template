# ML Training System - Final Architecture Plan

**Date**: 2025-10-29  
**Status**: ✅ Architecture Complete - Ready for Implementation  
**Mode**: Code Implementation Required

---

## Executive Summary

Comprehensive ML training architecture designed following `packages/ingestion/` conventions. The system extracts data from ClickHouse (ingested by SOTDataIngestion), builds features, trains LightGBM models, and stores versioned models with metadata tracking.

### Key Achievements

✅ **Architecture Design** - Complete system design following ingestion patterns  
✅ **Pipeline Specification** - Detailed 4-phase training workflow  
✅ **Feature Engineering** - Comprehensive feature building strategy  
✅ **Model Training** - LightGBM with cross-validation  
✅ **Model Storage** - Versioning and metadata tracking  
✅ **Integration Plan** - Seamless integration with existing ingestion  

---

## Architecture Overview

### System Structure

```
packages/
├── ingestion/              # ✅ Existing - Data ingestion from S3
│   └── sot_ingestion.py
│
├── storage/                # ✅ Existing - ClickHouse abstraction
│   ├── __init__.py
│   └── schema/
│
└── training/               # 🆕 NEW - ML training pipeline
    ├── __init__.py
    ├── feature_extraction.py   # Extract data from ClickHouse
    ├── feature_builder.py      # Build feature matrix
    ├── model_trainer.py        # Train LightGBM models
    ├── model_storage.py        # Save & version models
    └── model_training.py       # Main orchestration
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    S3 Bucket (SOT)                           │
│  alerts.parquet, features.parquet, clusters.parquet          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ 1. Ingestion (DONE)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    ClickHouse Database                       │
│  raw_alerts, raw_features, raw_clusters, raw_money_flows     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ 2. Training (NEW)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  Training Pipeline                           │
│  Extract → Build Features → Train → Save Model              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  Trained Models                              │
│  alert_scorer_v1.0.0.txt + metadata.json                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Four-Phase Training Pipeline

### Phase 1: Data Extraction
**Module**: `feature_extraction.py`

- Query ClickHouse for training data
- Extract alerts, features, clusters, money_flows
- Filter by date range and network
- Validate data completeness
- **Output**: Dictionary of pandas DataFrames

### Phase 2: Feature Engineering
**Module**: `feature_builder.py`

- Alert-level features (severity, volume, confidence)
- Address-level features (transaction stats, risk scores)
- Temporal features (day of week, is_weekend)
- Statistical features (z-scores, percentiles)
- Cluster features (membership, size, volume)
- Network features (degree, flow statistics)
- **Output**: Feature matrix (X) and labels (y)

### Phase 3: Model Training
**Module**: `model_trainer.py`

- Train/test split (80/20)
- Cross-validation (5-fold stratified)
- LightGBM training with early stopping
- Metric evaluation (AUC, precision, recall, F1)
- **Output**: Trained model + metrics

### Phase 4: Model Storage
**Module**: `model_storage.py`

- Save LightGBM model (.txt format)
- Save metadata (.json format)
- Store tracking info in ClickHouse
- Version management
- **Output**: Model artifacts + database record

---

## Core Components

### 1. FeatureExtractor Class

```python
class FeatureExtractor:
    def extract_training_data(network, start_date, end_date) -> Dict[str, pd.DataFrame]
    def _extract_alerts() -> pd.DataFrame
    def _extract_features() -> pd.DataFrame
    def _extract_clusters() -> pd.DataFrame
    def _extract_money_flows() -> pd.DataFrame
    def _validate_extracted_data()
```

### 2. FeatureBuilder Class

```python
class FeatureBuilder:
    def build_training_features(data) -> Tuple[pd.DataFrame, pd.Series]
    def _add_alert_features() -> pd.DataFrame
    def _add_address_features() -> pd.DataFrame
    def _add_temporal_features() -> pd.DataFrame
    def _add_statistical_features() -> pd.DataFrame
    def _add_cluster_features() -> pd.DataFrame
    def _add_network_features() -> pd.DataFrame
    def _finalize_features() -> pd.DataFrame
```

### 3. ModelTrainer Class

```python
class ModelTrainer:
    def train(X, y, hyperparameters, cv_folds) -> Tuple[lgb.Booster, Dict]
    def _get_default_hyperparameters() -> Dict
    def _evaluate_model() -> Dict
    def _cross_validate() -> Dict
```

### 4. ModelStorage Class

```python
class ModelStorage:
    def save_model(model, model_type, network, metrics, config) -> Path
    def _create_metadata() -> Dict
    def _store_in_clickhouse()
    def load_model(model_path) -> lgb.Booster
    def get_latest_model(model_type, network) -> Path
```

### 5. ModelTraining Orchestrator

```python
class ModelTraining(ABC):
    def run()  # Main workflow
    # Uses all above classes to execute 4-phase pipeline
```

---

## Database Schema

### New Table: trained_models

```sql
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
ORDER BY (network, model_type, created_at);
```

---

## Usage Examples

### Train Alert Scorer

```bash
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer
```

### Train All Model Types

```bash
# Alert scorer
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer

# Alert ranker
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_ranker

# Cluster scorer
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type cluster_scorer
```

### Multi-Network Training

```bash
for network in ethereum bitcoin polygon; do
    python -m packages.training.model_training \
        --network $network \
        --start-date 2024-01-01 \
        --end-date 2024-03-31 \
        --model-type alert_scorer
done
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure

- [ ] Create `packages/training/__init__.py`
- [ ] Implement `packages/training/feature_extraction.py`
- [ ] Implement `packages/training/feature_builder.py`
- [ ] Implement `packages/training/model_trainer.py`
- [ ] Implement `packages/training/model_storage.py`
- [ ] Implement `packages/training/model_training.py`

### Phase 2: Database Schema

- [ ] Create `packages/storage/schema/trained_models.sql`
- [ ] Update `packages/storage/__init__.py` MigrateSchema
- [ ] Test schema creation

### Phase 3: Integration Testing

- [ ] Test data extraction from ClickHouse
- [ ] Test feature building with sample data
- [ ] Test model training with small dataset
- [ ] Test model saving and loading
- [ ] Test metadata storage in ClickHouse
- [ ] Test end-to-end workflow

### Phase 4: Production Readiness

- [ ] Add error handling tests
- [ ] Test terminate_event handling
- [ ] Verify logging standards
- [ ] Test multiple networks
- [ ] Performance optimization
- [ ] Create training documentation

---

## Design Principles Followed

### Convention over Configuration ✅
- Follows exact pattern from [`SOTDataIngestion`](../../../packages/ingestion/sot_ingestion.py:17)
- Similar class structure and method naming
- Consistent use of loguru for logging
- Same error handling approach (fail-fast)

### No Data Migrations ✅
- Always extracts fresh data from ClickHouse
- No fallback values or default behaviors
- Raises exceptions when data missing

### Fail Fast ✅
- Validates data at extraction
- Raises ValueError for missing data
- No try/except with silent failures

### Clean Logging ✅
- Domain-focused messages
- No emoticons in logs
- No step numbers or progress indicators
- Rich context in extra fields

### ClickHouse Native ✅
- Direct SQL queries to ClickHouse
- Efficient data extraction
- Metadata tracking in database
- Model versioning in ClickHouse

---

## Key Differences from Old Code

| Aspect | Old (`alert_scoring/`) | New (`packages/training/`) |
|--------|------------------------|----------------------------|
| **Data Source** | File-based batches | ClickHouse queries |
| **Structure** | Nested modules | Flat, focused modules |
| **Error Handling** | Try/catch with defaults | Fail-fast with exceptions |
| **Logging** | Mixed approaches | Consistent loguru patterns |
| **Storage** | Local files only | Files + ClickHouse tracking |
| **Integration** | Standalone | Integrated with ingestion |
| **Convention** | Custom patterns | Follows ingestion pattern |

---

## Model Metadata Example

```json
{
  "model_id": "alert_scorer_ethereum_v1.0.0_20250129_103045",
  "model_type": "alert_scorer",
  "network": "ethereum",
  "version": "1.0.0",
  "created_at": "2025-01-29T10:30:45.123Z",
  "training_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "window_days": 7
  },
  "data_stats": {
    "num_samples": 125000,
    "num_features": 47,
    "positive_rate": 0.23
  },
  "metrics": {
    "test_auc": 0.8734,
    "test_precision": 0.8421,
    "test_recall": 0.7892,
    "test_f1": 0.8148,
    "cv_auc_mean": 0.8689,
    "cv_auc_std": 0.0043
  },
  "hyperparameters": {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8
  },
  "feature_names": ["alert_confidence_score", "volume_usd_log", "..."],
  "num_trees": 247,
  "best_iteration": 197
}
```

---

## Benefits of This Architecture

✅ **Consistent** - Mirrors ingestion patterns exactly  
✅ **Clean** - Clear separation of concerns  
✅ **Maintainable** - Easy to understand and extend  
✅ **Testable** - Each component independently testable  
✅ **Traceable** - Full logging and metadata tracking  
✅ **Scalable** - Handles multiple networks and date ranges  
✅ **Production-Ready** - Fail-fast, proper error handling  
✅ **Integrated** - Works seamlessly with existing ingestion  

---

## Documentation Created

1. **[ML_TRAINING_ARCHITECTURE.md](ML_TRAINING_ARCHITECTURE.md)** - High-level architecture design
2. **[ML_TRAINING_IMPLEMENTATION_GUIDE.md](ML_TRAINING_IMPLEMENTATION_GUIDE.md)** - Detailed implementation specs
3. **[ML_TRAINING_FINAL_PLAN.md](ML_TRAINING_FINAL_PLAN.md)** - This document

---

## Next Steps for Implementation

### Immediate (Switch to Code Mode)

1. **Create Core Modules**
   ```bash
   mkdir -p packages/training
   touch packages/training/__init__.py
   touch packages/training/feature_extraction.py
   touch packages/training/feature_builder.py
   touch packages/training/model_trainer.py
   touch packages/training/model_storage.py
   touch packages/training/model_training.py
   ```

2. **Create Database Schema**
   ```bash
   touch packages/storage/schema/trained_models.sql
   ```

3. **Implement in Order**
   - Start with `feature_extraction.py` (depends only on ClickHouse)
   - Then `feature_builder.py` (depends on pandas)
   - Then `model_trainer.py` (depends on LightGBM)
   - Then `model_storage.py` (depends on file I/O)
   - Finally `model_training.py` (orchestrates all above)

### Testing Strategy

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test full pipeline with sample data
3. **End-to-End Tests**: Test with real ingested data

### Validation Criteria

✅ Successfully extracts data from ClickHouse  
✅ Builds complete feature matrix  
✅ Trains model with reasonable metrics  
✅ Saves model with proper versioning  
✅ Stores metadata in ClickHouse  
✅ Can load and use saved models  
✅ Handles errors gracefully (fail-fast)  
✅ Logging is consistent and informative  

---

## Success Metrics

After implementation, the system should:

1. **Extract** training data from ClickHouse in < 30 seconds
2. **Build** features for 100K samples in < 60 seconds
3. **Train** model to AUC > 0.85 with 5-fold CV
4. **Save** model with complete metadata
5. **Track** all trained models in ClickHouse
6. **Handle** errors with clear messages
7. **Log** progress with rich context

---

## Conclusion

The ML training architecture is **complete and ready for implementation**. All design documents have been created with detailed specifications for:

- ✅ System architecture
- ✅ Data extraction strategy
- ✅ Feature engineering workflow
- ✅ Model training approach
- ✅ Model storage and versioning
- ✅ ClickHouse integration
- ✅ Error handling patterns
- ✅ Logging standards

**Status**: 🎯 Ready to switch to **Code Mode** for implementation

**Recommendation**: Use the detailed code examples in [ML_TRAINING_IMPLEMENTATION_GUIDE.md](ML_TRAINING_IMPLEMENTATION_GUIDE.md) as the implementation blueprint.
