# Training Package

Machine learning model training pipeline for the risk scoring system.

## Overview

This package provides a flexible, extensible training pipeline that allows miners to:
- Use SOT's baseline labeled dataset (address_labels)
- Add custom labeled datasets
- Implement custom labeling strategies
- Use any ML algorithm/framework
- Compete based on innovation

## Quick Start

### Basic Training (SOT Baseline)

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195
```

This uses:
- **Labels**: SOT's `raw_address_labels` table
- **Model**: XGBoost classifier
- **Features**: All available features from feature_builder

## Architecture

```
┌─────────────────────────────────────────┐
│         Data Extraction                  │
│  (FeatureExtractor)                      │
│  - Alerts, Features, Clusters, etc.      │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Label Derivation                 │
│  (LabelStrategy)                         │
│  - AddressLabelStrategy (default)        │
│  - Custom strategies (miner extension)   │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Feature Building                 │
│  (FeatureBuilder)                        │
│  - Merge alerts + features               │
│  - Engineer features                     │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Model Training                   │
│  (ModelTrainer)                          │
│  - XGBoostTrainer (default)              │
│  - Custom trainers (miner extension)     │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Model Storage                    │
│  (ModelStorage)                          │
│  - Save to disk                          │
│  - Store metadata in ClickHouse          │
└─────────────────────────────────────────┘
```

## Components

### 1. FeatureExtractor

Extracts training data from ClickHouse.

**Extracted Data:**
- Alerts (from raw_alerts)
- Features (from raw_features)
- Clusters (from raw_clusters)
- Money Flows (from raw_money_flows)
- Address Labels (from raw_address_labels) ← **Ground Truth Labels**

### 2. LabelStrategy

Derives training labels from data.

**Default (AddressLabelStrategy):**
- Uses `raw_address_labels` table
- Maps `risk_level` to binary labels:
  - high/critical → 1 (suspicious)
  - low/medium → 0 (normal)
- Uses `confidence_score` as sample weights

**Custom Options:**
- Add proprietary labeled datasets
- Combine multiple label sources
- Implement custom labeling logic

See [strategies/README.md](strategies/README.md) for details.

### 3. FeatureBuilder

Builds feature matrix for training.

**Process:**
1. Derive labels using LabelStrategy
2. Filter to labeled alerts only
3. Add alert-level features (severity, volume, etc.)
4. Add address-level features (from raw_features)
5. Add temporal features (day of week, etc.)
6. Add statistical features (z-scores, percentiles)
7. Add cluster features (if available)
8. Add network features (if available)
9. Add label features (from address_labels)
10. Finalize (drop non-numeric, handle missing values)

**Output:**
- X: Feature matrix (numeric only)
- y: Binary labels (0/1)

### 4. ModelTrainer

Trains ML model.

**Default (XGBoostTrainer):**
- XGBoost classifier
- Binary classification
- Cross-validation support
- Returns AUC and PR-AUC metrics

**Custom Options:**
- Neural networks
- LightGBM
- Ensemble methods
- Any sklearn-compatible model

See [strategies/README.md](strategies/README.md) for details.

### 5. ModelStorage

Saves trained models and metadata.

**Saves:**
- Model file (to disk)
- Model metadata (to ClickHouse trained_models table)
- Training config
- Performance metrics

## Customization

### Level 1: Add Custom Dataset

```python
import pandas as pd
from packages.storage import ClientFactory, get_connection_params

# Your labeled addresses
custom_labels = pd.DataFrame({
    'processing_date': ['2025-08-01'] * 100,
    'window_days': [195] * 100,
    'network': ['torus'] * 100,
    'address': [...],
    'label': [...],
    'risk_level': ['high', 'low', ...],
    'confidence_score': [0.9, 0.8, ...],
    'source': 'miner_custom'
})

# Insert into database
connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    client.insert_df('raw_address_labels', custom_labels)

# Train with combined dataset (SOT + yours)
# python packages/training/model_training.py ...
```

### Level 2: Custom Label Strategy

```python
from packages.training.strategies import LabelStrategy

class MyLabelStrategy(LabelStrategy):
    def derive_labels(self, alerts_df, data):
        # Your custom logic
        return alerts_df
    
    def validate_labels(self, alerts_df):
        return True

# Use in training
from packages.training.model_training import ModelTraining

training = ModelTraining(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    client=client,
    label_strategy=MyLabelStrategy()
)
training.run()
```

### Level 3: Custom Model Trainer

```python
from packages.training.strategies import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def train(self, X, y, sample_weights=None):
        # Your training logic
        return model
    
    def predict(self, X):
        return predictions
    
    def evaluate(self, X, y):
        return {'auc': 0.85}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass

# Use in training
training = ModelTraining(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    client=client,
    model_trainer=MyModelTrainer()
)
training.run()
```

### Level 4: Both Custom Labels + Custom Models

```python
training = ModelTraining(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    client=client,
    label_strategy=MyLabelStrategy(),
    model_trainer=MyModelTrainer()
)
training.run()
```

## Documentation

- **[Miner Customization Guide](../../docs/MINER_CUSTOMIZATION_GUIDE.md)** - Complete customization examples
- **[Strategies README](strategies/README.md)** - Strategy pattern details
- **[Architecture Document](../../docs/agent/2025-10-29/claude/TRAINING_LABELS_STRATEGY_FINAL.md)** - Design decisions

## Files

```
packages/training/
├── __init__.py
├── README.md (this file)
├── model_training.py          # Main training orchestrator
├── feature_extraction.py      # Extract data from ClickHouse
├── feature_builder.py         # Build feature matrix
├── model_storage.py           # Save models and metadata
├── model_trainer.py           # Legacy trainer (deprecated)
└── strategies/
    ├── __init__.py
    ├── README.md
    ├── base.py                # Abstract base classes
    ├── address_label_strategy.py  # Default label strategy
    └── xgboost_trainer.py     # Default model trainer
```

## Requirements

- pandas
- numpy
- xgboost
- scikit-learn
- loguru
- clickhouse-connect

## Examples

### Example 1: Train with Different Date Range

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-07-01 \
    --end-date 2025-07-31 \
    --model-type alert_scorer \
    --window-days 195
```

### Example 2: Train Different Model Type

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_ranker \
    --window-days 195
```

### Example 3: Custom Output Directory

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195 \
    --output-dir ./my_models
```

## Troubleshooting

### No labeled alerts found

**Problem:** `ValueError: No labeled alerts found`

**Solution:** 
- Check if `raw_address_labels` table has data
- Verify date range matches available data
- Check network parameter matches data

```sql
-- Check address_labels
SELECT COUNT(*) FROM raw_address_labels 
WHERE processing_date = '2025-08-01' AND window_days = 195;
```

### Feature extraction errors

**Problem:** Column mismatch errors

**Solution:**
- Verify schema matches expected columns
- Check that all required tables exist
- Run schema initialization if needed

```bash
python scripts/init_database.py
```

### Low model performance

**Problem:** AUC < 0.6

**Potential causes:**
- Insufficient labeled data
- Label quality issues
- Feature engineering needed
- Hyperparameter tuning needed

**Solutions:**
- Add more labeled addresses to `raw_address_labels`
- Implement custom LabelStrategy with better labels
- Implement custom ModelTrainer with better model
- Add custom features in FeatureBuilder

## Performance Tips

1. **Label Quality > Model Complexity**
   - Focus on high-quality labels first
   - Use confidence scores as weights
   - Validate labels carefully

2. **Feature Engineering**
   - Add domain-specific features
   - Use cross-network signals
   - Engineer temporal patterns

3. **Model Selection**
   - Start with XGBoost baseline
   - Try LightGBM for speed
   - Use neural nets for complex patterns
   - Ensemble multiple models

4. **Hyperparameter Tuning**
   - Use cross-validation
   - Grid search or Bayesian optimization
   - Track experiments carefully

## License

See project root LICENSE file.