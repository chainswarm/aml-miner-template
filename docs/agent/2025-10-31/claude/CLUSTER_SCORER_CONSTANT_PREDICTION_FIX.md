# Cluster Scorer Constant Prediction Bug - Root Cause & Fix

## Problem Summary

The cluster_scorer model produces constant predictions (0.044102 for all inputs) despite diverse feature inputs.

## Root Cause Analysis

### Model Metadata Investigation

Checking the trained model metadata at [`data/trained_models/torus/cluster_scorer_torus_v1.0.0_2025-08-01_2025-08-01_w195d_20251031_114740.json`](data/trained_models/torus/cluster_scorer_torus_v1.0.0_2025-08-01_2025-08-01_w195d_20251031_114740.json):

```json
{
  "model_type": "cluster_scorer",
  "data_stats": {
    "num_samples": 23,           ← Only 23 training samples
    "num_features": 245,
    "positive_rate": 0.0         ← ZERO positive labels!
  },
  "metrics": {
    "auc": 0.0,                  ← No predictive power
    "pr_auc": 0.0
  }
}
```

### Label Derivation Flow

1. **Alert Labels** (from [`_derive_labels_from_address_labels()`](packages/training/feature_builder.py:567-608)):
   ```python
   if risk in ['high', 'critical']:
       label_map[addr] = 1  # Positive
   elif risk in ['low', 'medium']:
       label_map[addr] = 0  # Negative
   ```

2. **Cluster Labels** (from [`_derive_cluster_labels()`](packages/training/feature_builder.py:221-271)):
   ```python
   # Cluster gets positive label if ANY alert is positive
   cluster_label = int(cluster_alert_labels.max())
   ```

### The Issue

The `raw_address_labels` table contains **ZERO addresses** with `risk_level IN ('high', 'critical')`:
- All addresses have `risk_level IN ('low', 'medium')`  
- Therefore: All alerts get label=0
- Therefore: All clusters get label=0
- Model trains with 100% negative samples
- Model learns to always predict the base probability

### Why 0.044102?

XGBoost's base_score parameter defaults to 0.5, but with:
- `objective='binary:logistic'`
- 100% negative training data (all y=0)
- The model learns: `sigmoid(log_odds)` where the log_odds shifts to produce ~0.044

This is the learned "always predict negative" constant.

## Solution

### Quick Fix: Add Positive Training Labels

**Option 1: Add synthetic high-risk addresses**

```python
import pandas as pd
from packages.storage import ClientFactory, get_connection_params

# Create synthetic high-risk labels
high_risk_addresses = pd.DataFrame({
    'processing_date': ['2025-08-01'] * 20,
    'window_days': [195] * 20,
    'network': ['torus'] * 20,
    'address': [f'0xHIGHRISK{i:03d}' for i in range(20)],  # Synthetic addresses
    'label': [1] * 20,
    'risk_level': ['high'] * 15 + ['critical'] * 5,
    'confidence_score': [0.8, 0.9] * 10,
    'source': 'synthetic_training_data',
    'labeled_at': ['2025-10-31T13:00:00'] * 20
})

# Insert into database
connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    client.execute('INSERT INTO raw_address_labels VALUES', high_risk_addresses.to_dict('records'))
```

**Option 2: Use real addresses from alerts with high severity**

```python
# Query high-severity alerts and promote them to training labels
query = """
SELECT DISTINCT
    processing_date,
    window_days,
    network,
    address,
    1 as label,
    CASE 
        WHEN severity = 'critical' THEN 'critical'
        ELSE 'high'
    END as risk_level,
    alert_confidence_score as confidence_score,
    'auto_labeled_from_alerts' as source,
    now() as labeled_at
FROM raw_alerts
WHERE processing_date = '2025-08-01'
  AND window_days = 195
  AND network = 'torus'
  AND severity IN ('high', 'critical')
  AND alert_confidence_score >= 0.7
LIMIT 30
"""

# Insert these as positive training labels
```

### Retrain Model

After adding positive labels:

```bash
python scripts/train_model.py \
  --network torus \
  --start-date 2025-08-01 \
  --end-date 2025-08-01 \
  --window-days 195 \
  --model-type cluster_scorer
```

Expected outcome:
- `positive_rate`: 20-40% (balanced dataset)
- `auc`: > 0.7 (predictive model)
- Predictions: Diverse scores across [0, 1]

## Prevention

### Validation in Training Pipeline

Add check in [`model_training.py`](packages/training/model_training.py):

```python
# After deriving labels, before training
positive_rate = y.mean()

if positive_rate == 0.0:
    raise ValueError(
        "Training data has ZERO positive labels! "
        "Cannot train binary classifier. "
        "Add high/critical risk addresses to raw_address_labels."
    )

if positive_rate < 0.05 or positive_rate > 0.95:
    logger.warning(
        f"Imbalanced training data: positive_rate={positive_rate:.2%}. "
        "Model may perform poorly. Recommended range: 10-90%"
    )
```

### Documentation Update

Update [`docs/training_guide.md`](docs/training_guide.md) to emphasize:

1. **Minimum label requirements**:
   - At least 10-20% positive labels
   - Minimum 50-100 samples per class
   
2. **Label quality checklist**:
   - Check `positive_rate` before training
   - Validate label distribution
   - Ensure both classes represented

## Summary

**Root Cause**: Zero positive training labels (all risk_level='low'/'medium')  
**Symptom**: Model predicts constant 0.044102 for all inputs  
**Fix**: Add positive labels (high/critical risk addresses) and retrain  
**Prevention**: Add validation checks in training pipeline  

The model is NOT broken - it learned perfectly from broken training data!