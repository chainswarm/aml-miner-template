# Implementation Plan: Feature Ordering Bug Fix

## Quick Summary

**Problem**: `label_confidence` feature is added at different positions during training vs inference, causing XGBoost to reject predictions with "feature_names mismatch" error.

**Root Cause**: The feature is added twice during training (once in label derivation, once in feature engineering) but only once during inference.

**Solution**: Remove the duplicate assignment in `_derive_labels_from_address_labels()` method.

---

## Implementation Steps

### Step 1: Update FeatureBuilder

**File**: `packages/training/feature_builder.py`

**Method**: `_derive_labels_from_address_labels()` (lines 107-148)

**Changes**:

```python
# BEFORE (lines 107-148)
def _derive_labels_from_address_labels(
    self,
    alerts_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    
    logger.info("Deriving labels from address_labels table")
    
    if labels_df.empty:
        raise ValueError("address_labels table is empty")
    
    label_map = {}
    confidence_map = {}  # ← REMOVE THIS LINE
    
    for _, row in labels_df.iterrows():
        addr = row['address']
        risk = row['risk_level'].lower()
        confidence = row.get('confidence_score', 1.0)  # ← REMOVE THIS LINE
        
        if risk in ['high', 'critical']:
            label_map[addr] = 1
            confidence_map[addr] = confidence  # ← REMOVE THIS LINE
        elif risk in ['low', 'medium']:
            label_map[addr] = 0
            confidence_map[addr] = confidence  # ← REMOVE THIS LINE
    
    alerts_df['label'] = alerts_df['address'].map(label_map)
    alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)  # ← REMOVE THIS LINE
    alerts_df['label_source'] = alerts_df['address'].map(
        lambda x: 'address_labels' if x in label_map else None
    )
    
    num_labeled = alerts_df['label'].notna().sum()
    num_positive = (alerts_df['label'] == 1).sum()
    num_negative = (alerts_df['label'] == 0).sum()
    
    logger.info(
        f"Labeled {num_labeled}/{len(alerts_df)} alerts: "
        f"{num_positive} positive, {num_negative} negative"
    )
    
    return alerts_df
```

```python
# AFTER (cleaned up)
def _derive_labels_from_address_labels(
    self,
    alerts_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    
    logger.info("Deriving labels from address_labels table")
    
    if labels_df.empty:
        raise ValueError("address_labels table is empty")
    
    label_map = {}
    
    for _, row in labels_df.iterrows():
        addr = row['address']
        risk = row['risk_level'].lower()
        
        if risk in ['high', 'critical']:
            label_map[addr] = 1
        elif risk in ['low', 'medium']:
            label_map[addr] = 0
    
    alerts_df['label'] = alerts_df['address'].map(label_map)
    alerts_df['label_source'] = alerts_df['address'].map(
        lambda x: 'address_labels' if x in label_map else None
    )
    
    num_labeled = alerts_df['label'].notna().sum()
    num_positive = (alerts_df['label'] == 1).sum()
    num_negative = (alerts_df['label'] == 0).sum()
    
    logger.info(
        f"Labeled {num_labeled}/{len(alerts_df)} alerts: "
        f"{num_positive} positive, {num_negative} negative"
    )
    
    return alerts_df
```

**Lines to Remove**:
- Line 119: `confidence_map = {}`
- Line 124: `confidence = row.get('confidence_score', 1.0)`
- Line 128: `confidence_map[addr] = confidence`
- Line 131: `confidence_map[addr] = confidence`
- Line 134: `alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)`

**Why**: `label_confidence` will still be added by `_add_label_features()` method (line 354-355), which is called consistently in both training and inference flows.

---

### Step 2: Verify AddressLabelStrategy Still Works

**File**: `packages/training/strategies/address_label_strategy.py`

**Method**: `get_sample_weights()` (lines 76-91)

**Current Code** (lines 84-85):
```python
if self.use_confidence_weights and 'label_confidence' in alerts_df.columns:
    weights = alerts_df['label_confidence'].fillna(1.0)
```

**Verification**: This should still work because:
1. `build_training_features()` calls `_add_label_features()` (line 82-83)
2. `_add_label_features()` adds `label_confidence` (line 354-355)
3. By the time `get_sample_weights()` is called, `label_confidence` will exist in the dataframe

**No changes needed** - just verify it works after Step 1.

---

### Step 3: Delete Old Models

**Why**: Models trained before the fix have different feature ordering and will fail to load.

**Command**:
```bash
# From project root
rm -rf data/trained_models/torus/alert_scorer_*.txt
rm -rf data/trained_models/torus/alert_scorer_*.json
rm -rf data/trained_models/torus/alert_ranker_*.txt
rm -rf data/trained_models/torus/alert_ranker_*.json
rm -rf data/trained_models/torus/cluster_scorer_*.txt
rm -rf data/trained_models/torus/cluster_scorer_*.json
```

Or delete the entire network directory:
```bash
rm -rf data/trained_models/torus/
```

---

### Step 4: Retrain Models

**Run training for each model type**:

```bash
# From project root
python scripts/examples/example_train.py
```

Or manually:
```bash
python scripts/train_model.py --network torus --model-type alert_scorer --start-date 2025-08-01 --end-date 2025-08-01 --window-days 195
```

**Expected Output**:
- No warnings about feature mismatch
- Model saves successfully to `data/trained_models/torus/`
- Metadata stored in ClickHouse `trained_models` table

---

### Step 5: Test Inference

**Run scoring pipeline**:

```bash
# From project root
python scripts/examples/example_score.py
```

Or manually:
```bash
python scripts/score_batch.py --network torus --processing-date 2025-08-01 --window-days 195
```

**Expected Output**:
- Model loads successfully (no feature_names mismatch error)
- Predictions generated for all alerts
- Scores written to ClickHouse `alert_scores` table
- Batch metadata updated successfully

---

### Step 6: Verify Full Pipeline

**Run complete pipeline**:

```bash
# From project root
python scripts/examples/example_full_pipeline.py
```

**Expected Output**:
```
================================================================================
STEP 1/3: Data Ingestion
================================================================================
✅ Data ingestion completed

================================================================================
STEP 2/3: Model Training
================================================================================
✅ Training workflow completed successfully

================================================================================
STEP 3/3: Risk Scoring
================================================================================
✅ Risk scoring workflow completed successfully
✅ Scores written to ClickHouse
```

**No errors should occur.**

---

## Testing Checklist

- [ ] Code changes applied to `feature_builder.py`
- [ ] Old models deleted
- [ ] New models trained successfully
- [ ] Inference runs without feature mismatch errors
- [ ] Full pipeline completes successfully
- [ ] Feature order is consistent between training and inference
- [ ] Sample weights still work correctly

---

## Validation

### Unit Test: Feature Order Consistency

Create test to verify feature order matches:

```python
# tests/test_feature_ordering.py

import pandas as pd
from packages.training.feature_builder import FeatureBuilder

def test_feature_order_consistency():
    """Verify training and inference produce same feature order"""
    
    # Mock data
    data = {
        'alerts': pd.DataFrame({...}),
        'features': pd.DataFrame({...}),
        'clusters': pd.DataFrame({...}),
        'money_flows': pd.DataFrame({...}),
        'address_labels': pd.DataFrame({...})
    }
    
    builder = FeatureBuilder()
    
    # Build training features
    X_train, y_train = builder.build_training_features(data)
    train_feature_order = list(X_train.columns)
    
    # Build inference features
    X_infer = builder.build_inference_features(data)
    infer_feature_order = list(X_infer.columns)
    
    # Assert same order
    assert train_feature_order == infer_feature_order, \
        f"Feature order mismatch:\nTrain: {train_feature_order}\nInfer: {infer_feature_order}"
    
    # Assert label_confidence exists
    assert 'label_confidence' in train_feature_order
    assert 'label_confidence' in infer_feature_order
    
    # Assert label_confidence at same position
    train_pos = train_feature_order.index('label_confidence')
    infer_pos = infer_feature_order.index('label_confidence')
    assert train_pos == infer_pos, \
        f"label_confidence position mismatch: train={train_pos}, infer={infer_pos}"
```

---

## Rollback Plan

If the fix causes issues:

1. **Revert code changes**:
   ```bash
   git checkout packages/training/feature_builder.py
   ```

2. **Restore old models** (if backed up):
   ```bash
   cp -r data/trained_models_backup/torus/ data/trained_models/
   ```

3. **Report issue** with specific error messages

---

## Success Metrics

The fix is successful when:

1. ✅ No `feature_names mismatch` errors during inference
2. ✅ `label_confidence` appears at the same position in training and inference
3. ✅ Full pipeline runs without errors
4. ✅ Model performance is unchanged (AUC, PR-AUC)
5. ✅ All 86 alerts receive risk scores

---

## Notes

- This is a **critical bug fix** - blocks production deployment
- **No breaking API changes** - internal implementation only
- **Minimal code changes** - only removes 5 lines
- **Must retrain models** after applying fix
- **Test thoroughly** before deploying to production

---

## Related Documentation

- [Feature Ordering Bug Fix Analysis](FEATURE_ORDERING_BUG_FIX.md)
- [Training Architecture](../2025-10-29/claude/ML_TRAINING_ARCHITECTURE.md)
- [Feature Builder Implementation](../../packages/training/feature_builder.py)