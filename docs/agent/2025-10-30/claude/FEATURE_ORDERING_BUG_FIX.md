# Feature Ordering Bug Fix

## Problem Analysis

### Error Message
```
ValueError: feature_names mismatch:
Training: ['window_days', 'alert_confidence_score', 'volume_usd', 'label_confidence', 'severity_encoded', ...]
Inference: ['window_days', 'alert_confidence_score', 'volume_usd', 'severity_encoded', 'volume_usd_log', ...]
```

### Root Cause

The `label_confidence` feature appears at position 4 during training but at position 56 during inference, causing XGBoost model loading to fail.

**Training Flow** ([`feature_builder.py:45-105`](../../packages/training/feature_builder.py#L45-L105)):

```python
def build_training_features(data):
    # Step 1: Derive labels (line 52-55)
    alerts_with_labels = self._derive_labels_from_address_labels(
        data['alerts'],
        data['address_labels']
    )
    # ⚠️ This adds 'label_confidence' column at line 134
    
    # Step 2-6: Add various features (lines 71-83)
    X = self._add_alert_features(X)           # Adds severity_encoded, volume_usd_log, etc.
    X = self._add_address_features(...)
    X = self._add_temporal_features(X)
    X = self._add_statistical_features(X)
    X = self._add_cluster_features(...)
    X = self._add_network_features(...)
    
    # Step 7: Add label features (line 82-83)
    X = self._add_label_features(X, data['address_labels'])
    # ⚠️ This OVERWRITES 'label_confidence' at line 354-355
```

**Inference Flow** ([`feature_builder.py:10-43`](../../packages/training/feature_builder.py#L10-L43)):

```python
def build_inference_features(data):
    X = data['alerts'].copy()
    # ⚠️ NO label derivation step
    
    # Steps 1-5: Add various features (lines 19-28)
    X = self._add_alert_features(X)           # Adds severity_encoded, volume_usd_log, etc.
    X = self._add_address_features(...)
    X = self._add_temporal_features(X)
    X = self._add_statistical_features(X)
    X = self._add_cluster_features(...)
    X = self._add_network_features(...)
    
    # Step 6: Add label features (line 30-31)
    X = self._add_label_features(X, data['address_labels'])
    # ⚠️ This is the FIRST TIME 'label_confidence' is added
```

### The Problem

1. **During training**: `label_confidence` is added TWICE:
   - First in `_derive_labels_from_address_labels()` (line 134) → appears early in column order
   - Second in `_add_label_features()` (line 354-355) → overwrites the value but keeps position

2. **During inference**: `label_confidence` is added ONCE:
   - Only in `_add_label_features()` (line 354-355) → appears late in column order (after all other features)

3. **XGBoost expects exact column order**: When model is saved, it stores feature names in order. During inference, if column order differs, it throws a `feature_names mismatch` error.

### Feature Position Comparison

| Feature | Training Position | Inference Position | Issue |
|---------|-------------------|-------------------|-------|
| window_days | 1 | 1 | ✓ |
| alert_confidence_score | 2 | 2 | ✓ |
| volume_usd | 3 | 3 | ✓ |
| **label_confidence** | **4** | **56** | ❌ **MISMATCH** |
| severity_encoded | 5 | 4 | ❌ Shifted |
| volume_usd_log | 6 | 5 | ❌ Shifted |
| ... | ... | ... | ❌ All shifted |

---

## Solution Design

### Option 1: Remove Duplicate Assignment (RECOMMENDED)

**Goal**: Ensure `label_confidence` is added only once, in the same place, for both training and inference.

**Change**: Remove the duplicate assignment in `_derive_labels_from_address_labels()`.

**Rationale**:
- `_add_label_features()` is called consistently in both training and inference
- `_derive_labels_from_address_labels()` should only handle label derivation, not feature engineering
- Keeps feature engineering logic in one place

**Implementation**:

```python
# packages/training/feature_builder.py

def _derive_labels_from_address_labels(self, alerts_df, labels_df):
    logger.info("Deriving labels from address_labels (SOT baseline)")
    
    if labels_df.empty:
        raise ValueError("address_labels table is empty")
    
    label_map = {}
    # ❌ REMOVE: confidence_map = {}
    
    for _, row in labels_df.iterrows():
        addr = row['address']
        risk = row['risk_level'].lower()
        # ❌ REMOVE: confidence = row.get('confidence_score', 1.0)
        
        if risk in ['high', 'critical']:
            label_map[addr] = 1
            # ❌ REMOVE: confidence_map[addr] = confidence
        elif risk in ['low', 'medium']:
            label_map[addr] = 0
            # ❌ REMOVE: confidence_map[addr] = confidence
    
    alerts_df['label'] = alerts_df['address'].map(label_map)
    # ❌ REMOVE: alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)
    alerts_df['label_source'] = alerts_df['address'].map(
        lambda x: 'address_labels' if x in label_map else None
    )
    
    # ... logging ...
    
    return alerts_df
```

**Result**:
- `label_confidence` is added ONLY in `_add_label_features()` for both training and inference
- Feature order is consistent
- XGBoost model can load successfully

### Option 2: Explicit Feature Ordering

**Goal**: Force consistent column order using a predefined feature list.

**Change**: Add a final reordering step in `_finalize_features()`.

**Rationale**:
- Guarantees order regardless of how features are added
- Makes feature order explicit and testable
- More robust to future changes

**Implementation**:

```python
# packages/training/feature_builder.py

class FeatureBuilder:
    
    # Define canonical feature order
    FEATURE_ORDER = [
        'window_days',
        'alert_confidence_score',
        'volume_usd',
        'severity_encoded',
        'volume_usd_log',
        'confidence_score',
        'address_type_encoded',
        # ... all other features in desired order ...
        'label_risk_level',
        'has_label',
        'label_confidence'  # Always last
    ]
    
    def _finalize_features(self, df):
        logger.info("Finalizing feature matrix")
        
        # ... existing drop_cols logic ...
        
        # Ensure consistent column order
        available_features = [f for f in self.FEATURE_ORDER if f in X.columns]
        X = X[available_features]
        
        # ... rest of finalization ...
        
        return X
```

**Result**:
- Features always appear in same order
- Easier to debug and test
- Slightly more maintenance (must update FEATURE_ORDER when adding features)

---

## Recommended Solution

**Use Option 1** (Remove Duplicate Assignment) because:

1. ✅ **Simplest fix** - Just remove duplicate lines
2. ✅ **Root cause fix** - Eliminates the actual problem (duplicate assignment)
3. ✅ **Minimal changes** - Only touches one method
4. ✅ **No maintenance burden** - No need to maintain FEATURE_ORDER list
5. ✅ **Clear semantics** - Label derivation separate from feature engineering

Option 2 is better for:
- ❌ Complex feature pipelines with many contributors
- ❌ When you want explicit control over feature order
- ❌ When you have dynamic features that may vary

Since we have a simple, controlled pipeline, Option 1 is preferred.

---

## Impact Analysis

### Files to Change

1. **`packages/training/feature_builder.py`**
   - Method: `_derive_labels_from_address_labels()` (lines 107-148)
   - Remove: Lines 119, 128, 131, 134

### Breaking Changes

**None** - This is a bug fix, not a breaking change.

- Models trained before this fix will be incompatible with inference after the fix
- Solution: Retrain models after applying fix
- This is expected and acceptable

### Testing Requirements

1. **Unit Test**: Verify feature order consistency
   ```python
   def test_feature_order_consistency():
       builder = FeatureBuilder()
       
       # Build training features
       X_train, y_train = builder.build_training_features(training_data)
       train_features = list(X_train.columns)
       
       # Build inference features
       X_infer = builder.build_inference_features(training_data)
       infer_features = list(X_infer.columns)
       
       # Should be identical
       assert train_features == infer_features
   ```

2. **Integration Test**: Verify model can load and predict
   ```python
   def test_model_inference_after_training():
       # Train model
       trainer = ModelTrainer(...)
       model, metrics = trainer.train(X, y)
       
       # Save model
       storage = ModelStorage(...)
       storage.save_model(model, ...)
       
       # Load model
       loaded_model = storage.load_model(...)
       
       # Should successfully predict
       predictions = loaded_model.predict_proba(X_new)
       assert predictions is not None
   ```

---

## Implementation Plan

### Phase 1: Fix Core Issue (CRITICAL)

**Priority**: P0 (Blocks production)

**Steps**:

1. Update [`feature_builder.py:_derive_labels_from_address_labels()`](../../packages/training/feature_builder.py#L107-L148)
   - Remove lines 119, 128, 131, 134 (confidence_map and label_confidence assignment)
   - Keep only label derivation logic

2. Verify change doesn't affect label strategy
   - Check [`address_label_strategy.py`](../../packages/training/strategies/address_label_strategy.py)
   - Ensure it still gets confidence from `_add_label_features()` output

### Phase 2: Retrain Models (REQUIRED)

**Priority**: P0 (Must follow Phase 1)

**Steps**:

1. Delete old trained models
   ```bash
   rm -f data/trained_models/*/alert_scorer_*.txt
   rm -f data/trained_models/*/alert_scorer_*.json
   ```

2. Retrain all models
   ```bash
   python scripts/train_model.py --network torus --model-type alert_scorer
   python scripts/train_model.py --network torus --model-type alert_ranker
   python scripts/train_model.py --network torus --model-type cluster_scorer
   ```

3. Verify new models work with scoring
   ```bash
   python scripts/score_batch.py --network torus --processing-date 2025-08-01
   ```

### Phase 3: Add Safeguards (RECOMMENDED)

**Priority**: P1 (Nice to have)

**Steps**:

1. Add feature order validation test
2. Add CI check to ensure training/inference consistency
3. Document feature engineering guidelines

---

## Alternative Approaches Considered

### 1. Always Use label_confidence from Training

**Idea**: Store label_confidence during training and pass it during inference.

**Rejected because**:
- ❌ Breaks the stateless inference model
- ❌ Requires storing additional metadata
- ❌ Doesn't address root cause

### 2. Use Feature Selection

**Idea**: Use only features that exist in model metadata.

**Rejected because**:
- ❌ Doesn't fix ordering issue
- ❌ Just masks the problem
- ❌ Makes debugging harder

### 3. Reorder Columns at Prediction Time

**Idea**: Reorder dataframe columns to match model feature_names before prediction.

**Rejected because**:
- ❌ Performance overhead
- ❌ Fragile (easy to break)
- ❌ Doesn't fix root cause

---

## Validation Criteria

The fix is successful when:

1. ✅ Training and inference produce identical feature order
2. ✅ `label_confidence` appears at same position in both
3. ✅ Models can load and predict without errors
4. ✅ Full pipeline runs successfully: ingest → train → score
5. ✅ No regression in model performance

---

## References

- Error log: [example_full_pipeline.py output](#task)
- Code: [`packages/training/feature_builder.py`](../../packages/training/feature_builder.py)
- Related: [`packages/training/strategies/address_label_strategy.py`](../../packages/training/strategies/address_label_strategy.py)