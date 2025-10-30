# Training Labels Strategy - Final Architecture

## Key Insight: We Already Have Labels!

The `raw_address_labels` table contains SOT's baseline labeled dataset:
- Exchange addresses
- Mixer addresses  
- Scam addresses
- Trusted parties
- Other labeled entities with risk levels

**This is the ground truth baseline that all miners start with.**

---

## Final Architecture: Strategy Pattern + SOT Baseline + Miner Extensions

### Core Principles

1. **SOT Baseline Dataset**: Use `raw_address_labels` as ground truth
   - Join alerts with address_labels to derive labels
   - Risk levels become training labels
   - High/Critical risk = positive class, Low/Medium = negative class

2. **Miner Flexibility (Option 5)**: Abstract strategy pattern
   - Miners can use ANY ML algorithm (not just XGBoost)
   - Miners can add their own custom datasets
   - Miners can override/extend labeling logic
   - Template provides basic implementation as example

3. **Extensibility**: Miner customization points
   - Custom feature engineering
   - Custom label derivation
   - Custom model architectures
   - Custom datasets (added to address_labels)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SOT Data (Baseline)                      │
├─────────────────────────────────────────────────────────────┤
│ • raw_alerts                                                 │
│ • raw_features                                               │
│ • raw_clusters                                               │
│ • raw_money_flows                                            │
│ • raw_address_labels ← GROUND TRUTH LABELS                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Template (Abstract Implementation)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │   LabelStrategy (Abstract Base Class)       │             │
│  ├────────────────────────────────────────────┤             │
│  │ • derive_labels()                           │             │
│  │ • validate_labels()                         │             │
│  │ • get_label_weights()                       │             │
│  └────────────────────────────────────────────┘             │
│                      ↑                                       │
│                      │ implements                            │
│  ┌────────────────────────────────────────────┐             │
│  │  AddressLabelStrategy (Default)             │             │
│  ├────────────────────────────────────────────┤             │
│  │ • Join alerts with address_labels           │             │
│  │ • Map risk_level to binary labels           │             │
│  │ • Use confidence_score as weights           │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │   ModelTrainer (Abstract Base Class)        │             │
│  ├────────────────────────────────────────────┤             │
│  │ • train()                                   │             │
│  │ • predict()                                 │             │
│  │ • evaluate()                                │             │
│  └────────────────────────────────────────────┘             │
│                      ↑                                       │
│                      │ implements                            │
│  ┌────────────────────────────────────────────┐             │
│  │  XGBoostTrainer (Default Example)           │             │
│  ├────────────────────────────────────────────┤             │
│  │ • Basic XGBoost implementation              │             │
│  │ • Standard hyperparameters                  │             │
│  │ • Cross-validation                          │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Miner Customization (Extensions)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option 1: Extend Label Strategy                            │
│  ┌────────────────────────────────────────────┐             │
│  │  MinerLabelStrategy (Custom)                │             │
│  ├────────────────────────────────────────────┤             │
│  │ • Add custom labeled datasets               │             │
│  │ • Custom label derivation logic             │             │
│  │ • Combine SOT + proprietary labels          │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  Option 2: Custom Model                                     │
│  ┌────────────────────────────────────────────┐             │
│  │  NeuralNetTrainer (Custom)                  │             │
│  ├────────────────────────────────────────────┤             │
│  │ • Custom neural network architecture        │             │
│  │ • Different training approach               │             │
│  │ • Custom evaluation metrics                 │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  Option 3: Both                                             │
│  • Custom labels + Custom models                            │
│  • Full flexibility to innovate                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Fix Immediate Issue (Use Address Labels)

**Goal:** Get training working by using existing `raw_address_labels` as ground truth.

#### 1.1 Update FeatureBuilder
```python
# packages/training/feature_builder.py

class FeatureBuilder:
    def build_training_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        
        logger.info("Building training features")
        
        # Derive labels from address_labels table
        alerts_with_labels = self._add_address_labels_as_ground_truth(
            data['alerts'],
            data['address_labels']
        )
        
        # Filter to only labeled alerts
        labeled_alerts = alerts_with_labels[
            alerts_with_labels['label'].notna()
        ].copy()
        
        if len(labeled_alerts) == 0:
            raise ValueError(
                "No labeled alerts found. "
                "address_labels table must contain labels for alert addresses"
            )
        
        y = labeled_alerts['label']
        
        # Build features as before
        X = self._add_alert_features(labeled_alerts)
        X = self._add_address_features(X, data['features'])
        # ... rest of feature engineering
        
        return X, y
    
    def _add_address_labels_as_ground_truth(
        self,
        alerts_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join alerts with address_labels to derive ground truth labels.
        
        Label mapping:
        - High/Critical risk = 1 (positive - suspicious)
        - Low/Medium risk = 0 (negative - normal)
        - No label = NaN (filtered out)
        """
        logger.info("Deriving labels from address_labels table")
        
        # Create label mapping from address_labels
        label_map = {}
        for _, row in labels_df.iterrows():
            addr = row['address']
            risk = row['risk_level'].lower()
            
            if risk in ['high', 'critical']:
                label_map[addr] = 1  # Suspicious
            elif risk in ['low', 'medium']:
                label_map[addr] = 0  # Normal
            # else: uncertain, leave as NaN
        
        # Apply labels to alerts
        alerts_df['label'] = alerts_df['address'].map(label_map)
        
        # Track label source for transparency
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

#### 1.2 Update Feature Column References
Fix column name mismatches from earlier (these now reference address_labels columns):
```python
def _add_address_features(self, alerts_df, features_df):
    merged = alerts_df.merge(
        features_df,
        on=['address', 'processing_date', 'window_days'],
        how='left',
        suffixes=('', '_feat')
    )
    
    # Use correct column names from raw_features schema
    merged['total_volume'] = (
        merged['total_in_usd'].fillna(0) +
        merged['total_out_usd'].fillna(0)
    )
    
    merged['is_exchange_flag'] = merged['is_exchange_like'].fillna(False).astype(int)
    merged['is_mixer_flag'] = merged['is_mixer_like'].fillna(False).astype(int)
    
    # ... etc
```

---

### Phase 2: Add Strategy Pattern (Extensibility)

**Goal:** Enable miners to customize labeling and training strategies.

#### 2.1 Create Abstract Base Classes

```python
# packages/training/strategies/__init__.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd

class LabelStrategy(ABC):
    """Abstract base class for label derivation strategies."""
    
    @abstractmethod
    def derive_labels(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Derive labels for alerts.
        
        Args:
            alerts_df: Alerts DataFrame
            data: Dict with all extracted data (features, labels, etc.)
            
        Returns:
            alerts_df with 'label' column added
        """
        pass
    
    @abstractmethod
    def validate_labels(self, alerts_df: pd.DataFrame) -> bool:
        """Validate that labels are properly derived."""
        pass
    
    def get_label_weights(self, alerts_df: pd.DataFrame) -> pd.Series:
        """Optional: Return sample weights for training."""
        return pd.Series(1.0, index=alerts_df.index)


class ModelTrainer(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None
    ) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save trained model."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load trained model."""
        pass
```

#### 2.2 Implement Default Strategies

```python
# packages/training/strategies/address_label_strategy.py

from .base import LabelStrategy
from loguru import logger

class AddressLabelStrategy(LabelStrategy):
    """
    Default label strategy using SOT's address_labels table.
    
    This is the baseline implementation that all miners start with.
    """
    
    def __init__(
        self,
        positive_risk_levels: list = ['high', 'critical'],
        negative_risk_levels: list = ['low', 'medium'],
        use_confidence_weights: bool = True
    ):
        self.positive_risk_levels = positive_risk_levels
        self.negative_risk_levels = negative_risk_levels
        self.use_confidence_weights = use_confidence_weights
    
    def derive_labels(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Join alerts with address_labels to derive labels."""
        
        logger.info("Deriving labels from address_labels (SOT baseline)")
        
        labels_df = data.get('address_labels')
        if labels_df is None or labels_df.empty:
            raise ValueError("No address_labels data available")
        
        # Create label mapping
        label_map = {}
        confidence_map = {}
        
        for _, row in labels_df.iterrows():
            addr = row['address']
            risk = row['risk_level'].lower()
            confidence = row.get('confidence_score', 1.0)
            
            if risk in self.positive_risk_levels:
                label_map[addr] = 1
                confidence_map[addr] = confidence
            elif risk in self.negative_risk_levels:
                label_map[addr] = 0
                confidence_map[addr] = confidence
        
        # Apply to alerts
        alerts_df['label'] = alerts_df['address'].map(label_map)
        alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)
        alerts_df['label_source'] = 'sot_address_labels'
        
        return alerts_df
    
    def validate_labels(self, alerts_df: pd.DataFrame) -> bool:
        """Validate labels exist and are binary."""
        if 'label' not in alerts_df.columns:
            return False
        
        labeled = alerts_df['label'].notna().sum()
        if labeled == 0:
            return False
        
        # Check labels are binary
        unique_labels = alerts_df['label'].dropna().unique()
        if not set(unique_labels).issubset({0, 1}):
            return False
        
        return True
    
    def get_label_weights(self, alerts_df: pd.DataFrame) -> pd.Series:
        """Use confidence scores as sample weights."""
        if self.use_confidence_weights and 'label_confidence' in alerts_df.columns:
            return alerts_df['label_confidence'].fillna(1.0)
        return pd.Series(1.0, index=alerts_df.index)
```

```python
# packages/training/strategies/xgboost_trainer.py

from .base import ModelTrainer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from loguru import logger
import joblib

class XGBoostTrainer(ModelTrainer):
    """
    Default XGBoost trainer implementation.
    
    This is a basic example that miners can extend or replace.
    """
    
    def __init__(self, hyperparameters: dict = None):
        self.hyperparameters = hyperparameters or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
        self.model = None
    
    def train(self, X, y, sample_weights=None):
        """Train XGBoost model."""
        logger.info(f"Training XGBoost with {len(X)} samples")
        
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        
        if sample_weights is not None:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)
        
        logger.success("XGBoost training completed")
        return self.model
    
    def predict(self, X):
        """Generate probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        auc_score = roc_auc_score(y, y_pred)
        
        precision, recall, _ = precision_recall_curve(y, y_pred)
        pr_auc = auc(recall, precision)
        
        return {
            'auc': float(auc_score),
            'pr_auc': float(pr_auc)
        }
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
```

#### 2.3 Update Training Pipeline

```python
# packages/training/model_training.py

from .strategies import AddressLabelStrategy, XGBoostTrainer

class ModelTraining:
    def __init__(
        self,
        network: str,
        start_date: str,
        end_date: str,
        model_type: str,
        window_days: int,
        label_strategy: LabelStrategy = None,
        model_trainer: ModelTrainer = None
    ):
        # ... existing init ...
        
        # Use default strategies if not provided
        self.label_strategy = label_strategy or AddressLabelStrategy()
        self.model_trainer = model_trainer or XGBoostTrainer()
    
    def run(self):
        # Extract data
        data = extractor.extract_training_data(...)
        
        # Derive labels using strategy
        alerts_with_labels = self.label_strategy.derive_labels(
            data['alerts'],
            data
        )
        
        # Validate
        if not self.label_strategy.validate_labels(alerts_with_labels):
            raise ValueError("Label validation failed")
        
        # Update data
        data['alerts'] = alerts_with_labels
        
        # Build features
        X, y = builder.build_training_features(data)
        
        # Get sample weights
        weights = self.label_strategy.get_label_weights(
            data['alerts'][data['alerts']['label'].notna()]
        )
        
        # Train model
        model = self.model_trainer.train(X, y, weights)
        
        # Evaluate
        metrics = self.model_trainer.evaluate(X, y)
        
        # Save
        self.model_trainer.save(model_path)
```

---

### Phase 3: Enable Miner Extensions

#### 3.1 Document Extension Points

```markdown
# Miner Customization Guide

## Customizing Labels

### Option 1: Add Custom Dataset to address_labels

```python
# Insert your custom labeled addresses into the database
custom_labels = pd.DataFrame({
    'processing_date': ['2025-08-01'] * 100,
    'window_days': [195] * 100,
    'network': ['torus'] * 100,
    'address': [...],  # Your addresses
    'label': [...],  # Your labels
    'risk_level': ['high', 'low', ...],
    'confidence_score': [0.9, 0.8, ...],
    'source': 'miner_custom_dataset'
})

# Ingest into ClickHouse
client.insert_df('raw_address_labels', custom_labels)
```

### Option 2: Create Custom Label Strategy

```python
from packages.training.strategies import LabelStrategy

class MyCustomLabelStrategy(LabelStrategy):
    def derive_labels(self, alerts_df, data):
        # Start with SOT baseline
        baseline_strategy = AddressLabelStrategy()
        alerts_df = baseline_strategy.derive_labels(alerts_df, data)
        
        # Add your custom logic
        # e.g., use proprietary intelligence
        custom_labels = self._load_proprietary_labels()
        alerts_df = alerts_df.merge(custom_labels, on='address', how='left')
        
        # Combine labels (your logic)
        alerts_df['label'] = alerts_df['label'].fillna(
            alerts_df['custom_label']
        )
        
        return alerts_df
```

## Customizing Models

```python
from packages.training.strategies import ModelTrainer
import torch
import torch.nn as nn

class NeuralNetTrainer(ModelTrainer):
    def __init__(self):
        self.model = self._build_network()
    
    def _build_network(self):
        # Your custom neural network
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def train(self, X, y, sample_weights=None):
        # Your training loop
        optimizer = torch.optim.Adam(self.model.parameters())
        # ... training code ...
    
    def predict(self, X):
        # Your inference code
        with torch.no_grad():
            return self.model(torch.tensor(X.values)).numpy()
```
```

---

## Benefits of This Architecture

### For Template Maintainers
- ✅ Clean, extensible codebase
- ✅ Easy to add new default strategies
- ✅ Clear separation of concerns
- ✅ Testable components

### For Miners
- ✅ Start with working baseline (SOT address_labels)
- ✅ Full flexibility to customize any component
- ✅ Can add proprietary datasets
- ✅ Can use any ML framework/algorithm
- ✅ Clear extension points

### For Competition
- ✅ Fair baseline (everyone gets SOT labels)
- ✅ Competitive advantage through innovation
- ✅ Better labels OR better models = better scores
- ✅ Transparent evaluation

---

## Summary

**Starting Point (All Miners):**
- SOT baseline: `raw_address_labels` provides ground truth
- Template provides working example with AddressLabelStrategy + XGBoostTrainer
- Miners can run training immediately

**Customization (Miner Choice):**
- Add custom datasets to `raw_address_labels`
- Implement custom `LabelStrategy`
- Implement custom `ModelTrainer`  
- Or both!

**Result:**
- Template is production-ready with SOT baseline
- Miners have maximum flexibility to innovate
- Fair competition based on skill, not just data access