# Miner Customization Guide

This guide explains how miners can customize the training pipeline to gain competitive advantage.

## Overview

The template provides a flexible architecture with two main extension points:

1. **Label Strategies** - How labels are derived from data
2. **Model Trainers** - Which ML algorithms/models to use

All miners start with the same **SOT baseline** (address_labels table), but can customize both strategies to improve performance.

---

## Quick Start

### Default Implementation (SOT Baseline)

```python
from packages.training.strategies import AddressLabelStrategy, XGBoostTrainer

# Uses SOT address_labels for ground truth
label_strategy = AddressLabelStrategy()

# Uses basic XGBoost classifier
model_trainer = XGBoostTrainer()
```

### Running Training with Defaults

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195
```

---

## Customization Option 1: Add Your Own Labeled Dataset

### Step 1: Prepare Your Dataset

```python
import pandas as pd

# Your proprietary labeled addresses
custom_labels = pd.DataFrame({
    'processing_date': ['2025-08-01'] * 100,
    'window_days': [195] * 100,
    'network': ['torus'] * 100,
    'address': ['0xabc...', '0xdef...', ...],
    'label': ['scam_address', 'mixer', 'exchange', ...],
    'risk_level': ['high', 'critical', 'low', ...],
    'confidence_score': [0.95, 0.90, 0.85, ...],
    'source': 'miner_custom_intelligence'
})
```

### Step 2: Insert into ClickHouse

```python
from packages.storage import ClientFactory, get_connection_params

connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    client.insert_df('raw_address_labels', custom_labels)
```

### Step 3: Train with Combined Dataset

The default `AddressLabelStrategy` will automatically use both SOT and your custom labels!

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195
```

---

## Customization Option 2: Create Custom Label Strategy

### Example: Combine Multiple Label Sources with Custom Logic

```python
# packages/training/strategies/custom_label_strategy.py

from typing import Dict
import pandas as pd
from loguru import logger
from packages.training.strategies import LabelStrategy, AddressLabelStrategy


class CustomLabelStrategy(LabelStrategy):
    
    def __init__(self):
        self.base_strategy = AddressLabelStrategy()
    
    def derive_labels(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        logger.info("Deriving labels with custom strategy")
        
        # Start with SOT baseline
        alerts_df = self.base_strategy.derive_labels(alerts_df, data)
        
        # Add your custom logic
        # Example 1: Use proprietary threat intelligence
        proprietary_labels = self._load_proprietary_intelligence()
        alerts_df = self._merge_proprietary_labels(alerts_df, proprietary_labels)
        
        # Example 2: Use behavioral heuristics
        alerts_df = self._apply_behavioral_heuristics(alerts_df, data)
        
        # Example 3: Ensemble multiple label sources
        alerts_df = self._ensemble_label_sources(alerts_df)
        
        return alerts_df
    
    def _load_proprietary_intelligence(self) -> pd.DataFrame:
        # Load from your private database/API
        return pd.read_csv('path/to/proprietary_labels.csv')
    
    def _merge_proprietary_labels(
        self,
        alerts_df: pd.DataFrame,
        proprietary: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge with higher priority for proprietary data
        merged = alerts_df.merge(
            proprietary[['address', 'label', 'confidence']],
            on='address',
            how='left',
            suffixes=('_sot', '_prop')
        )
        
        # Use proprietary label if available, otherwise SOT
        merged['label'] = merged['label_prop'].fillna(merged['label_sot'])
        merged['label_confidence'] = merged['confidence'].fillna(
            merged['label_confidence']
        )
        
        return merged
    
    def _apply_behavioral_heuristics(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        # Example: High-volume alerts from new addresses are suspicious
        features = data['features']
        
        # Merge with features
        merged = alerts_df.merge(
            features[['address', 'total_volume_usd', 'is_new_address']],
            on='address',
            how='left'
        )
        
        # Heuristic: New address + high volume = likely suspicious
        high_vol_threshold = merged['total_volume_usd'].quantile(0.9)
        suspicious_mask = (
            (merged['is_new_address'] == True) &
            (merged['total_volume_usd'] > high_vol_threshold) &
            (merged['label'].isna())  # Only if not already labeled
        )
        
        merged.loc[suspicious_mask, 'label'] = 1
        merged.loc[suspicious_mask, 'label_source'] = 'heuristic'
        
        return merged
    
    def _ensemble_label_sources(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        # Combine multiple label sources with voting or confidence weighting
        # Implementation depends on your strategy
        return alerts_df
    
    def validate_labels(self, alerts_df: pd.DataFrame) -> bool:
        # Use base validation
        return self.base_strategy.validate_labels(alerts_df)
    
    def get_label_weights(self, alerts_df: pd.DataFrame) -> pd.Series:
        # Use confidence scores as weights
        return alerts_df['label_confidence'].fillna(1.0)
```

### Using Custom Strategy

```python
from packages.training.model_training import ModelTraining
from packages.training.strategies.custom_label_strategy import CustomLabelStrategy

# In your custom training script
label_strategy = CustomLabelStrategy()

training = ModelTraining(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    client=client,
    label_strategy=label_strategy  # Use custom strategy
)

training.run()
```

---

## Customization Option 3: Create Custom Model Trainer

### Example: Neural Network Trainer

```python
# packages/training/strategies/neural_net_trainer.py

from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from packages.training.strategies import ModelTrainer


class NeuralNetTrainer(ModelTrainer):
    
    def __init__(
        self,
        hidden_layers: list = None,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_network(self, input_dim: int) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None
    ) -> Any:
        
        logger.info(f"Training neural network with {len(X)} samples")
        
        # Build model
        self.model = self._build_network(X.shape[1]).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(self.device)
        
        if sample_weights is not None:
            weights = torch.FloatTensor(sample_weights.values).to(self.device)
        else:
            weights = torch.ones(len(y)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor, weights)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.BCELoss(reduction='none')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y, batch_w in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                weighted_loss = (loss * batch_w.unsqueeze(1)).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += weighted_loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        logger.success("Neural network training completed")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        y_pred = self.predict(X)
        
        auc_score = roc_auc_score(y, y_pred)
        precision, recall, _ = precision_recall_curve(y, y_pred)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc': float(auc_score),
            'pr_auc': float(pr_auc)
        }
        
        logger.info(f"Neural net evaluation: AUC={metrics['auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
        return metrics
    
    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        # Note: Need to know input_dim to rebuild model
        raise NotImplementedError("Load requires model architecture info")
```

### Example: LightGBM Trainer

```python
# packages/training/strategies/lightgbm_trainer.py

from typing import Dict, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from loguru import logger
from packages.training.strategies import ModelTrainer


class LightGBMTrainer(ModelTrainer):
    
    def __init__(self, hyperparameters: dict = None):
        self.hyperparameters = hyperparameters or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None
    ) -> Any:
        
        logger.info(f"Training LightGBM with {len(X)} samples")
        
        train_data = lgb.Dataset(
            X,
            label=y,
            weight=sample_weights if sample_weights is not None else None
        )
        
        self.model = lgb.train(
            self.hyperparameters,
            train_data,
            num_boost_round=100
        )
        
        logger.success("LightGBM training completed")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        
        auc_score = roc_auc_score(y, y_pred)
        precision, recall, _ = precision_recall_curve(y, y_pred)
        pr_auc = auc(recall, precision)
        
        return {
            'auc': float(auc_score),
            'pr_auc': float(pr_auc)
        }
    
    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
    
    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
```

---

## Customization Option 4: Both Custom Labels AND Custom Models

### Complete Example

```python
# my_custom_training.py

from packages.storage import ClientFactory, get_connection_params
from packages.training.model_training import ModelTraining
from packages.training.strategies.custom_label_strategy import CustomLabelStrategy
from packages.training.strategies.neural_net_trainer import NeuralNetTrainer

# Setup
connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    # Use custom strategies
    label_strategy = CustomLabelStrategy()
    model_trainer = NeuralNetTrainer(
        hidden_layers=[256, 128, 64],
        learning_rate=0.001,
        epochs=100
    )
    
    training = ModelTraining(
        network='torus',
        start_date='2025-08-01',
        end_date='2025-08-01',
        client=client,
        model_type='alert_scorer',
        window_days=195,
        label_strategy=label_strategy,
        model_trainer=model_trainer
    )
    
    training.run()
```

---

## Best Practices

### 1. Start with SOT Baseline
Always test the baseline before customizing:
```python
# First run
label_strategy = AddressLabelStrategy()  # SOT baseline
model_trainer = XGBoostTrainer()  # Default model
```

### 2. Incremental Improvements
Change one thing at a time:
```python
# Second run - custom labels, default model
label_strategy = CustomLabelStrategy()
model_trainer = XGBoostTrainer()

# Third run - custom labels, custom model
label_strategy = CustomLabelStrategy()
model_trainer = NeuralNetTrainer()
```

### 3. Track Your Changes
Log what you're changing in training_config:
```python
training_config = {
    'label_strategy': label_strategy.__class__.__name__,
    'model_trainer': model_trainer.__class__.__name__,
    'custom_params': {...}
}
```

### 4. Validate Labels
Always validate your custom labels:
```python
def validate_labels(self, alerts_df):
    assert 'label' in alerts_df.columns
    assert alerts_df['label'].notna().sum() > 0
    assert set(alerts_df['label'].dropna().unique()).issubset({0, 1})
    return True
```

### 5. Use Sample Weights
Leverage confidence scores:
```python
def get_label_weights(self, alerts_df):
    # Higher weight for high-confidence labels
    return alerts_df['label_confidence'].fillna(1.0)
```

---

## Competition Strategy Tips

### Data Quality > Model Complexity
Better labels often beat better models:
- ✅ Focus on high-quality labeled dataset first
- ✅ Validate labels carefully
- ⚠️ Complex models without good labels won't help

### Combine Multiple Sources
Ensemble different label sources:
- SOT baseline (everyone has this)
- Your proprietary intelligence
- Behavioral heuristics
- Community datasets
- External APIs (blockchain explorers, etc.)

### Model Diversity
Different models for different patterns:
- XGBoost: Good baseline, fast
- Neural Nets: Better for complex patterns
- LightGBM: Faster than XGBoost, similar performance
- Ensemble: Combine multiple models

### Feature Engineering
The template provides basic features, but you can add:
- Custom graph features
- Time-series patterns
- Domain-specific indicators
- Cross-network signals

---

## Debugging Tips

### Check Label Distribution
```python
logger.info(f"Positive labels: {(y == 1).sum()}")
logger.info(f"Negative labels: {(y == 0).sum()}")
logger.info(f"Label ratio: {y.mean():.2%}")
```

### Inspect Features
```python
logger.info(f"Feature matrix shape: {X.shape}")
logger.info(f"Feature columns: {list(X.columns)}")
logger.info(f"Missing values: {X.isna().sum().sum()}")
```

### Test Predictions
```python
y_pred = model_trainer.predict(X)
logger.info(f"Prediction range: {y_pred.min():.3f} - {y_pred.max():.3f}")
logger.info(f"Mean prediction: {y_pred.mean():.3f}")
```

---

## Summary

**Three Levels of Customization:**

1. **Easy**: Add your labeled dataset to `raw_address_labels`
2. **Medium**: Implement custom `LabelStrategy`
3. **Advanced**: Implement both custom `LabelStrategy` and `ModelTrainer`

**Remember:**
- Everyone starts with SOT baseline (fair competition)
- Competitive advantage comes from innovation
- Better data OR better models (or both!) = better scores
- Document your customizations for reproducibility