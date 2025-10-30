# Training Strategies

This package provides extensible training strategies for the risk scoring system.

## Architecture

The training pipeline uses a **Strategy Pattern** to allow miners to customize:

1. **Label Derivation** - How training labels are created from data
2. **Model Training** - Which ML algorithms/models to use

## Base Classes

### LabelStrategy

Abstract base class for deriving training labels.

```python
class LabelStrategy(ABC):
    @abstractmethod
    def derive_labels(self, alerts_df, data) -> pd.DataFrame:
        """Derive labels for alerts"""
        pass
    
    @abstractmethod
    def validate_labels(self, alerts_df) -> bool:
        """Validate labels are correctly derived"""
        pass
    
    def get_label_weights(self, alerts_df) -> pd.Series:
        """Return sample weights for training (optional)"""
        return pd.Series(1.0, index=alerts_df.index)
```

### ModelTrainer

Abstract base class for training ML models.

```python
class ModelTrainer(ABC):
    @abstractmethod
    def train(self, X, y, sample_weights=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save trained model"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load trained model"""
        pass
```

## Default Implementations

### AddressLabelStrategy

Uses SOT's `raw_address_labels` table as ground truth.

**Label Mapping:**
- `risk_level` in ['high', 'critical'] → label = 1 (suspicious)
- `risk_level` in ['low', 'medium'] → label = 0 (normal)
- Other values → unlabeled (filtered out)

**Features:**
- Uses confidence scores as sample weights
- Configurable risk level thresholds
- Logs label statistics

**Usage:**
```python
from packages.training.strategies import AddressLabelStrategy

strategy = AddressLabelStrategy(
    positive_risk_levels=['high', 'critical'],
    negative_risk_levels=['low', 'medium'],
    use_confidence_weights=True
)
```

### XGBoostTrainer

Basic XGBoost classifier implementation.

**Default Hyperparameters:**
- max_depth: 6
- learning_rate: 0.1
- n_estimators: 100
- objective: binary:logistic
- eval_metric: auc

**Features:**
- Supports sample weights
- Returns AUC and PR-AUC metrics
- Model persistence with joblib

**Usage:**
```python
from packages.training.strategies import XGBoostTrainer

trainer = XGBoostTrainer(
    hyperparameters={
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200
    }
)
```

## Creating Custom Strategies

### Custom Label Strategy Example

```python
from packages.training.strategies import LabelStrategy

class MyLabelStrategy(LabelStrategy):
    def derive_labels(self, alerts_df, data):
        # Your custom logic here
        # e.g., combine multiple sources
        return alerts_df
    
    def validate_labels(self, alerts_df):
        # Validate your labels
        return True
```

### Custom Model Trainer Example

```python
from packages.training.strategies import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def train(self, X, y, sample_weights=None):
        # Your training logic
        return model
    
    def predict(self, X):
        # Your prediction logic
        return predictions
    
    def evaluate(self, X, y):
        # Your evaluation logic
        return {'auc': 0.85}
    
    def save(self, path):
        # Save model
        pass
    
    def load(self, path):
        # Load model
        pass
```

## Using Custom Strategies

```python
from packages.training.model_training import ModelTraining
from my_strategies import MyLabelStrategy, MyModelTrainer

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

## SOT Baseline

All miners start with the same baseline:
- **Labels**: AddressLabelStrategy (uses raw_address_labels)
- **Model**: XGBoostTrainer (basic XGBoost)

This ensures:
- Fair competition
- Working baseline for all miners
- Clear performance benchmarks

## Competitive Advantage

Miners can gain competitive advantage through:

1. **Better Labels**
   - Proprietary labeled datasets
   - External data sources
   - Advanced labeling heuristics
   - Ensemble labeling strategies

2. **Better Models**
   - Neural networks
   - Ensemble methods
   - Custom architectures
   - Advanced hyperparameter tuning

3. **Both**
   - Custom labels + custom models = maximum flexibility

See [MINER_CUSTOMIZATION_GUIDE.md](../../../docs/MINER_CUSTOMIZATION_GUIDE.md) for detailed examples.