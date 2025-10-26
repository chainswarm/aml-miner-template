# Customization Guide

Learn how to customize and extend the AML Miner Template for your specific needs.

## Table of Contents

- [Overview](#overview)
- [Adding Custom Features](#adding-custom-features)
- [Modifying Model Architecture](#modifying-model-architecture)
- [Changing Hyperparameters](#changing-hyperparameters)
- [Extending API Endpoints](#extending-api-endpoints)
- [Advanced Techniques](#advanced-techniques)
- [Feature Engineering Tips](#feature-engineering-tips)

## Overview

The AML Miner Template is designed to be highly customizable. You can:

- Add custom features for better detection
- Modify model architecture and algorithms
- Create new API endpoints
- Implement advanced ML techniques
- Integrate external data sources

## Adding Custom Features

### Understanding Feature Pipeline

Features are built in [`aml_miner/features/feature_builder.py`](../aml_miner/features/feature_builder.py):

```python
class FeatureBuilder:
    def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
        # Feature engineering happens here
        pass
```

### Example: Adding Transaction Velocity Feature

**Before:**

```python
def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
    features = []
    for alert in alerts:
        alert_features = [
            np.log1p(alert['amount_usd']),
            alert['transaction_count'] / 100.0
        ]
        features.append(alert_features)
    return np.array(features)
```

**After (with velocity feature):**

```python
def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
    features = []
    for alert in alerts:
        # Calculate velocity (transactions per day)
        velocity = alert['transaction_count'] / max(alert.get('time_window_days', 1), 1)
        
        alert_features = [
            np.log1p(alert['amount_usd']),
            alert['transaction_count'] / 100.0,
            velocity  # New feature
        ]
        features.append(alert_features)
    return np.array(features)
```

### Example: Adding Network Risk Scores

```python
# Define network risk scores
NETWORK_RISK = {
    'bitcoin': 0.5,
    'ethereum': 0.4,
    'monero': 0.9,
    'litecoin': 0.3
}

def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
    features = []
    for alert in alerts:
        network_risk = NETWORK_RISK.get(alert['network'], 0.5)
        
        alert_features = [
            np.log1p(alert['amount_usd']),
            alert['transaction_count'] / 100.0,
            network_risk  # Network-specific risk
        ]
        features.append(alert_features)
    return np.array(features)
```

### Example: Adding Time-Based Features

```python
from datetime import datetime

def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
    features = []
    for alert in alerts:
        timestamp = datetime.fromisoformat(alert['timestamp'])
        
        # Time-based features
        hour = timestamp.hour / 24.0  # Normalize to [0, 1]
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        is_night = 1.0 if hour < 0.25 or hour > 0.75 else 0.0
        
        alert_features = [
            np.log1p(alert['amount_usd']),
            alert['transaction_count'] / 100.0,
            hour,
            is_weekend,
            is_night
        ]
        features.append(alert_features)
    return np.array(features)
```

### Example: Adding Aggregated Features

```python
def build_alert_features(self, alerts: List[Dict]) -> np.ndarray:
    # Calculate global statistics for normalization
    all_amounts = [a['amount_usd'] for a in alerts]
    mean_amount = np.mean(all_amounts)
    std_amount = np.std(all_amounts)
    
    features = []
    for alert in alerts:
        # Z-score normalization
        amount_zscore = (alert['amount_usd'] - mean_amount) / (std_amount + 1e-8)
        
        # Percentile-based feature
        amount_percentile = sum(1 for a in all_amounts if a <= alert['amount_usd']) / len(all_amounts)
        
        alert_features = [
            np.log1p(alert['amount_usd']),
            amount_zscore,
            amount_percentile
        ]
        features.append(alert_features)
    return np.array(features)
```

### Testing New Features

After adding features, test them:

```bash
python scripts/test_features.py
```

Verify feature importance:

```python
from aml_miner.models import AlertScorer
import matplotlib.pyplot as plt

# Load trained model
model = AlertScorer.load('trained_models/alert_scorer.json')

# Get feature importance
importance = model.model.get_score(importance_type='weight')
print("Feature Importance:", importance)

# Plot
plt.barh(range(len(importance)), list(importance.values()))
plt.yticks(range(len(importance)), list(importance.keys()))
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## Modifying Model Architecture

### Changing Model Type

**Default: XGBoost**

```python
from xgboost import XGBClassifier

self.model = XGBClassifier(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05
)
```

**Alternative: LightGBM**

```python
from lightgbm import LGBMClassifier

self.model = LGBMClassifier(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31
)
```

**Alternative: CatBoost**

```python
from catboost import CatBoostClassifier

self.model = CatBoostClassifier(
    depth=6,
    iterations=200,
    learning_rate=0.05,
    verbose=False
)
```

### Adding Neural Network Model

Create new model in [`aml_miner/models/neural_scorer.py`](../aml_miner/models/neural_scorer.py):

```python
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np
from .base_model import BaseModel

class NeuralScorer(BaseModel):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy().flatten()
```

### Ensemble Models

Combine multiple models:

```python
from typing import List
import numpy as np
from .base_model import BaseModel

class EnsembleScorer(BaseModel):
    def __init__(self, models: List[BaseModel], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
```

Usage:

```python
from aml_miner.models import AlertScorer, NeuralScorer, EnsembleScorer

# Load individual models
xgb_scorer = AlertScorer.load('trained_models/alert_scorer_xgb.json')
neural_scorer = NeuralScorer.load('trained_models/alert_scorer_neural.pt')

# Create ensemble
ensemble = EnsembleScorer(
    models=[xgb_scorer, neural_scorer],
    weights=[0.7, 0.3]  # 70% XGBoost, 30% Neural
)

# Use ensemble for predictions
scores = ensemble.predict(X_test)
```

## Changing Hyperparameters

### Configuration File Method

Edit [`aml_miner/config/model_config.yaml`](../aml_miner/config/model_config.yaml):

**Before:**

```yaml
models:
  alert_scorer:
    max_depth: 6
    n_estimators: 200
    learning_rate: 0.05
```

**After:**

```yaml
models:
  alert_scorer:
    max_depth: 8
    n_estimators: 300
    learning_rate: 0.03
    min_child_weight: 3
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 1.0
```

### Programmatic Method

```python
from aml_miner.models import AlertScorer

# Create model with custom hyperparameters
scorer = AlertScorer(
    max_depth=8,
    n_estimators=300,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train
scorer.fit(X_train, y_train)
```

### Domain-Specific Hyperparameters

For imbalanced datasets:

```python
scorer = AlertScorer(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    scale_pos_weight=3.0,  # Handle imbalance (ratio of negatives to positives)
    eval_metric='aucpr'     # Use PR-AUC instead of ROC-AUC
)
```

For speed optimization:

```python
scorer = AlertScorer(
    max_depth=4,           # Shallower trees
    n_estimators=100,      # Fewer trees
    learning_rate=0.1,     # Faster learning
    subsample=0.7,         # Sample 70% of data
    tree_method='hist'     # Histogram-based method
)
```

For better generalization:

```python
scorer = AlertScorer(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.03,
    min_child_weight=5,    # More conservative splits
    gamma=0.1,             # Minimum loss reduction
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0         # L2 regularization
)
```

## Extending API Endpoints

### Adding New Endpoint

Edit [`aml_miner/api/routes.py`](../aml_miner/api/routes.py):

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

# New schema
class BatchScoreRequest(BaseModel):
    alert_ids: List[str]
    batch_size: int = 100

class BatchScoreResponse(BaseModel):
    results: List[Dict]
    total_processed: int

# New endpoint
@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest):
    results = []
    
    # Process in batches
    for i in range(0, len(request.alert_ids), request.batch_size):
        batch = request.alert_ids[i:i + request.batch_size]
        # Process batch
        batch_scores = process_alert_batch(batch)
        results.extend(batch_scores)
    
    return BatchScoreResponse(
        results=results,
        total_processed=len(results)
    )
```

### Adding Authentication

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@router.post("/score/alerts")
async def score_alerts(
    request: AlertScoreRequest,
    token: str = Depends(verify_token)
):
    # Protected endpoint
    return score_alerts_internal(request)
```

### Adding Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/score/alerts")
@limiter.limit("100/minute")
async def score_alerts(request: AlertScoreRequest):
    # Limited to 100 requests per minute
    return score_alerts_internal(request)
```

### Adding Caching

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def get_cached_score(alert_hash: str):
    # Cache frequently requested scores
    pass

@router.post("/score/alerts")
async def score_alerts(request: AlertScoreRequest):
    results = []
    for alert in request.alerts:
        # Create hash of alert
        alert_hash = hashlib.md5(
            json.dumps(alert, sort_keys=True).encode()
        ).hexdigest()
        
        # Check cache
        score = get_cached_score(alert_hash)
        if score is None:
            score = model.predict([alert])
            # Cache for future requests
        
        results.append(score)
    
    return {"scores": results}
```

## Advanced Techniques

### Feature Selection

Use feature selection to identify most important features:

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Select top K features
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
print(f"Selected features: {selected_indices}")

# Train model on selected features
model.fit(X_selected, y_train)
```

Integrate into [`aml_miner/features/feature_selector.py`](../aml_miner/features/feature_selector.py):

```python
class FeatureSelector:
    def __init__(self, method: str = 'mutual_info', k: int = 10):
        self.method = method
        self.k = k
        self.selector = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        self.selector.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)
```

### Automated Feature Engineering

Use automated feature engineering libraries:

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='alerts')
es = es.add_dataframe(
    dataframe_name='alerts',
    dataframe=alerts_df,
    index='alert_id'
)

# Generate features automatically
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='alerts',
    max_depth=2
)
```

### Online Learning

Implement incremental learning for continuous model updates:

```python
from sklearn.linear_model import SGDClassifier

class OnlineScorer(BaseModel):
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss')
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # Update model with new data
        self.model.partial_fit(X, y, classes=[0, 1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
```

### Model Interpretability

Add SHAP explanations:

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model.model)

# Get SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
```

Add to API endpoint:

```python
@router.post("/explain/alert")
async def explain_alert(request: AlertScoreRequest):
    # Get features
    features = feature_builder.build_alert_features(request.alerts)
    
    # Get prediction
    score = model.predict(features)
    
    # Get explanation
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(features)
    
    return {
        "score": float(score[0]),
        "feature_contributions": shap_values[0].tolist()
    }
```

## Feature Engineering Tips

### Domain-Specific Features

**Transaction Patterns:**

```python
# Rapid succession transactions
def calculate_burst_score(transactions: List[Dict]) -> float:
    if len(transactions) < 2:
        return 0.0
    
    timestamps = sorted([t['timestamp'] for t in transactions])
    intervals = [
        (timestamps[i+1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    
    # High score if many transactions in short time
    avg_interval = sum(intervals) / len(intervals)
    return 1.0 / (1.0 + avg_interval / 60.0)  # Score based on avg minutes
```

**Network-Based Features:**

```python
# Mixing service indicator
MIXING_PATTERNS = ['tornado_cash', 'wasabi_wallet', 'samourai']

def detect_mixing(address: str, patterns: List[str]) -> float:
    for pattern in patterns:
        if pattern in address.lower():
            return 1.0
    return 0.0
```

**Behavioral Features:**

```python
# Unusual time patterns
def is_unusual_time(timestamp: datetime) -> float:
    hour = timestamp.hour
    # Business hours: 9-17
    if 9 <= hour <= 17:
        return 0.0
    # Night hours: 22-6
    elif hour >= 22 or hour <= 6:
        return 1.0
    else:
        return 0.5
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (range [0,1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust scaling (using median and IQR)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Handling Missing Values

```python
# Imputation strategies
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Median imputation (better for outliers)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Forward fill (for time series)
df.fillna(method='ffill', inplace=True)
```

### Creating Interaction Features

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Manual interactions
amount = alert['amount_usd']
count = alert['transaction_count']

# Meaningful interactions
avg_transaction = amount / (count + 1)  # Average per transaction
volume_velocity = amount * count         # Combined volume and frequency
```

### Temporal Features

```python
from datetime import datetime

def extract_temporal_features(timestamp: datetime) -> Dict:
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'is_weekend': timestamp.weekday() >= 5,
        'is_month_end': timestamp.day >= 25,
        'quarter': (timestamp.month - 1) // 3 + 1
    }
```

## Testing Customizations

After making changes, verify everything works:

```bash
# Test features
python scripts/test_features.py

# Test training
python scripts/verify_training.py

# Test API
python scripts/verify_api.py
```

## Best Practices

1. **Version control** - Track all customizations in git
2. **Test incrementally** - Test each change before moving on
3. **Document changes** - Keep notes on why you made each change
4. **Benchmark performance** - Compare new features against baseline
5. **Monitor in production** - Track metrics after deployment

## Next Steps

- Experiment with different features
- Try ensemble methods
- Optimize for your specific use case
- Deploy and monitor in production

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

Happy customizing! 🛠️