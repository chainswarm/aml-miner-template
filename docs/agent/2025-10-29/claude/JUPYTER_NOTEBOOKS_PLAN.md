# Jupyter Notebooks for ML Training

**Date**: 2025-10-29  
**Purpose**: Interactive notebooks for model development and analysis  

---

## Notebook Suite Overview

Create a comprehensive set of Jupyter notebooks to support the ML training workflow:

```
notebooks/
├── 01_data_exploration.ipynb          # Explore ingested data
├── 02_feature_analysis.ipynb          # Analyze features
├── 03_model_training.ipynb            # Interactive training
├── 04_hyperparameter_tuning.ipynb     # Optimize parameters
├── 05_model_evaluation.ipynb          # Evaluate models
├── 06_model_comparison.ipynb          # Compare multiple models
├── 07_feature_importance.ipynb        # Feature analysis
├── 08_error_analysis.ipynb            # Analyze predictions
└── README.md                          # Notebook guide
```

---

## Notebook Descriptions

### 1. Data Exploration (`01_data_exploration.ipynb`)

**Purpose**: Understand the ingested data before training

**Contents**:
- Connect to ClickHouse
- Query raw_alerts, raw_features, raw_clusters
- Data statistics and distributions
- Time series analysis
- Missing value analysis
- Class balance analysis
- Correlation heatmaps

**Key Visualizations**:
```python
# Alert volume over time
# Severity distribution
# Network activity patterns
# Address type distributions
# Volume distributions (log scale)
```

### 2. Feature Analysis (`02_feature_analysis.ipynb`)

**Purpose**: Analyze engineered features

**Contents**:
- Build features using FeatureBuilder
- Feature distributions
- Feature correlations
- Multicollinearity detection
- Feature scaling analysis
- Outlier detection

**Key Visualizations**:
```python
# Feature correlation matrix
# Distribution plots for each feature
# Box plots for feature groups
# Scatter plots for feature pairs
# PCA visualization
```

### 3. Model Training (`03_model_training.ipynb`)

**Purpose**: Interactive model training and quick iteration

**Contents**:
- Load data using FeatureExtractor
- Build features using FeatureBuilder
- Train models with different parameters
- Quick metrics visualization
- Save/load models
- Compare metrics across runs

**Key Capabilities**:
```python
# Train with custom date ranges
# Try different model types
# Adjust hyperparameters interactively
# Visualize training progress
# Quick model comparison
```

### 4. Hyperparameter Tuning (`04_hyperparameter_tuning.ipynb`)

**Purpose**: Systematic hyperparameter optimization

**Contents**:
- Grid search implementation
- Random search implementation
- Bayesian optimization (using Optuna)
- Cross-validation results
- Parameter importance analysis
- Best parameter visualization

**Key Visualizations**:
```python
# Parameter vs metric plots
# Parallel coordinate plots
# Contour plots for 2D parameter space
# Optimization history
```

### 5. Model Evaluation (`05_model_evaluation.ipynb`)

**Purpose**: Comprehensive model evaluation

**Contents**:
- Load trained models
- ROC curves and AUC
- Precision-Recall curves
- Confusion matrices
- Classification reports
- Score distributions
- Calibration curves
- Lift charts

**Key Visualizations**:
```python
# ROC curve with multiple models
# PR curve comparison
# Confusion matrix heatmap
# Score distribution plots
# Calibration plots
```

### 6. Model Comparison (`06_model_comparison.ipynb`)

**Purpose**: Compare multiple trained models

**Contents**:
- Load models from ClickHouse tracking
- Compare metrics side-by-side
- Statistical significance tests
- Model ensemble experiments
- Best model selection

**Key Visualizations**:
```python
# Metric comparison bar charts
# Radar plots for multi-metric comparison
# Violin plots for CV scores
# Model performance timeline
```

### 7. Feature Importance (`07_feature_importance.ipynb`)

**Purpose**: Understand which features drive predictions

**Contents**:
- SHAP value analysis
- LightGBM feature importance
- Permutation importance
- Partial dependence plots
- Feature interaction detection

**Key Visualizations**:
```python
# Feature importance bar chart
# SHAP summary plots
# SHAP force plots for individual predictions
# Partial dependence plots
# SHAP interaction heatmap
```

### 8. Error Analysis (`08_error_analysis.ipynb`)

**Purpose**: Analyze model mistakes and improve

**Contents**:
- False positive analysis
- False negative analysis
- Error patterns by feature values
- Misclassified example inspection
- Threshold optimization
- Segment-specific performance

**Key Visualizations**:
```python
# Error distribution plots
# Feature distributions for FP vs FN
# Confusion at different thresholds
# Segment performance heatmap
```

---

## Implementation Plan

### Structure

```python
# Standard notebook header
import sys
sys.path.insert(0, '../')  # Add project root to path

from packages.training import FeatureExtractor, FeatureBuilder, ModelTrainer
from packages.storage import ClientFactory, get_connection_params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Configuration
NETWORK = 'ethereum'
START_DATE = '2024-01-01'
END_DATE = '2024-03-31'
WINDOW_DAYS = 7
```

### Helper Functions

Create `notebooks/notebook_utils.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def setup_plotting():
    """Configure matplotlib and seaborn styles"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

def plot_metric_comparison(metrics_dict: Dict[str, float], title: str):
    """Plot bar chart comparing metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_feature_distributions(df: pd.DataFrame, features: List[str]):
    """Plot distributions for multiple features"""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        axes[idx].hist(df[feature], bins=50, edgecolor='black')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame, figsize=(12, 10)):
    """Plot correlation heatmap"""
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_pr_curve(y_true, y_pred_proba, model_name='Model'):
    """Plot Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.tight_layout()
    return fig
```

---

## Usage Examples

### Quick Start

```python
# In any notebook
from notebook_utils import *
from packages.training import FeatureExtractor, FeatureBuilder
from packages.storage import ClientFactory, get_connection_params

# Setup
setup_plotting()
connection_params = get_connection_params('ethereum')
client_factory = ClientFactory(connection_params)

# Extract data
with client_factory.client_context() as client:
    extractor = FeatureExtractor(client)
    data = extractor.extract_training_data(
        start_date='2024-01-01',
        end_date='2024-01-31',
        window_days=7
    )

# Build features
builder = FeatureBuilder()
X, y = builder.build_training_features(data)

# Visualize
plot_correlation_matrix(X)
plot_feature_distributions(X, X.columns[:6].tolist())
```

### Model Training

```python
# Train and evaluate
from packages.training import ModelTrainer

trainer = ModelTrainer(model_type='alert_scorer')
model, metrics = trainer.train(X, y, cv_folds=5)

# Visualize results
print(f"Test AUC: {metrics['test_auc']:.4f}")
print(f"CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

# Plot predictions
y_pred_proba = model.predict(X_test)
plot_roc_curve(y_test, y_pred_proba, model_name='Alert Scorer')
plot_pr_curve(y_test, y_pred_proba, model_name='Alert Scorer')
```

---

## Benefits

### ✅ Interactive Development
- Quick experimentation
- Immediate feedback
- Visual results
- Easy iteration

### ✅ Data Understanding
- Explore patterns
- Identify issues
- Validate assumptions
- Document findings

### ✅ Model Optimization
- Test parameters quickly
- Compare approaches
- Visualize improvements
- Track experiments

### ✅ Collaboration
- Share insights
- Reproducible analysis
- Document decisions
- Communicate results

### ✅ Production Validation
- Validate before deployment
- Test on new data
- Monitor drift
- Debug issues

---

## Integration with Training System

Notebooks use the same components:

```python
# Notebooks call the same code as production
from packages.training import (
    FeatureExtractor,
    FeatureBuilder,
    ModelTrainer,
    ModelStorage
)

# Same data extraction
extractor = FeatureExtractor(client)
data = extractor.extract_training_data(...)

# Same feature building
builder = FeatureBuilder()
X, y = builder.build_training_features(data)

# Same training
trainer = ModelTrainer(model_type='alert_scorer')
model, metrics = trainer.train(X, y)
```

This ensures:
- **Consistency** - Same logic in notebooks and production
- **Reliability** - Test notebook code → use in production
- **Maintainability** - One codebase to maintain

---

## Setup

1. **Install Jupyter**
   ```bash
   pip install jupyter notebook jupyterlab
   pip install matplotlib seaborn plotly
   pip install shap  # For SHAP analysis
   pip install optuna  # For hyperparameter tuning
   ```

2. **Create Notebooks Directory**
   ```bash
   mkdir notebooks
   touch notebooks/notebook_utils.py
   touch notebooks/README.md
   ```

3. **Launch Jupyter**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. **Access**
   - Open browser to `http://localhost:8888`
   - Navigate to notebooks directory
   - Start exploring!

---

## Recommended Workflow

1. **Start with Exploration** → `01_data_exploration.ipynb`
   - Understand your data
   - Identify patterns
   - Check data quality

2. **Analyze Features** → `02_feature_analysis.ipynb`
   - Review feature distributions
   - Check correlations
   - Identify important features

3. **Train Models** → `03_model_training.ipynb`
   - Quick experiments
   - Test different approaches
   - Iterate rapidly

4. **Optimize** → `04_hyperparameter_tuning.ipynb`
   - Find best parameters
   - Systematic search
   - Cross-validate

5. **Evaluate** → `05_model_evaluation.ipynb`
   - Comprehensive metrics
   - Compare to baseline
   - Validate performance

6. **Analyze Errors** → `08_error_analysis.ipynb`
   - Understand mistakes
   - Find improvement areas
   - Iterate

7. **Deploy** → Use production training system
   - Take learnings from notebooks
   - Train final model with CLI
   - Deploy to production

---

## Next Steps

Would you like me to:

1. **Create the actual notebooks** with complete code?
2. **Create just the templates** with placeholders?
3. **Create a subset** (e.g., top 3 most useful notebooks)?
4. **Add specific analysis** (e.g., SHAP, Optuna integration)?

Let me know what would be most helpful!
