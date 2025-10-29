# ML Training Notebooks

Interactive Jupyter notebooks for model development, analysis, and experimentation.

## Overview

This directory contains a complete suite of notebooks for the ML training workflow:

1. **[`01_data_exploration.ipynb`](01_data_exploration.ipynb)** - Explore ingested data from ClickHouse
2. **[`02_feature_analysis.ipynb`](02_feature_analysis.ipynb)** - Analyze engineered features
3. **[`03_model_training.ipynb`](03_model_training.ipynb)** - Interactive model training
4. **[`04_hyperparameter_tuning.ipynb`](04_hyperparameter_tuning.ipynb)** - Optimize model parameters
5. **[`05_model_evaluation.ipynb`](05_model_evaluation.ipynb)** - Comprehensive model evaluation
6. **[`06_model_comparison.ipynb`](06_model_comparison.ipynb)** - Compare multiple models
7. **[`07_feature_importance.ipynb`](07_feature_importance.ipynb)** - Analyze feature importance
8. **[`08_error_analysis.ipynb`](08_error_analysis.ipynb)** - Analyze prediction errors

## Quick Start

### Prerequisites

```bash
pip install jupyter notebook jupyterlab
pip install matplotlib seaborn plotly
pip install shap optuna
```

### Launch Jupyter

```bash
jupyter lab
```

Then navigate to the notebooks directory and start exploring!

## Workflow

### Recommended Sequence

1. **Data Exploration** → Understand your raw data
2. **Feature Analysis** → Review engineered features
3. **Model Training** → Quick experimentation
4. **Hyperparameter Tuning** → Optimize performance
5. **Model Evaluation** → Comprehensive metrics
6. **Error Analysis** → Identify improvements
7. **Deploy** → Use production training system

## Usage

### Standard Imports

All notebooks use these standard imports:

```python
import sys
sys.path.insert(0, '../')

from packages.training import FeatureExtractor, FeatureBuilder, ModelTrainer
from packages.storage import ClientFactory, get_connection_params
from notebook_utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
```

### Configuration

Configure your analysis in each notebook:

```python
NETWORK = 'ethereum'
START_DATE = '2024-01-01'
END_DATE = '2024-03-31'
WINDOW_DAYS = 7
```

### Helper Functions

The [`notebook_utils.py`](notebook_utils.py) module provides visualization helpers:

- `setup_plotting()` - Configure matplotlib/seaborn
- `plot_metric_comparison()` - Bar charts for metrics
- `plot_feature_distributions()` - Distribution histograms
- `plot_correlation_matrix()` - Correlation heatmap
- `plot_roc_curve()` - ROC curve
- `plot_pr_curve()` - Precision-Recall curve
- `plot_confusion_matrix()` - Confusion matrix heatmap

## Integration

These notebooks use the same production code:

```python
# Same extraction
extractor = FeatureExtractor(client)
data = extractor.extract_training_data(...)

# Same feature building
builder = FeatureBuilder()
X, y = builder.build_training_features(data)

# Same training
trainer = ModelTrainer(model_type='alert_scorer')
model, metrics = trainer.train(X, y)
```

This ensures consistency between notebook experiments and production models.

## Benefits

- **Interactive Development** - Quick experimentation with immediate feedback
- **Data Understanding** - Explore patterns and validate assumptions
- **Model Optimization** - Test parameters and compare approaches
- **Collaboration** - Share insights and document decisions
- **Production Validation** - Validate before deployment

## Tips

- Run cells sequentially to avoid dependency issues
- Use markdown cells to document findings and decisions
- Save notebooks with outputs for reproducibility
- Clear outputs before committing to version control
- Test notebook code in production scripts before deployment

## Support

For questions or issues:
- Check individual notebook documentation
- Review [`JUPYTER_NOTEBOOKS_PLAN.md`](../docs/agent/2025-10-29/claude/JUPYTER_NOTEBOOKS_PLAN.md)
- See [`ML_TRAINING_ARCHITECTURE.md`](../docs/agent/2025-10-29/claude/ML_TRAINING_ARCHITECTURE.md)