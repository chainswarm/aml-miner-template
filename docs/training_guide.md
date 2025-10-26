# Training Guide

Complete guide to training custom AML detection models.

## Table of Contents

- [Overview](#overview)
- [Training Data](#training-data)
- [Training Alert Scorer](#training-alert-scorer)
- [Training Alert Ranker](#training-alert-ranker)
- [Training Cluster Scorer](#training-cluster-scorer)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Management](#model-management)
- [Best Practices](#best-practices)

## Overview

The AML Miner Template includes three trainable models:

1. **Alert Scorer** - Binary classification to score individual alerts
2. **Alert Ranker** - Learning-to-rank model to prioritize alerts
3. **Cluster Scorer** - Classification model for transaction cluster analysis

Each model uses XGBoost and can be trained independently or together.

### Training Pipeline

```
Download Data → Feature Engineering → Model Training → Evaluation → Deployment
```

## Training Data

### Data Requirements

Training data must include:

- **Alert data**: Individual alerts with features and labels
- **Cluster data**: Transaction clusters with metadata
- **Labels**: Ground truth for supervised learning

### Download Training Data

Use the provided download script:

```bash
# Download a specific batch
bash scripts/download_batch.sh 1

# Download multiple batches
bash scripts/download_batch.sh 1 2 3
```

The script downloads data to `data/batch_N/` directories.

### Data Format

Alert data (`alerts.csv`):
```csv
alert_id,network,address,amount_usd,transaction_count,risk_category,label
alert-001,bitcoin,1A1z...,50000,10,high_value,1
alert-002,ethereum,0x74...,25000,5,medium_value,0
```

Cluster data (`clusters.csv`):
```csv
cluster_id,network,total_volume_usd,transaction_count,unique_addresses,pattern_matches,label
cluster-001,bitcoin,1000000,150,45,mixing;layering,1
cluster-002,ethereum,500000,80,25,none,0
```

### Data Validation

Verify downloaded data:

```bash
python scripts/validate_submission.py
```

Expected output:
```
✓ Data format valid
✓ Required columns present
✓ Label distribution: 70% negative, 30% positive
Data ready for training
```

## Training Alert Scorer

The alert scorer performs binary classification on individual alerts.

### Basic Training

```bash
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --output-path trained_models/alert_scorer.json
```

Expected output:
```
Loading data from data/batch_1/alerts.csv
Building features...
Training alert scorer...
Validation AUC: 0.8532
Model saved to trained_models/alert_scorer.json
```

### With Custom Parameters

```bash
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --output-path trained_models/alert_scorer.json \
  --max-depth 8 \
  --n-estimators 200 \
  --learning-rate 0.05 \
  --test-split 0.2
```

### Training Script

Alternatively, use the comprehensive training script:

```bash
python scripts/train_models.py \
  --model-type scorer \
  --data-path data/batch_1/alerts.csv
```

### Feature Engineering

The scorer uses these features:

- `amount_usd_log` - Logarithm of transaction amount
- `transaction_count_norm` - Normalized transaction count
- `network_encoded` - One-hot encoded network type
- `risk_category_encoded` - One-hot encoded risk category
- `temporal_features` - Hour, day, month from timestamp

### Understanding Metrics

**AUC (Area Under ROC Curve)**
- Measures ability to distinguish positive/negative cases
- Range: 0.5 (random) to 1.0 (perfect)
- Good: > 0.75, Excellent: > 0.85

**Precision**
- Proportion of true positives among predicted positives
- Important when false positives are costly

**Recall**
- Proportion of actual positives correctly identified
- Important when missing positives is costly

**F1 Score**
- Harmonic mean of precision and recall
- Good balance metric

### Example Training Session

```bash
$ python -m aml_miner.training.train_scorer --data-path data/batch_1/alerts.csv

Loading data: 10,000 samples
Train set: 8,000 samples
Test set: 2,000 samples

Building features...
Features created: 15 dimensions

Training XGBoost model...
[0] train-auc:0.8123 valid-auc:0.7945
[50] train-auc:0.8891 valid-auc:0.8234
[100] train-auc:0.9234 valid-auc:0.8456
[150] train-auc:0.9456 valid-auc:0.8512
Early stopping at iteration 152

Final Metrics:
- AUC: 0.8532
- Precision: 0.8245
- Recall: 0.7891
- F1: 0.8064

Model saved to trained_models/alert_scorer.json
```

## Training Alert Ranker

The alert ranker uses learning-to-rank to prioritize alerts.

### Basic Training

```bash
python -m aml_miner.training.train_ranker \
  --data-path data/batch_1/alerts.csv \
  --output-path trained_models/alert_ranker.json
```

Expected output:
```
Loading data from data/batch_1/alerts.csv
Building features...
Training alert ranker...
Validation NDCG@10: 0.7823
Model saved to trained_models/alert_ranker.json
```

### With Custom Parameters

```bash
python -m aml_miner.training.train_ranker \
  --data-path data/batch_1/alerts.csv \
  --output-path trained_models/alert_ranker.json \
  --max-depth 6 \
  --n-estimators 150 \
  --learning-rate 0.1
```

### Understanding Ranking Metrics

**NDCG (Normalized Discounted Cumulative Gain)**
- Measures ranking quality
- Considers position of relevant items
- NDCG@10: Quality of top 10 results
- Range: 0.0 to 1.0
- Good: > 0.70, Excellent: > 0.85

**MAP (Mean Average Precision)**
- Average precision across all positions
- Good for binary relevance

### Example Training Session

```bash
$ python -m aml_miner.training.train_ranker --data-path data/batch_1/alerts.csv

Loading data: 10,000 samples
Creating ranking groups by timestamp...
Train groups: 320
Test groups: 80

Building features...
Features created: 15 dimensions

Training LambdaMART model...
[0] train-ndcg@10:0.6234 valid-ndcg@10:0.6012
[50] train-ndcg@10:0.7823 valid-ndcg@10:0.7456
[100] train-ndcg@10:0.8345 valid-ndcg@10:0.7789
[150] train-ndcg@10:0.8567 valid-ndcg@10:0.7823
Early stopping at iteration 152

Final Metrics:
- NDCG@10: 0.7823
- NDCG@5: 0.8012
- MAP: 0.7645

Model saved to trained_models/alert_ranker.json
```

## Training Cluster Scorer

The cluster scorer classifies transaction clusters.

### Basic Training

```bash
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/clusters.csv \
  --output-path trained_models/cluster_scorer.json \
  --model-type cluster
```

### With Custom Parameters

```bash
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/clusters.csv \
  --output-path trained_models/cluster_scorer.json \
  --model-type cluster \
  --max-depth 10 \
  --n-estimators 300 \
  --learning-rate 0.03
```

### Cluster Features

The cluster scorer uses:

- `total_volume_usd_log` - Logarithm of total volume
- `transaction_count_norm` - Normalized transaction count
- `unique_addresses_norm` - Normalized unique addresses
- `density` - Cluster density metric
- `pattern_features` - One-hot encoded pattern matches

## Hyperparameter Tuning

Use the hyperparameter tuner for automatic optimization.

### Basic Tuning

```bash
python -m aml_miner.training.hyperparameter_tuner \
  --data-path data/batch_1/alerts.csv \
  --model-type scorer \
  --n-trials 50
```

### Advanced Tuning

```bash
python -m aml_miner.training.hyperparameter_tuner \
  --data-path data/batch_1/alerts.csv \
  --model-type scorer \
  --n-trials 100 \
  --cv-folds 5 \
  --metric auc \
  --output-path best_params.json
```

### Tuning Parameters

The tuner searches over:

- `max_depth`: [3, 4, 5, 6, 8, 10]
- `n_estimators`: [100, 200, 300, 500]
- `learning_rate`: [0.01, 0.03, 0.05, 0.1, 0.2]
- `min_child_weight`: [1, 3, 5]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]

### Example Tuning Session

```bash
$ python -m aml_miner.training.hyperparameter_tuner \
    --data-path data/batch_1/alerts.csv \
    --model-type scorer \
    --n-trials 50

Hyperparameter Tuning for Alert Scorer
======================================

Trial 1/50: max_depth=6, n_estimators=200, lr=0.05
  Cross-validation AUC: 0.8234

Trial 2/50: max_depth=8, n_estimators=300, lr=0.03
  Cross-validation AUC: 0.8456

...

Trial 50/50: max_depth=5, n_estimators=250, lr=0.07
  Cross-validation AUC: 0.8123

Best Parameters Found:
{
  "max_depth": 8,
  "n_estimators": 300,
  "learning_rate": 0.03,
  "min_child_weight": 3,
  "subsample": 0.8,
  "colsample_bytree": 0.8
}

Best Cross-validation AUC: 0.8567
Parameters saved to best_params.json
```

### Using Best Parameters

```bash
# Load best parameters and train
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --params-file best_params.json \
  --output-path trained_models/alert_scorer_tuned.json
```

## Model Evaluation

### Comprehensive Evaluation

```bash
python scripts/verify_training.py
```

Expected output:
```
Evaluating Alert Scorer...
  AUC: 0.8532
  Precision: 0.8245
  Recall: 0.7891
  F1: 0.8064
  ✓ Alert scorer passed

Evaluating Alert Ranker...
  NDCG@10: 0.7823
  MAP: 0.7645
  ✓ Alert ranker passed

Evaluating Cluster Scorer...
  AUC: 0.8912
  Precision: 0.8567
  Recall: 0.8234
  F1: 0.8398
  ✓ Cluster scorer passed

All models evaluated successfully!
```

### Manual Evaluation

```python
from aml_miner.models import AlertScorer
from aml_miner.utils import DataLoader
from sklearn.metrics import roc_auc_score, classification_report

# Load model
model = AlertScorer.load('trained_models/alert_scorer.json')

# Load test data
loader = DataLoader('data/batch_1/alerts.csv')
X_test, y_test = loader.load_features()

# Predict
predictions = model.predict(X_test)

# Evaluate
auc = roc_auc_score(y_test, predictions)
print(f"AUC: {auc:.4f}")

# Detailed report
y_pred_binary = (predictions > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))
```

### Cross-Validation

```python
from aml_miner.training import train_with_cv

# 5-fold cross-validation
cv_scores = train_with_cv(
    data_path='data/batch_1/alerts.csv',
    model_type='scorer',
    cv_folds=5
)

print(f"Mean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

## Model Management

### Saving Models

Models are automatically saved in XGBoost JSON format:

```bash
trained_models/
├── alert_scorer.json
├── alert_ranker.json
└── cluster_scorer.json
```

### Versioning Models

Include version in filename:

```bash
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --output-path trained_models/alert_scorer_v1.0.0.json
```

### Model Metadata

Create metadata file for each model:

```json
{
  "model_name": "alert_scorer_v1.0.0",
  "training_date": "2025-01-15",
  "data_version": "batch_1",
  "samples_count": 10000,
  "metrics": {
    "auc": 0.8532,
    "precision": 0.8245,
    "recall": 0.7891
  },
  "hyperparameters": {
    "max_depth": 8,
    "n_estimators": 300,
    "learning_rate": 0.03
  }
}
```

### Loading Models in Production

Update model paths in `aml_miner/config/model_config.yaml`:

```yaml
models:
  alert_scorer:
    path: trained_models/alert_scorer_v1.0.0.json
  alert_ranker:
    path: trained_models/alert_ranker_v1.0.0.json
  cluster_scorer:
    path: trained_models/cluster_scorer_v1.0.0.json
```

## Best Practices

### Data Quality

- **Clean labels**: Ensure ground truth is accurate
- **Balanced data**: Aim for 30-70% positive samples
- **Sufficient samples**: Minimum 1,000 samples per model
- **Feature diversity**: Include various transaction types

### Training Process

- **Train/test split**: Use 80/20 or 70/30 split
- **Cross-validation**: Use 5-fold CV for robust estimates
- **Early stopping**: Prevent overfitting
- **Feature scaling**: Normalize continuous features

### Model Selection

- **Start simple**: Begin with default parameters
- **Tune gradually**: Adjust one parameter at a time
- **Monitor metrics**: Track both training and validation
- **Compare models**: Keep multiple versions

### Avoiding Overfitting

Signs of overfitting:
- Training AUC > 0.95, Validation AUC < 0.80
- Large gap between train and validation metrics
- Model performs poorly on new data

Solutions:
- Increase `min_child_weight`
- Reduce `max_depth`
- Add regularization (`reg_alpha`, `reg_lambda`)
- Use more training data
- Enable early stopping

### Feature Engineering

Good features are:
- **Relevant**: Related to AML detection
- **Stable**: Consistent across time
- **Non-leaky**: Don't include future information
- **Interpretable**: Explainable to stakeholders

### Monitoring Performance

Track metrics over time:

```python
import json
from datetime import datetime

metrics = {
    'timestamp': datetime.now().isoformat(),
    'model': 'alert_scorer_v1.0.0',
    'auc': 0.8532,
    'precision': 0.8245,
    'recall': 0.7891
}

# Append to metrics log
with open('metrics_log.jsonl', 'a') as f:
    f.write(json.dumps(metrics) + '\n')
```

### Retraining Schedule

- **Weekly**: If you have fresh data
- **Monthly**: For stable datasets
- **On-demand**: When performance degrades
- **After incidents**: When new patterns emerge

### Production Deployment

Before deploying:

1. Validate on hold-out test set
2. Compare against baseline model
3. A/B test with small traffic percentage
4. Monitor production metrics
5. Have rollback plan ready

### Performance Optimization

For faster training:

```bash
# Use GPU if available
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --tree-method gpu_hist

# Reduce boosting rounds
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --n-estimators 100

# Increase learning rate (carefully)
python -m aml_miner.training.train_scorer \
  --data-path data/batch_1/alerts.csv \
  --learning-rate 0.1
```

## Troubleshooting

### Low Training Performance

If AUC < 0.70:
- Check data quality and labels
- Add more features
- Increase model complexity
- Get more training data

### Overfitting

If validation AUC much lower than training:
- Reduce `max_depth`
- Increase `min_child_weight`
- Add regularization
- Use early stopping

### Memory Issues

For large datasets:
- Train in batches
- Reduce feature dimensions
- Use sampling for hyperparameter tuning

### Slow Training

- Reduce number of estimators
- Use early stopping
- Enable GPU acceleration
- Parallelize with `n_jobs=-1`

## Next Steps

After training your models:

1. **Evaluate thoroughly** - Use [`verify_training.py`](../scripts/verify_training.py)
2. **Deploy to API** - Update model paths in config
3. **Monitor performance** - Track metrics in production
4. **Iterate** - Continuously improve with new data

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [API Reference](api_reference.md)
- [Customization Guide](customization.md)

Happy training! 🎯