# Training Quickstart Guide

Get your first model trained in 5 minutes!

## Prerequisites

1. **Database Running**: ClickHouse must be running with data
2. **Environment Setup**: Python environment with dependencies installed
3. **Data Available**: At least one batch of data ingested

## Quick Start

### Step 1: Verify Data

Check you have data to train on:

```bash
# Check if you have alerts
python -c "
from packages.storage import ClientFactory, get_connection_params
params = get_connection_params('torus')
factory = ClientFactory(params)
with factory.client_context() as client:
    result = client.command('SELECT COUNT(*) FROM raw_alerts WHERE processing_date = \\'2025-08-01\\' AND window_days = 195')
    print(f'Alerts: {result}')
    result = client.command('SELECT COUNT(*) FROM raw_address_labels WHERE processing_date = \\'2025-08-01\\' AND window_days = 195')
    print(f'Address Labels: {result}')
"
```

Expected output:
```
Alerts: 86
Address Labels: 126
```

### Step 2: Run Training

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195
```

### Step 3: Check Results

Look for success messages:

```
✓ Extracting training data from ClickHouse
✓ Deriving labels from address_labels (SOT baseline)
✓ Labeled X/Y alerts: A positive, B negative
✓ Building feature matrix
✓ Training XGBoost with X samples, Y features
✓ XGBoost training completed
✓ Model evaluation: AUC=0.XXXX, PR-AUC=0.YYYY
✓ Training workflow completed successfully
```

Model saved to: `trained_models/torus/{model_id}/`

## What Just Happened?

1. **Data Extraction**: Loaded alerts, features, and labels from ClickHouse
2. **Label Derivation**: Used address_labels to create ground truth
   - High/Critical risk = Suspicious (label=1)
   - Low/Medium risk = Normal (label=0)
3. **Feature Building**: Created feature matrix from alerts + features
4. **Model Training**: Trained XGBoost classifier
5. **Evaluation**: Measured performance with AUC metrics
6. **Storage**: Saved model to disk and metadata to database

## Understanding the Output

### Label Statistics
```
Labeled 50/86 alerts: 15 positive, 35 negative
```
- **50/86**: 50 alerts had matching address labels out of 86 total
- **15 positive**: 15 alerts from high/critical risk addresses
- **35 negative**: 35 alerts from low/medium risk addresses

### Performance Metrics
```
AUC=0.8234, PR-AUC=0.7156
```
- **AUC**: 0.5 = random, 1.0 = perfect (0.82 is good)
- **PR-AUC**: Precision-Recall AUC, better for imbalanced data

## Common Issues

### "No labeled alerts found"

**Cause**: No overlap between alerts and address_labels

**Solution**: Check date ranges match:
```bash
# Check what dates you have
python -c "
from packages.storage import ClientFactory, get_connection_params
params = get_connection_params('torus')
factory = ClientFactory(params)
with factory.client_context() as client:
    result = client.query('SELECT DISTINCT processing_date FROM raw_alerts ORDER BY processing_date')
    print('Available dates:', [row[0] for row in result.result_rows])
"
```

### "Column not found" errors

**Cause**: Schema mismatch

**Solution**: Re-initialize database:
```bash
python scripts/init_database.py
```

### Low AUC (< 0.6)

**Cause**: Not enough labeled data or poor label quality

**Solutions**:
1. Add more labeled addresses to `raw_address_labels`
2. Use longer date range (--start-date to --end-date)
3. Customize label strategy (see [MINER_CUSTOMIZATION_GUIDE.md](docs/MINER_CUSTOMIZATION_GUIDE.md))

## Next Steps

### Option 1: Train on Different Dates
```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-07-01 \
    --end-date 2025-07-31 \
    --model-type alert_scorer \
    --window-days 195
```

### Option 2: Add Custom Labeled Addresses

See [ADDING_CUSTOM_LABELS.md](docs/ADDING_CUSTOM_LABELS.md)

### Option 3: Customize Training

See [MINER_CUSTOMIZATION_GUIDE.md](docs/MINER_CUSTOMIZATION_GUIDE.md) for:
- Custom label strategies
- Custom ML models
- Advanced techniques

## Training Parameters

### Required
- `--network`: Network identifier (e.g., torus, ethereum)
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)

### Optional
- `--model-type`: alert_scorer | alert_ranker | cluster_scorer (default: alert_scorer)
- `--window-days`: 7 | 30 | 90 | 195 (default: 7)
- `--output-dir`: Custom output directory (default: trained_models/{network})

## Understanding the Baseline

**What you get out-of-the-box:**

✅ **Labels**: SOT's address_labels (exchanges, mixers, scams, etc.)  
✅ **Features**: Comprehensive blockchain features from raw_features  
✅ **Model**: XGBoost classifier with standard hyperparameters  
✅ **Evaluation**: AUC and PR-AUC metrics  

**What you can customize:**

🔧 **Labels**: Add your own labeled addresses  
🔧 **Label Strategy**: Custom logic for deriving labels  
🔧 **Features**: Add custom feature engineering  
🔧 **Model**: Use any ML algorithm (neural nets, LightGBM, etc.)  
🔧 **Hyperparameters**: Tune for better performance  

## Help & Documentation

- **Quick Start**: This file (TRAINING_QUICKSTART.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **Customization**: [docs/MINER_CUSTOMIZATION_GUIDE.md](docs/MINER_CUSTOMIZATION_GUIDE.md)
- **Package Details**: [packages/training/README.md](packages/training/README.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Support

If you encounter issues:
1. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. Review error messages carefully
3. Verify data is available in ClickHouse
4. Check schema matches expected format

---

**Ready to customize? See [docs/MINER_CUSTOMIZATION_GUIDE.md](docs/MINER_CUSTOMIZATION_GUIDE.md)**