# Troubleshooting Guide

Common issues and solutions for the risk scoring training pipeline.

## Training Issues

### "No labeled alerts found"

**Error:**
```
ValueError: No labeled alerts found. 
address_labels table must contain labels for alert addresses
```

**Cause:** No alerts have matching address labels in the database.

**Solutions:**

1. **Check if address_labels has data:**
```python
from packages.storage import ClientFactory, get_connection_params
params = get_connection_params('torus')
factory = ClientFactory(params)
with factory.client_context() as client:
    result = client.command(
        "SELECT COUNT(*) FROM raw_address_labels "
        "WHERE processing_date = '2025-08-01' AND window_days = 195"
    )
    print(f"Address labels count: {result}")
```

2. **Check if alerts have data:**
```python
with client_factory.client_context() as client:
    result = client.command(
        "SELECT COUNT(*) FROM raw_alerts "
        "WHERE processing_date = '2025-08-01' AND window_days = 195"
    )
    print(f"Alerts count: {result}")
```

3. **Check for overlap:**
```python
with client_factory.client_context() as client:
    result = client.query("""
        SELECT COUNT(DISTINCT a.address)
        FROM raw_alerts a
        INNER JOIN raw_address_labels l ON a.address = l.address
        WHERE a.processing_date = '2025-08-01' 
          AND a.window_days = 195
          AND l.processing_date = '2025-08-01'
          AND l.window_days = 195
    """)
    print(f"Alerts with labels: {result.result_rows[0][0]}")
```

4. **Add custom labels** if no overlap exists:
See [ADDING_CUSTOM_LABELS.md](ADDING_CUSTOM_LABELS.md)

---

### "Column not found" errors

**Error:**
```
DatabaseError: Unknown expression identifier `total_received_usd`
```

**Cause:** Schema mismatch between code and database.

**Solutions:**

1. **Check schema version:**
```bash
grep -r "total_received_usd" packages/storage/schema/
# Should not find anything - old column name
```

2. **Re-initialize database:**
```bash
python scripts/init_database.py
```

3. **Verify schema matches code:**
```python
with client_factory.client_context() as client:
    result = client.query("DESCRIBE raw_features")
    columns = [row[0] for row in result.result_rows]
    print("Columns:", columns)
    # Should see: total_in_usd, total_out_usd (not total_received_usd, total_sent_usd)
```

---

### Low Model Performance (AUC < 0.6)

**Symptoms:**
```
Model evaluation: AUC=0.52, PR-AUC=0.48
```

**Causes & Solutions:**

1. **Not enough labeled data**
```python
# Check label count
with client_factory.client_context() as client:
    result = client.query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN risk_level IN ('high', 'critical') THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN risk_level IN ('low', 'medium') THEN 1 ELSE 0 END) as negative
        FROM raw_address_labels
        WHERE processing_date = '2025-08-01' AND window_days = 195
    """)
    print(result.result_rows[0])
```

**Solution:** Add more labeled addresses or use longer date range.

2. **Poor label quality**

Check label sources and confidence:
```python
with client_factory.client_context() as client:
    result = client.query("""
        SELECT source, AVG(confidence_score), COUNT(*)
        FROM raw_address_labels
        GROUP BY source
    """)
    for row in result.result_rows:
        print(f"{row[0]}: confidence={row[1]:.2f}, count={row[2]}")
```

**Solution:** Focus on high-confidence labels or implement custom label strategy.

3. **Class imbalance**

Check positive/negative ratio:
```python
print(f"Positive rate: {y.mean():.1%}")
# Should be between 10% and 90%
```

**Solution:** Use sample weights or adjust strategy.

---

### Memory Issues

**Error:**
```
MemoryError: Unable to allocate array
```

**Cause:** Too much data loaded into memory.

**Solutions:**

1. **Reduce date range:**
```bash
# Instead of months
--start-date 2025-07-01 --end-date 2025-07-31

# Use days
--start-date 2025-08-01 --end-date 2025-08-01
```

2. **Sample data:**
```python
# In feature_extraction.py, add LIMIT
query = f"""
    SELECT ...
    FROM raw_alerts
    WHERE ...
    ORDER BY RAND()  -- Random sampling
    LIMIT 10000
"""
```

3. **Use batching:**
```python
# Process in chunks
for date in date_range:
    data = extract_training_data(date, date)
    model.partial_fit(X, y)
```

---

## Data Issues

### Missing Data

**Check what data exists:**
```python
from packages.storage import ClientFactory, get_connection_params

params = get_connection_params('torus')
factory = ClientFactory(params)

with factory.client_context() as client:
    tables = ['raw_alerts', 'raw_features', 'raw_clusters', 
              'raw_money_flows', 'raw_address_labels']
    
    for table in tables:
        result = client.command(f"SELECT COUNT(*) FROM {table}")
        print(f"{table}: {result} rows")
```

**Solution:** Run ingestion to populate data:
```bash
python packages/ingestion/sot_ingestion.py --network torus
```

---

### Date Mismatch

**Error:** Data for one date but trying to train on another.

**Check available dates:**
```python
with client_factory.client_context() as client:
    result = client.query("""
        SELECT DISTINCT processing_date, window_days, COUNT(*)
        FROM raw_alerts
        GROUP BY processing_date, window_days
        ORDER BY processing_date, window_days
    """)
    for row in result.result_rows:
        print(f"Date: {row[0]}, Window: {row[1]}, Alerts: {row[2]}")
```

**Solution:** Use dates that actually exist in the database.

---

## Configuration Issues

### Wrong Network

**Error:** No data for specified network.

**Check networks:**
```python
with client_factory.client_context() as client:
    result = client.query("SELECT DISTINCT network FROM raw_address_labels")
    networks = [row[0] for row in result.result_rows]
    print(f"Available networks: {networks}")
```

**Solution:** Use correct network name in `--network` parameter.

---

### Connection Issues

**Error:**
```
ConnectionError: Unable to connect to ClickHouse
```

**Solutions:**

1. **Check ClickHouse is running:**
```bash
docker ps | grep clickhouse
```

2. **Check connection params:**
```bash
cat .env
# Should have CLICKHOUSE_HOST, CLICKHOUSE_PORT, etc.
```

3. **Test connection:**
```python
from packages.storage import get_connection_params, ClientFactory

try:
    params = get_connection_params('torus')
    factory = ClientFactory(params)
    with factory.client_context() as client:
        result = client.command("SELECT 1")
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
```

---

## Model Issues

### Model Not Saving

**Error:** Model trained but not found in output directory.

**Check output directory:**
```python
from pathlib import Path

output_dir = Path('trained_models/torus')
print(f"Output dir exists: {output_dir.exists()}")
print(f"Contents: {list(output_dir.glob('*'))}")
```

**Solutions:**

1. **Check permissions:**
```bash
ls -la trained_models/
```

2. **Specify custom directory:**
```bash
python packages/training/model_training.py \
    --output-dir ./my_models \
    ...
```

3. **Check for errors in logs:**
```bash
# Look for "Saving model" messages
grep "Saving model" your.log
```

---

### Model Loading Errors

**Error:**
```
FileNotFoundError: Model file not found
```

**Solutions:**

1. **Check model path:**
```python
from pathlib import Path
model_path = Path('trained_models/torus/model_id/model.joblib')
print(f"Exists: {model_path.exists()}")
```

2. **List available models:**
```python
import os
for root, dirs, files in os.walk('trained_models'):
    for file in files:
        if file.endswith('.joblib'):
            print(os.path.join(root, file))
```

---

## Strategy Issues

### Custom Strategy Not Used

**Symptom:** Custom strategy defined but default still used.

**Check:**
```python
# Make sure you're passing it to ModelTraining
training = ModelTraining(
    ...,
    label_strategy=MyCustomStrategy(),  # Did you add this?
    model_trainer=MyCustomTrainer()     # And this?
)
```

---

### Import Errors

**Error:**
```
ImportError: cannot import name 'MyCustomStrategy'
```

**Solutions:**

1. **Check file location:**
```bash
ls packages/training/strategies/
# Should see your custom_strategy.py
```

2. **Check import path:**
```python
# Correct
from packages.training.strategies.custom_strategy import MyCustomStrategy

# Not from relative import if running from different directory
```

3. **Add to __init__.py:**
```python
# packages/training/strategies/__init__.py
from .custom_strategy import MyCustomStrategy
__all__.append('MyCustomStrategy')
```

---

## Performance Issues

### Slow Training

**Solutions:**

1. **Reduce data size:**
   - Smaller date range
   - Sample alerts
   - Fewer features

2. **Use faster model:**
```python
# Instead of XGBoost with many estimators
trainer = XGBoostTrainer(hyperparameters={'n_estimators': 50})

# Or use LightGBM (faster)
from packages.training.strategies.lightgbm_trainer import LightGBMTrainer
trainer = LightGBMTrainer()
```

3. **Parallelize:**
```python
trainer = XGBoostTrainer(hyperparameters={'n_jobs': -1})  # Use all cores
```

---

### Slow Feature Extraction

**Solutions:**

1. **Add indexes to database:**
```sql
CREATE INDEX IF NOT EXISTS idx_date_window 
ON raw_alerts(processing_date, window_days);
```

2. **Reduce data:**
```python
# Limit query results
query = f"SELECT ... FROM raw_alerts WHERE ... LIMIT 10000"
```

3. **Use sampling:**
```python
query = f"SELECT ... FROM raw_alerts WHERE ... AND rand() < 0.1"  # 10% sample
```

---

## Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or with loguru
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

---

## Getting Help

1. **Check logs carefully** - error messages usually indicate the problem
2. **Verify data** - most issues are data-related
3. **Test incrementally** - start simple, add complexity gradually
4. **Check documentation**:
   - [TRAINING_QUICKSTART.md](../TRAINING_QUICKSTART.md)
   - [ADDING_CUSTOM_LABELS.md](ADDING_CUSTOM_LABELS.md)
   - [MINER_CUSTOMIZATION_GUIDE.md](MINER_CUSTOMIZATION_GUIDE.md)

---

## Common Patterns

### Full Diagnostic Check

```python
from packages.storage import ClientFactory, get_connection_params

params = get_connection_params('torus')
factory = ClientFactory(params)

with factory.client_context() as client:
    # Check all tables
    tables = {
        'raw_alerts': 'processing_date, window_days',
        'raw_features': 'processing_date, window_days',
        'raw_address_labels': 'processing_date, window_days',
        'raw_clusters': 'processing_date, window_days',
        'raw_money_flows': 'processing_date, window_days'
    }
    
    date = '2025-08-01'
    window = 195
    
    print(f"Data check for {date}, window={window}:")
    print("-" * 60)
    
    for table, fields in tables.items():
        query = f"""
            SELECT COUNT(*) 
            FROM {table} 
            WHERE processing_date = '{date}' 
              AND window_days = {window}
        """
        try:
            result = client.command(query)
            print(f"{table:25} : {result:>10} rows")
        except Exception as e:
            print(f"{table:25} : ERROR - {str(e)[:30]}")
```

This will show you exactly what data is available.