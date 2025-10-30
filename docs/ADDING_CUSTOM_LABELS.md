# Adding Custom Labels Guide

Learn how to add your own labeled addresses to improve model performance.

## Why Add Custom Labels?

The SOT baseline provides `raw_address_labels` with exchanges, mixers, scams, etc. But you can:

✅ Add proprietary threat intelligence  
✅ Include addresses from your own research  
✅ Label addresses from external sources  
✅ Gain competitive advantage through better data  

## Quick Method: Insert into Database

### Step 1: Prepare Your Data

Create a CSV or DataFrame with your labeled addresses:

```python
import pandas as pd

# Your custom labeled addresses
custom_labels = pd.DataFrame({
    'processing_date': '2025-08-01',  # Same as training date
    'window_days': 195,               # Same as training window
    'network': 'torus',               # Your network
    'address': [
        '0xabc123...',
        '0xdef456...',
        '0xghi789...',
        # ... more addresses
    ],
    'label': [
        'scam_address',
        'mixer',
        'exchange',
        # ... corresponding labels
    ],
    'risk_level': [
        'critical',  # Maps to label=1 (suspicious)
        'high',      # Maps to label=1 (suspicious)
        'low',       # Maps to label=0 (normal)
        # ... corresponding risk levels
    ],
    'confidence_score': [
        0.95,  # How confident you are (0.0-1.0)
        0.90,
        0.85,
        # ... confidence scores
    ],
    'source': 'miner_custom_intelligence'  # Track where labels came from
})
```

### Step 2: Insert into ClickHouse

```python
from packages.storage import ClientFactory, get_connection_params

# Connect to database
connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    # Insert your custom labels
    client.insert_df('raw_address_labels', custom_labels)
    
    # Verify insertion
    result = client.command(
        f"SELECT COUNT(*) FROM raw_address_labels "
        f"WHERE source = 'miner_custom_intelligence'"
    )
    print(f"Inserted {result} custom labels")
```

### Step 3: Train with Combined Dataset

```bash
python packages/training/model_training.py \
    --network torus \
    --start-date 2025-08-01 \
    --end-date 2025-08-01 \
    --model-type alert_scorer \
    --window-days 195
```

The default `AddressLabelStrategy` will automatically use both SOT and your custom labels!

## Label Schema

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processing_date` | Date | Date this label applies to | '2025-08-01' |
| `window_days` | UInt16 | Window size | 195 |
| `network` | String | Network identifier | 'torus' |
| `address` | String | Blockchain address | '0xabc...' |
| `label` | String | Label description | 'scam_address' |
| `risk_level` | String | Risk level (see below) | 'high' |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `confidence_score` | Float32 | Confidence (0.0-1.0) | 0.5 |
| `source` | String | Where label came from | '' |
| `address_type` | String | Type of address | 'unknown' |
| `address_subtype` | String | Subtype | '' |

### Risk Level Mapping

| Risk Level | Training Label | Meaning |
|------------|---------------|---------|
| `critical` | 1 (positive) | Definitely suspicious |
| `high` | 1 (positive) | Likely suspicious |
| `medium` | 0 (negative) | Likely normal |
| `low` | 0 (negative) | Definitely normal |
| Other | Unlabeled | Not used for training |

## Label Sources

### 1. Manual Research

```python
# Addresses you manually verified
manual_labels = pd.DataFrame({
    'address': ['0x123...', '0x456...'],
    'label': ['confirmed_scam', 'known_exchange'],
    'risk_level': ['critical', 'low'],
    'confidence_score': [1.0, 1.0],
    'source': 'manual_verification'
})
```

### 2. External APIs

```python
import requests

def label_from_etherscan(address):
    # Example: Get labels from Etherscan
    response = requests.get(
        f'https://api.etherscan.io/api',
        params={
            'module': 'account',
            'action': 'balance',
            'address': address,
            # ... other params
        }
    )
    # Parse and return label
    return label_info

# Batch label addresses
addresses = ['0x123...', '0x456...']
labels = [label_from_etherscan(addr) for addr in addresses]
```

### 3. Community Datasets

```python
# Load from external source
community_labels = pd.read_csv('https://example.com/scam_addresses.csv')

# Transform to expected format
custom_labels = pd.DataFrame({
    'processing_date': '2025-08-01',
    'window_days': 195,
    'network': 'torus',
    'address': community_labels['address'],
    'label': community_labels['type'],
    'risk_level': community_labels['risk'],
    'confidence_score': 0.7,  # Lower confidence for community data
    'source': 'community_dataset'
})
```

### 4. Behavioral Analysis

```python
# Example: Label based on behavioral patterns
from packages.storage import ClientFactory, get_connection_params

connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    # Find addresses with suspicious patterns
    query = """
        SELECT 
            address,
            total_volume_usd,
            is_mixer_like,
            behavioral_anomaly_score
        FROM raw_features
        WHERE processing_date = '2025-08-01'
          AND window_days = 195
          AND behavioral_anomaly_score > 0.8
    """
    result = client.query(query)
    
    # Create labels from behavioral analysis
    behavioral_labels = pd.DataFrame({
        'processing_date': '2025-08-01',
        'window_days': 195,
        'network': 'torus',
        'address': [row[0] for row in result.result_rows],
        'label': 'high_anomaly_behavior',
        'risk_level': 'high',
        'confidence_score': [row[3] for row in result.result_rows],
        'source': 'behavioral_heuristic'
    })
```

## Best Practices

### 1. Use Confidence Scores

Higher confidence = more weight in training:

```python
custom_labels['confidence_score'] = [
    1.0,  # Manually verified
    0.9,  # External API
    0.7,  # Community source
    0.5,  # Heuristic-based
]
```

### 2. Track Label Sources

Always set the `source` field:

```python
custom_labels['source'] = 'proprietary_intelligence_v1'
```

This helps you:
- Track which labels help performance
- Debug label quality issues
- Reproduce experiments

### 3. Validate Before Inserting

```python
def validate_labels(df):
    # Check required fields
    required = ['processing_date', 'window_days', 'network', 'address', 'label', 'risk_level']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Check risk levels
    valid_risk = ['low', 'medium', 'high', 'critical']
    invalid = df[~df['risk_level'].isin(valid_risk)]
    if len(invalid) > 0:
        raise ValueError(f"Invalid risk levels: {invalid['risk_level'].unique()}")
    
    # Check confidence scores
    if (df['confidence_score'] < 0).any() or (df['confidence_score'] > 1).any():
        raise ValueError("Confidence scores must be between 0 and 1")
    
    return True

validate_labels(custom_labels)
```

### 4. Version Your Labels

```python
# Version 1
custom_labels['source'] = 'miner_labels_v1'

# Later, improved version
improved_labels['source'] = 'miner_labels_v2'
```

Then you can compare:

```python
# Train with v1
training = ModelTraining(..., label_strategy=CustomStrategy(version='v1'))

# Train with v2
training = ModelTraining(..., label_strategy=CustomStrategy(version='v2'))

# Compare performance
```

### 5. Balance Your Dataset

Avoid extreme class imbalance:

```python
# Check balance
positive = (custom_labels['risk_level'].isin(['high', 'critical'])).sum()
negative = (custom_labels['risk_level'].isin(['low', 'medium'])).sum()

print(f"Positive: {positive}, Negative: {negative}, Ratio: {positive/negative:.2f}")

# Aim for ratio between 0.1 and 10
# If too imbalanced, sample or collect more labels
```

## Advanced: Custom Label Strategy

For more control, implement a custom label strategy:

```python
# packages/training/strategies/custom_label_strategy.py

from packages.training.strategies import LabelStrategy, AddressLabelStrategy
import pandas as pd

class CustomLabelStrategy(LabelStrategy):
    def __init__(self, version='v1'):
        self.base_strategy = AddressLabelStrategy()
        self.version = version
    
    def derive_labels(self, alerts_df, data):
        # Start with SOT baseline
        alerts_df = self.base_strategy.derive_labels(alerts_df, data)
        
        # Add your custom labels with higher priority
        custom_labels = self._load_custom_labels(data['address_labels'], self.version)
        alerts_df = self._merge_custom_labels(alerts_df, custom_labels)
        
        return alerts_df
    
    def _load_custom_labels(self, labels_df, version):
        # Filter to your custom labels
        return labels_df[labels_df['source'] == f'miner_labels_{version}']
    
    def _merge_custom_labels(self, alerts_df, custom_labels):
        # Merge with priority for custom labels
        merged = alerts_df.merge(
            custom_labels[['address', 'risk_level', 'confidence_score']],
            on='address',
            how='left',
            suffixes=('_sot', '_custom')
        )
        
        # Use custom label if available
        merged['label'] = merged['label_custom'].fillna(merged['label_sot'])
        merged['label_confidence'] = merged['confidence_score_custom'].fillna(
            merged['label_confidence']
        )
        
        return merged
    
    def validate_labels(self, alerts_df):
        return self.base_strategy.validate_labels(alerts_df)
```

## Monitoring Label Quality

### Check Label Distribution

```python
from packages.storage import ClientFactory, get_connection_params

connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    query = """
        SELECT 
            source,
            risk_level,
            COUNT(*) as count,
            AVG(confidence_score) as avg_confidence
        FROM raw_address_labels
        WHERE processing_date = '2025-08-01'
          AND window_days = 195
        GROUP BY source, risk_level
        ORDER BY source, risk_level
    """
    result = client.query(query)
    
    print("Label Distribution:")
    for row in result.result_rows:
        print(f"  {row[0]:30} | {row[1]:10} | Count: {row[2]:5} | Conf: {row[3]:.2f}")
```

### Compare Performance

```python
# Baseline (SOT only)
baseline_auc = 0.75

# With your custom labels
custom_auc = 0.82

improvement = (custom_auc - baseline_auc) / baseline_auc
print(f"Improvement: {improvement:.1%}")
```

## Example: Complete Workflow

```python
import pandas as pd
from packages.storage import ClientFactory, get_connection_params

# 1. Prepare custom labels
custom_labels = pd.DataFrame({
    'processing_date': '2025-08-01',
    'window_days': 195,
    'network': 'torus',
    'address': ['0x123...', '0x456...', '0x789...'],
    'label': ['scam', 'exchange', 'mixer'],
    'risk_level': ['critical', 'low', 'high'],
    'confidence_score': [1.0, 0.9, 0.95],
    'source': 'my_custom_intelligence_v1'
})

# 2. Validate
assert set(custom_labels['risk_level']).issubset({'low', 'medium', 'high', 'critical'})
assert custom_labels['confidence_score'].between(0, 1).all()

# 3. Insert
connection_params = get_connection_params('torus')
client_factory = ClientFactory(connection_params)

with client_factory.client_context() as client:
    client.insert_df('raw_address_labels', custom_labels)
    print(f"Inserted {len(custom_labels)} custom labels")

# 4. Train and compare
# First, train with baseline (delete custom labels temporarily)
# Then, re-add custom labels and train again
# Compare AUC scores
```

## Tips for Competitive Advantage

1. **Quality > Quantity**: 100 high-confidence labels > 1000 uncertain labels
2. **Diverse Sources**: Combine multiple label sources
3. **Keep It Fresh**: Update labels as threats evolve
4. **Validate Continuously**: Check label accuracy regularly
5. **Track Performance**: Measure which label sources help most

## Next Steps

- **Beginner**: Add 10-20 manually verified addresses
- **Intermediate**: Integrate external API for labels
- **Advanced**: Implement custom label strategy with ensemble logic
- **Expert**: Build automated labeling pipeline

See [MINER_CUSTOMIZATION_GUIDE.md](MINER_CUSTOMIZATION_GUIDE.md) for more advanced customization.