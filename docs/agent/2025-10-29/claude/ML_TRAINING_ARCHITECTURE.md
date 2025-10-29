# ML Training Architecture

**Date**: 2025-10-29  
**Purpose**: Design ML training system following ingestion conventions  
**Status**: Architecture Design

---

## Executive Summary

Design a clean ML training pipeline in `packages/training/` that follows the same conventions as `packages/ingestion/`, enabling systematic model training from ingested ClickHouse data.

### Key Principles

✅ **Convention over Configuration** - Follow ingestion patterns  
✅ **No Data Migrations** - Always start fresh  
✅ **Fail Fast** - Raise exceptions, no fallbacks  
✅ **LLM-Generated Code** - No manual work breakdown  
✅ **ClickHouse Native** - Direct database integration  

---

## Architecture Overview

### Directory Structure

```
packages/
├── ingestion/              # Existing - Data ingestion
│   ├── __init__.py
│   └── sot_ingestion.py
│
├── storage/                # Existing - ClickHouse abstraction
│   ├── __init__.py
│   └── schema/
│
└── training/               # NEW - ML training pipeline
    ├── __init__.py
    ├── model_training.py   # Main training orchestration
    ├── feature_extraction.py  # Extract from ClickHouse
    └── model_storage.py    # Model versioning & persistence
```

---

## Training Workflow

### Conceptual Flow

```
┌─────────────────────────────────────────────────────┐
│  1. Data Extraction (from ClickHouse)               │
│     - Query raw_alerts, raw_features, etc.          │
│     - Filter by date range                          │
│     - Load into memory as DataFrames                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  2. Feature Engineering                             │
│     - Build alert features                          │
│     - Build network features                        │
│     - Build temporal features                       │
│     - Build statistical features                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  3. Model Training                                  │
│     - Train/test split                              │
│     - Cross-validation                              │
│     - Hyperparameter optimization                   │
│     - Model evaluation                              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  4. Model Storage                                   │
│     - Save model artifacts                          │
│     - Store metadata                                │
│     - Version tracking                              │
│     - Performance metrics                           │
└─────────────────────────────────────────────────────┘
```

### Parallel with Ingestion

| **Ingestion**                | **Training**                    |
|------------------------------|----------------------------------|
| Download from S3             | Extract from ClickHouse          |
| Validate parquet files       | Validate feature completeness    |
| Transform & clean            | Feature engineering              |
| Load into ClickHouse         | Train models                     |
| Store metadata               | Store model + metadata           |

---

## Core Components

### 1. ModelTraining Class

Main orchestration following [`SOTDataIngestion`](../../packages/ingestion/sot_ingestion.py:17) pattern:

```python
# packages/training/model_training.py

from abc import ABC
from pathlib import Path
from typing import Dict, Any
import lightgbm as lgb
import pandas as pd
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import ClientFactory

class ModelTraining(ABC):
    
    def __init__(
        self, 
        network: str,
        start_date: str,
        end_date: str,
        client: Client,
        output_dir: Path
    ):
        self.network = network
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Main training workflow"""
        if terminate_event.is_set():
            logger.info("Termination requested before start")
            return
        
        logger.info(
            "Starting training workflow",
            extra={
                "network": self.network,
                "start_date": self.start_date,
                "end_date": self.end_date
            }
        )
        
        # Check if model already exists
        if self._model_exists():
            logger.success("Model already trained for this configuration")
            return
        
        # Extract data from ClickHouse
        logger.info("Extracting training data from ClickHouse")
        data = self._extract_training_data()
        
        if terminate_event.is_set():
            logger.warning("Termination requested after extraction")
            return
        
        # Build features
        logger.info("Building features")
        X, y = self._build_features(data)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after feature building")
            return
        
        # Train model
        logger.info("Training model")
        model, metrics = self._train_model(X, y)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after training")
            return
        
        # Save model and metadata
        logger.info("Saving model and metadata")
        self._save_model(model, metrics)
        
        logger.success(
            "Training workflow completed successfully",
            extra={
                "model_path": self.output_dir,
                "metrics": metrics
            }
        )
```

### 2. Feature Extraction

Extract data directly from ClickHouse:

```python
# packages/training/feature_extraction.py

import pandas as pd
from clickhouse_connect.driver import Client
from loguru import logger

class FeatureExtractor:
    
    def __init__(self, client: Client):
        self.client = client
    
    def extract_training_data(
        self,
        network: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Extract all required data for training"""
        
        logger.info("Extracting alerts")
        alerts = self._extract_alerts(network, start_date, end_date)
        
        logger.info("Extracting features")
        features = self._extract_features(network, start_date, end_date)
        
        logger.info("Extracting clusters")
        clusters = self._extract_clusters(network, start_date, end_date)
        
        logger.info("Extracting money flows")
        money_flows = self._extract_money_flows(network, start_date, end_date)
        
        logger.success(
            "Data extraction completed",
            extra={
                "alerts": len(alerts),
                "features": len(features),
                "clusters": len(clusters),
                "money_flows": len(money_flows)
            }
        )
        
        return {
            'alerts': alerts,
            'features': features,
            'clusters': clusters,
            'money_flows': money_flows
        }
    
    def _extract_alerts(
        self,
        network: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Extract alerts from ClickHouse"""
        
        query = f"""
            SELECT *
            FROM raw_alerts
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            raise ValueError(
                f"No alerts found for date range {start_date} to {end_date}"
            )
        
        df = pd.DataFrame(
            result.result_rows,
            columns=[col[0] for col in result.column_names]
        )
        
        logger.info(f"Extracted {len(df)} alerts")
        return df
```

### 3. Model Storage

Version and persist trained models:

```python
# packages/training/model_storage.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import lightgbm as lgb
from loguru import logger

class ModelStorage:
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: lgb.Booster,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> Path:
        """Save model with metadata"""
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        version = metadata.get('version', '1.0.0')
        
        model_filename = f"{model_type}_v{version}_{timestamp}.txt"
        model_path = self.output_dir / model_filename
        
        # Save LightGBM model
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata_content = {
            'model_type': model_type,
            'version': version,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_names': model.feature_name(),
            'num_trees': model.num_trees(),
            **metadata
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_content, f, indent=2)
        
        logger.success(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: Path) -> lgb.Booster:
        """Load model from disk"""
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = lgb.Booster(model_file=str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def get_latest_model(self, model_type: str) -> Path:
        """Get most recent model of given type"""
        
        pattern = f"{model_type}_v*.txt"
        models = sorted(self.output_dir.glob(pattern), reverse=True)
        
        if not models:
            raise FileNotFoundError(
                f"No models found for type {model_type}"
            )
        
        latest = models[0]
        logger.info(f"Latest model: {latest}")
        
        return latest
```

---

## Training Script

Following the ingestion script pattern:

```python
# packages/training/model_training.py (continued - main entry point)

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='alert_scorer')
    parser.add_argument('--output-dir', type=Path, default=Path('trained_models'))
    args = parser.parse_args()
    
    service_name = f'{args.network}-{args.model_type}-training'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info(
        "Initializing model training",
        extra={
            "network": args.network,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "model_type": args.model_type
        }
    )
    
    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        training = ModelTraining(
            network=args.network,
            start_date=args.start_date,
            end_date=args.end_date,
            client=client,
            output_dir=args.output_dir
        )
        
        training.run()
```

---

## Usage Examples

### Train Alert Scorer

```bash
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer \
    --output-dir trained_models/
```

### Train Alert Ranker

```bash
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_ranker \
    --output-dir trained_models/
```

### Train Cluster Scorer

```bash
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type cluster_scorer \
    --output-dir trained_models/
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ClickHouse Database                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │raw_alerts  │  │raw_features│  │raw_clusters│            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         └───────────────┴────────────────┘                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ SQL Query (date filtered)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  FeatureExtractor                            │
│  - Extract alerts, features, clusters, money_flows           │
│  - Convert to pandas DataFrames                              │
│  - Validate completeness                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ DataFrames
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  FeatureBuilder                              │
│  - Alert-level features                                      │
│  - Network features                                          │
│  - Temporal features                                         │
│  - Statistical features                                      │
│  - Cluster features                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Feature Matrix (X, y)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  ModelTrainer                                │
│  - Train/test split                                          │
│  - Cross-validation                                          │
│  - LightGBM training                                         │
│  - Metric evaluation                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Trained Model + Metrics
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  ModelStorage                                │
│  - Save .txt file                                            │
│  - Save .json metadata                                       │
│  - Version tracking                                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
                  trained_models/
                  ├── alert_scorer_v1.0.0_20250129_103045.txt
                  ├── alert_scorer_v1.0.0_20250129_103045.json
                  └── ...
```

---

## Model Metadata Schema

```json
{
  "model_type": "alert_scorer",
  "version": "1.0.0",
  "timestamp": "20250129_103045",
  "network": "ethereum",
  "training_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"
  },
  "data_stats": {
    "num_samples": 125000,
    "num_features": 47,
    "positive_rate": 0.23
  },
  "metrics": {
    "test_auc": 0.8734,
    "test_precision": 0.8421,
    "test_recall": 0.7892,
    "test_f1": 0.8148,
    "cv_auc_mean": 0.8689,
    "cv_auc_std": 0.0043
  },
  "hyperparameters": {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "min_data_in_leaf": 20
  },
  "feature_names": [
    "alert_confidence_score",
    "volume_usd",
    "severity_encoded",
    "..."
  ],
  "num_trees": 247
}
```

---

## Integration with Existing System

### ClickHouse Schema Extension

Add table for tracking trained models:

```sql
-- packages/storage/schema/trained_models.sql

CREATE TABLE IF NOT EXISTS trained_models (
    model_id String,
    model_type String,
    version String,
    network String,
    training_start_date Date,
    training_end_date Date,
    created_at DateTime,
    model_path String,
    metrics_json String,
    hyperparameters_json String,
    feature_names Array(String),
    num_samples UInt32,
    num_features UInt16,
    test_auc Float32,
    cv_auc_mean Float32,
    cv_auc_std Float32
)
ENGINE = MergeTree()
ORDER BY (network, model_type, created_at)
SETTINGS index_granularity = 8192;
```

### Update MigrateSchema

```python
# packages/storage/__init__.py

class MigrateSchema:
    def run_migrations(self):
        schemas = [
            "raw_alerts.sql",
            "raw_features.sql",
            "raw_clusters.sql",
            "raw_money_flows.sql",
            "alert_scores.sql",
            "alert_rankings.sql",
            "cluster_scores.sql",
            "batch_metadata.sql",
            "trained_models.sql",  # NEW
        ]
        
        for schema_file in schemas:
            apply_schema(self.client, schema_file)
```

---

## Error Handling

Following fail-fast principle:

```python
# No fallback values
def extract_alerts(self, start_date, end_date):
    result = self.client.query(query)
    
    if not result.result_rows:
        # Fail fast - raise exception
        raise ValueError(
            f"No alerts found for {start_date} to {end_date}"
        )
    
    return pd.DataFrame(result.result_rows)

# No default values in feature extraction
def build_features(self, data):
    if 'alerts' not in data:
        raise ValueError("Missing alerts data")
    
    if data['alerts'].empty:
        raise ValueError("Alerts dataframe is empty")
    
    # Continue with feature building...
```

---

## Logging Standards

Following ingestion conventions:

```python
# Domain-focused, no step numbers
logger.info("Extracting training data from ClickHouse")  # ✅
logger.info("Step 1: Extracting training data")          # ❌

# No emoticons
logger.success("Model training completed")  # ✅
logger.success("Model training completed 🎉")  # ❌

# Rich context in extra
logger.info(
    "Training completed",
    extra={
        "model_type": "alert_scorer",
        "test_auc": 0.8734,
        "training_time_sec": 245.3
    }
)
```

---

## Next Steps

1. **Implement Core Classes**
   - `ModelTraining` - Main orchestration
   - `FeatureExtractor` - ClickHouse data extraction
   - `ModelStorage` - Model persistence

2. **Add Schema**
   - `trained_models.sql` - Model tracking table

3. **Create Training Scripts**
   - Alert scorer training
   - Alert ranker training
   - Cluster scorer training

4. **Testing**
   - Integration with ingestion pipeline
   - End-to-end training workflow
   - Model versioning

5. **Documentation**
   - Training guide
   - API documentation
   - Example workflows

---

## Benefits of This Architecture

✅ **Consistent** - Follows ingestion patterns exactly  
✅ **Clean** - Clear separation of concerns  
✅ **Maintainable** - Easy to understand and extend  
✅ **Testable** - Each component can be tested independently  
✅ **Traceable** - Full logging and metadata tracking  
✅ **Scalable** - Can handle multiple networks and date ranges  
✅ **Production-Ready** - Fail-fast, no fallbacks, proper error handling  

---

## Comparison: Old vs New

| Aspect | Old (`alert_scoring/`) | New (`packages/training/`) |
|--------|------------------------|----------------------------|
| Data Source | File-based batches | ClickHouse queries |
| Structure | Nested modules | Flat, focused modules |
| Error Handling | Try/catch with defaults | Fail-fast with exceptions |
| Logging | Mixed approaches | Consistent loguru patterns |
| Storage | Local files | Versioned with metadata |
| Integration | Standalone | Integrated with ingestion |
| Convention | Custom | Follows ingestion pattern |
