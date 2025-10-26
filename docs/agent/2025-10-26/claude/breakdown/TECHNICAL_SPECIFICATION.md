# AML Miner Template - Technical Specification
**Date**: 2025-10-26  
**Purpose**: Detailed technical specifications for implementing aml-miner-template

---

## 1. Project Structure

### 1.1 Final Directory Tree

```
aml-miner-template/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── aml_miner/                      # Main package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── api/                        # FastAPI server
│   │   ├── __init__.py
│   │   ├── server.py               # Main application
│   │   ├── routes.py               # Route handlers
│   │   └── schemas.py              # Pydantic models
│   │
│   ├── models/                     # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py           # Abstract base
│   │   ├── alert_scorer.py         # Binary classifier
│   │   ├── alert_ranker.py         # Learning-to-rank
│   │   └── cluster_scorer.py       # Cluster scorer
│   │
│   ├── features/                   # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_builder.py      # Build features
│   │   └── feature_selector.py     # Select features
│   │
│   ├── training/                   # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_scorer.py         # Train alert scorer
│   │   ├── train_ranker.py         # Train ranker
│   │   └── hyperparameter_tuner.py # HPO
│   │
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load batch data
│   │   ├── validators.py           # Validation
│   │   └── determinism.py          # Reproducibility
│   │
│   └── config/                     # Configuration
│       ├── __init__.py
│       ├── settings.py             # Settings class
│       └── model_config.yaml       # Hyperparameters
│
├── trained_models/                 # Pretrained models
│   ├── alert_scorer_v1.0.0.txt
│   ├── alert_ranker_v1.0.0.txt
│   ├── cluster_scorer_v1.0.0.txt
│   └── model_metadata.json
│
├── scripts/                        # Utility scripts
│   ├── download_batch.sh
│   ├── train_models.py
│   └── validate_submission.py
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_features.py
│   └── test_determinism.py
│
└── docs/                           # Documentation
    ├── quickstart.md
    ├── training_guide.md
    ├── customization.md
    └── api_reference.md
```

---

## 2. Data Structures

### 2.1 Batch Data Schema

#### Alerts DataFrame
From [`analyzers_alerts`](../../../sot/schema/analyzers_alerts.sql:1) schema:
```python
{
    # Time series dimensions
    "window_days": int,           # Window size in days
    "processing_date": date,      # Date of processing
    
    # Primary identifiers
    "alert_id": str,              # Unique alert identifier
    "address": str,               # Address being alerted on
    
    # Alert classification
    "typology_type": str,         # AML typology classification
    
    # Pattern reference (if pattern-based alert)
    "pattern_id": str,            # Associated pattern ID
    "pattern_type": str,          # Type of pattern
    
    # Alert details
    "severity": str,              # "low", "medium", "high", "critical"
    "suspected_address_type": str,     # Address classification
    "suspected_address_subtype": str,  # Address subtype
    "alert_confidence_score": float,   # Confidence [0-1]
    "description": str,           # Alert description
    "volume_usd": Decimal,        # Transaction volume in USD
    
    # Evidence and context
    "evidence_json": str,         # Detailed evidence (JSON string)
    "risk_indicators": List[str], # Risk indicators triggered
    
    # Version tracking
    "_version": int               # For deduplication
}
```

#### Features DataFrame
```python
{
    "alert_id": str,              # Links to alerts
    "feature_name": str,          # Name of feature
    "feature_value": float,       # Value of feature
    "feature_type": str           # Type: "numeric", "categorical", etc.
}
```

#### Clusters DataFrame
```python
{
    "cluster_id": str,            # Unique cluster identifier
    "entity_ids": List[str],      # Entities in cluster
    "alert_ids": List[str],       # Alerts in cluster
    "cluster_size": int,          # Number of entities
    "density": float,             # Graph density
    "total_flow": float,          # Total money flow
    "pattern_type": str           # Detected pattern type
}
```

#### Money Flows DataFrame
```python
{
    "flow_id": str,               # Unique flow identifier
    "source_id": str,             # Source entity
    "target_id": str,             # Target entity
    "amount": float,              # Flow amount
    "timestamp": datetime,        # Flow time
    "cluster_id": str,            # Associated cluster
    "hop_distance": int           # Distance in graph
}
```

---

## 3. Model Specifications

### 3.1 Alert Scorer (Binary Classifier)

**Algorithm**: LightGBM Binary Classifier

**Input Features** (example set):
- Alert attributes: amount, frequency, duration
- Entity features: risk_score, entity_type, country
- Network features: in_degree, out_degree, pagerank
- Cluster features: cluster_size, density, pattern_match
- Temporal features: hour, day_of_week, time_since_last
- Statistical features: amount_z_score, frequency_percentile

**Output**:
- Risk score: float in [0, 1]
- Explanation: JSON with top-5 features

**Hyperparameters** (initial):
```yaml
num_leaves: 31
max_depth: -1
learning_rate: 0.05
n_estimators: 100
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 0.1
random_state: 42
objective: binary
metric: auc
```

**Training Data Requirements**:
- Minimum: 10,000 labeled alerts
- Positive class ratio: 1-10%
- Features: ~50-100 features

**Performance Targets**:
- Inference: < 0.5ms per alert
- AUC-ROC: > 0.85 on validation
- Memory: < 100MB model size

### 3.2 Alert Ranker (Learning-to-Rank)

**Algorithm**: LightGBM Ranker (LambdaRank)

**Input Features**: Same as Alert Scorer + relative features
- Pairwise comparisons
- Within-batch rankings

**Output**:
- Ranked list of alert IDs
- Relevance scores

**Hyperparameters**:
```yaml
num_leaves: 31
max_depth: -1
learning_rate: 0.05
n_estimators: 100
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
objective: lambdarank
metric: ndcg
ndcg_eval_at: [5, 10, 20]
```

**Training Data Requirements**:
- Query groups (batches)
- Relevance labels (0-4 scale)
- Minimum: 100 query groups

**Performance Targets**:
- NDCG@10: > 0.80
- Inference: < 1ms per batch

### 3.3 Cluster Scorer

**Algorithm**: LightGBM Binary Classifier

**Input Features**:
- Cluster statistics: size, density, diameter
- Money flow: total_in, total_out, net_flow
- Graph metrics: modularity, clustering_coefficient
- Pattern features: layering_score, circular_score
- Temporal: time_span, burst_indicator

**Output**:
- Cluster risk score: float in [0, 1]

**Hyperparameters**: Similar to Alert Scorer

**Performance Targets**:
- Inference: < 2ms per cluster
- AUC-ROC: > 0.80

---

## 4. API Specifications

### 4.1 Endpoints

#### POST /score/alerts

**Request**:
```json
{
  "alerts": [
    {
      "alert_id": "alert_001",
      "entity_id": "entity_123",
      "alert_type": "high_volume",
      "timestamp": "2025-10-26T10:00:00Z",
      "amount": 50000.0,
      "frequency": 15,
      "cluster_id": "cluster_abc"
    }
  ],
  "features": [
    {
      "alert_id": "alert_001",
      "feature_name": "amount_z_score",
      "feature_value": 2.5
    }
  ],
  "clusters": [
    {
      "cluster_id": "cluster_abc",
      "entity_ids": ["entity_123", "entity_456"],
      "cluster_size": 2,
      "density": 0.8
    }
  ],
  "money_flows": [
    {
      "flow_id": "flow_001",
      "source_id": "entity_123",
      "target_id": "entity_456",
      "amount": 10000.0
    }
  ]
}
```

**Response**:
```json
[
  {
    "alert_id": "alert_001",
    "score": 0.87,
    "model_version": "v1.0.0",
    "latency_ms": 0.15,
    "explain_json": "{\"top_features\": [{\"name\": \"amount_z_score\", \"contribution\": 0.35}, ...]}"
  }
]
```

**Performance**:
- Latency: < 1ms per alert
- Throughput: 1000+ alerts/second
- Timeout: 120 seconds

#### GET /health

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": {
    "alert_scorer": true,
    "alert_ranker": true,
    "cluster_scorer": true
  },
  "uptime_seconds": 3600,
  "memory_mb": 1024
}
```

#### GET /version

**Response**:
```json
{
  "api_version": "1.0.0",
  "alert_scorer_version": "v1.0.0",
  "alert_ranker_version": "v1.0.0",
  "cluster_scorer_version": "v1.0.0",
  "python_version": "3.11.5",
  "lightgbm_version": "4.3.0"
}
```

---

## 5. Configuration System

### 5.1 Settings Class

**File**: `aml_miner/config/settings.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TRAINED_MODELS_DIR: Path = BASE_DIR / "trained_models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model paths
    ALERT_SCORER_PATH: Path = TRAINED_MODELS_DIR / "alert_scorer_v1.0.0.txt"
    ALERT_RANKER_PATH: Path = TRAINED_MODELS_DIR / "alert_ranker_v1.0.0.txt"
    CLUSTER_SCORER_PATH: Path = TRAINED_MODELS_DIR / "cluster_scorer_v1.0.0.txt"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 120
    
    # Feature settings
    MAX_FEATURES: int = 100
    FEATURE_SELECTION_METHOD: str = "importance"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "logs" / "api.log"
    
    # Determinism
    RANDOM_SEED: int = 42
    
    # Performance
    BATCH_SIZE: int = 1000
    MAX_MEMORY_MB: int = 2048
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 5.2 Model Config YAML

**File**: `aml_miner/config/model_config.yaml`

```yaml
alert_scorer:
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.05
  n_estimators: 100
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
  random_state: 42
  objective: binary
  metric: auc
  verbosity: -1
  n_jobs: -1

alert_ranker:
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.05
  n_estimators: 100
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  objective: lambdarank
  metric: ndcg
  ndcg_eval_at: [5, 10, 20]
  verbosity: -1
  n_jobs: -1

cluster_scorer:
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.05
  n_estimators: 100
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
  random_state: 42
  objective: binary
  metric: auc
  verbosity: -1
  n_jobs: -1

feature_engineering:
  temporal_features:
    - hour_of_day
    - day_of_week
    - is_weekend
    - time_since_last_alert
  
  amount_features:
    - amount_raw
    - amount_log
    - amount_z_score
    - amount_percentile
  
  network_features:
    - in_degree
    - out_degree
    - pagerank
    - betweenness_centrality
  
  cluster_features:
    - cluster_size
    - cluster_density
    - pattern_match_score

training:
  validation_split: 0.2
  test_split: 0.1
  cv_folds: 5
  early_stopping_rounds: 50
  min_training_samples: 10000
```

---

## 6. Feature Engineering Details

### 6.1 Feature Categories

**Alert-Level Features** (~30 features):
```python
{
    # Raw attributes
    "amount": float,
    "frequency": int,
    "duration_hours": float,
    
    # Normalized
    "amount_log": float,
    "amount_z_score": float,
    "amount_percentile": float,
    
    # Temporal
    "hour": int,
    "day_of_week": int,
    "is_weekend": bool,
    "is_business_hours": bool,
    
    # Entity
    "entity_risk_score": float,
    "entity_age_days": float,
    "entity_type_encoded": int,
    "entity_country_risk": float
}
```

**Network Features** (~20 features):
```python
{
    # Graph metrics
    "in_degree": int,
    "out_degree": int,
    "total_degree": int,
    "pagerank": float,
    "betweenness_centrality": float,
    "closeness_centrality": float,
    
    # Flow metrics
    "total_inflow": float,
    "total_outflow": float,
    "net_flow": float,
    "flow_balance": float,
    
    # Structural
    "connected_components": int,
    "local_clustering_coefficient": float
}
```

**Cluster Features** (~15 features):
```python
{
    "cluster_size": int,
    "cluster_density": float,
    "cluster_diameter": int,
    "modularity": float,
    "total_flow": float,
    "avg_flow": float,
    "pattern_type_encoded": int,
    "layering_score": float,
    "circular_score": float
}
```

**Statistical Aggregations** (~15 features):
```python
{
    # Aggregates over time windows
    "amount_mean_7d": float,
    "amount_std_7d": float,
    "frequency_sum_7d": int,
    "frequency_max_1d": int,
    
    # Percentiles
    "amount_pct_rank": float,
    "frequency_pct_rank": float
}
```

### 6.2 Feature Builder Implementation

Key functions:
- `build_alert_features(alerts_df, features_df) -> pd.DataFrame`
- `build_network_features(money_flows_df) -> pd.DataFrame`
- `build_cluster_features(clusters_df) -> pd.DataFrame`
- `build_temporal_features(alerts_df) -> pd.DataFrame`
- `build_statistical_features(alerts_df, window='7d') -> pd.DataFrame`

---

## 7. Training Pipeline

### 7.1 Data Preparation

```python
def prepare_training_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    # Load batch files
    batches = load_batch_files(data_dir)
    
    # Concatenate all batches
    all_alerts = pd.concat([b['alerts'] for b in batches])
    all_features = pd.concat([b['features'] for b in batches])
    all_clusters = pd.concat([b['clusters'] for b in batches])
    
    # Build features
    X = build_all_features(all_alerts, all_features, all_clusters)
    
    # Extract labels
    y = all_alerts['ground_truth']
    
    # Validate
    assert len(X) == len(y)
    assert y.isnull().sum() == 0
    
    return X, y
```

### 7.2 Training Loop

```python
def train_alert_scorer(X, y, config):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create dataset
    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)
    
    # Train
    model = lgb.train(
        config,
        train_data,
        valid_sets=[train_data, test_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(10)
        ]
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    
    logger.info(f"Test AUC: {auc:.4f}")
    
    # Save
    model.save_model("alert_scorer_v1.0.0.txt")
    
    return model
```

---

## 8. Determinism Requirements

### 8.1 Sources of Randomness

Must set seeds for:
- Python random
- NumPy random
- LightGBM random_state
- Hash-based operations (use sorted keys)
- Parallel operations (use n_jobs deterministically)

### 8.2 Implementation

```python
def set_deterministic_mode(seed: int = 42):
    """Set all random seeds for reproducibility"""
    import random
    import numpy as np
    import lightgbm as lgb
    
    random.seed(seed)
    np.random.seed(seed)
    
    # LightGBM uses random_state parameter
    # Set in config
    
    # Ensure single-threaded for determinism
    # (or use deterministic multi-threading)
    
    logger.info(f"Deterministic mode enabled (seed={seed})")
```

### 8.3 Validation

Test that must pass:
```python
def test_determinism():
    # Load same batch twice
    batch1 = load_batch("test_batch.parquet")
    batch2 = load_batch("test_batch.parquet")
    
    # Score twice
    scores1 = scorer.predict(batch1)
    scores2 = scorer.predict(batch2)
    
    # Must be identical
    np.testing.assert_array_equal(scores1, scores2)
```

---

## 9. Performance Optimization

### 9.1 Model Loading
- Load models once at startup
- Cache in memory
- Use global variables or singletons

### 9.2 Batch Processing
- Process multiple alerts at once
- Use vectorized operations (pandas/numpy)
- Minimize Python loops

### 9.3 Feature Computation
- Pre-compute expensive features
- Cache intermediate results
- Use efficient data structures (numpy arrays)

### 9.4 API Optimization
- Async handlers
- Connection pooling
- Response streaming for large batches

---

## 10. Error Handling

### 10.1 Error Types

```python
class MinerError(Exception):
    """Base exception for miner errors"""
    pass

class ModelNotLoadedError(MinerError):
    """Model not loaded"""
    pass

class InvalidBatchError(MinerError):
    """Invalid batch data"""
    pass

class FeatureComputationError(MinerError):
    """Error computing features"""
    pass

class PredictionError(MinerError):
    """Error making predictions"""
    pass
```

### 10.2 Logging Strategy

Use structured logging:
```python
logger.info("Scoring alerts", alert_count=len(alerts), batch_id=batch_id)
logger.error("Model prediction failed", error=str(e), alert_id=alert_id)
```

---

## 11. Docker Deployment

### 11.1 Dockerfile Strategy

Multi-stage build:
1. Build stage: Install dependencies
2. Runtime stage: Copy only needed files
3. Health check: Ping /health endpoint
4. Non-root user: Run as unprivileged user

### 11.2 Environment Variables

```bash
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
RANDOM_SEED=42
```

---

## 12. Testing Strategy

### 12.1 Test Coverage Targets

- Unit tests: > 80% coverage
- Integration tests: All API endpoints
- Determinism tests: 100% pass rate
- Performance tests: Meet latency targets

### 12.2 Fixtures

```python
@pytest.fixture
def sample_batch():
    return {
        "alerts": pd.DataFrame([...]),
        "features": pd.DataFrame([...]),
        "clusters": pd.DataFrame([...])
    }

@pytest.fixture
def trained_model():
    model = AlertScorerModel()
    model.load_model("trained_models/alert_scorer_v1.0.0.txt")
    return model
```

---

## Summary

This specification provides:
- ✅ Complete directory structure
- ✅ Data schemas
- ✅ Model specifications
- ✅ API contracts
- ✅ Configuration system
- ✅ Feature engineering details
- ✅ Training pipeline
- ✅ Determinism requirements
- ✅ Performance targets
- ✅ Error handling
- ✅ Testing strategy

Ready for implementation.