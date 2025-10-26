# AML Miner Template - Implementation Breakdown
**Date**: 2025-10-26  
**Purpose**: Detailed breakdown of implementation tasks for aml-miner-template repository

---

## Current State Analysis

### What Exists
- Empty directory structure under `template/`:
  - `template/api/__init__.py`
  - `template/api/server.py` (empty)
  - `template/models/__init__.py`
  - `template/models/alert_scorer.py` (empty)
  - `template/models/alert_ranker.py` (empty)
  - `template/models/cluster_scorer.py` (empty)
  - `template/features/__init__.py`
  - `template/training/__init__.py`
- Empty root files:
  - `pyproject.toml` (empty)
  - `Dockerfile` (empty)
  - `README.md` (empty)

### What's Missing
Almost everything - we need to implement the entire template as specified in the architecture document.

---

## Implementation Plan

### Phase 1: Project Foundation & Configuration

#### 1.1 Directory Structure Refactoring
**Current**: `template/`  
**Target**: `aml_miner/` (per architecture spec)

**Rationale**: The architecture document specifies the main package should be `aml_miner` not `template`, as it will be importable via `pip install -e .`

**Actions**:
- Rename `template/` → `aml_miner/`
- Create missing directories:
  - `aml_miner/config/`
  - `aml_miner/utils/`
  - `trained_models/`
  - `scripts/`
  - `tests/`
  - `docs/`

#### 1.2 Package Configuration Files

**File**: `pyproject.toml`
- Build system configuration
- Project metadata (name, version, description)
- Dependencies:
  - Core: FastAPI, uvicorn, pydantic, pandas, numpy
  - ML: scikit-learn, lightgbm, shap
  - Utils: loguru, httpx, pyarrow
- Development dependencies: pytest, black, ruff, mypy
- Entry points: `aml-miner-serve`, `aml-miner-train`

**File**: `requirements.txt`
- Pinned versions of all dependencies
- Generated from pyproject.toml

**File**: `.env.example`
- Configuration template showing:
  - Model paths
  - API settings
  - Feature configurations
  - Logging levels

#### 1.3 Docker Setup

**File**: `Dockerfile`
- Multi-stage build
- Python 3.11+ base image
- Install dependencies
- Copy application code
- Expose port 8000
- Health check endpoint
- CMD to run uvicorn server

**File**: `docker-compose.yml`
- Service definition for miner API
- Volume mounts for models
- Environment variables
- Port mapping

---

### Phase 2: Core Package Implementation

#### 2.1 Package Initialization

**File**: `aml_miner/__init__.py`
```python
from aml_miner.version import __version__
```

**File**: `aml_miner/version.py`
```python
__version__ = "1.0.0"
```

#### 2.2 Configuration System

**File**: `aml_miner/config/settings.py`
- Pydantic Settings class
- Environment variable loading
- Model paths configuration
- API settings (host, port, timeout)
- Feature flags
- Validation rules

**File**: `aml_miner/config/model_config.yaml`
- LightGBM hyperparameters
- Feature selection settings
- Training configurations
- Determinism settings (seed values)

#### 2.3 Utilities

**File**: `aml_miner/utils/data_loader.py`
- Load batch data from parquet files
- Parse alerts, features, clusters, money_flows
- Data validation
- Memory-efficient loading

**File**: `aml_miner/utils/validators.py`
- Input validation functions
- Data integrity checks
- Schema validation
- Error handling

**File**: `aml_miner/utils/determinism.py`
- Seed setting for reproducibility
- Random state management
- Ensure consistent results across runs

---

### Phase 3: ML Models Implementation

#### 3.1 Base Model Class

**File**: `aml_miner/models/base_model.py`
- Abstract base class for all models
- Common interface:
  - `load_model(path)` - Load from text file
  - `save_model(path)` - Save to text file
  - `predict(X)` - Make predictions
  - `prepare_features(...)` - Feature engineering
  - `create_explanations(...)` - SHAP explanations
- Model versioning
- Logging decorators

#### 3.2 Alert Scorer Model

**File**: `aml_miner/models/alert_scorer.py`
- Extends `BaseModel`
- LightGBM classifier for binary alert scoring
- Features:
  - Alert attributes (amount, frequency, etc.)
  - Entity risk scores
  - Cluster membership features
  - Network graph features
- Output: Risk score [0, 1]
- SHAP explanations for interpretability

#### 3.3 Alert Ranker Model

**File**: `aml_miner/models/alert_ranker.py`
- Extends `BaseModel`
- Learning-to-rank model (LightGBM Ranker)
- Features: Same as scorer + relative importance
- Output: Ranked list of alerts
- Used for prioritization

#### 3.4 Cluster Scorer Model

**File**: `aml_miner/models/cluster_scorer.py`
- Extends `BaseModel`
- Scores entire alert clusters
- Features:
  - Cluster statistics (size, density)
  - Money flow patterns
  - Entity types in cluster
  - Graph centrality measures
- Output: Cluster risk score [0, 1]

---

### Phase 4: Feature Engineering

#### 4.1 Feature Builder

**File**: `aml_miner/features/feature_builder.py`
- Build features from raw data
- Alert-level features:
  - Transaction amounts (log, normalized)
  - Frequency patterns
  - Time-based features
  - Entity attributes
- Cluster-level features:
  - Graph metrics (PageRank, betweenness)
  - Money flow statistics
  - Pattern detection
- Network features:
  - In/out degree
  - Connected components
  - Community detection

#### 4.2 Feature Selector

**File**: `aml_miner/features/feature_selector.py`
- Feature importance ranking
- Correlation analysis
- Remove redundant features
- Select top-K features
- Save feature lists for consistency

---

### Phase 5: Training Pipelines

#### 5.1 Alert Scorer Training

**File**: `aml_miner/training/train_scorer.py`
- Load training data (historical SOT batches)
- Feature engineering
- Train LightGBM classifier
- Cross-validation
- Model evaluation (AUC, precision, recall)
- Save model to text file
- Logging and metrics

**CLI Interface**:
```bash
python aml_miner/training/train_scorer.py \
  --data-dir ./data \
  --output trained_models/scorer_v1.0.0.txt \
  --eval-metric auc
```

#### 5.2 Alert Ranker Training

**File**: `aml_miner/training/train_ranker.py`
- Similar to scorer training
- Use LambdaRank objective
- Optimize for NDCG metric
- Save ranker model

#### 5.3 Hyperparameter Tuning

**File**: `aml_miner/training/hyperparameter_tuner.py`
- Optuna-based tuning (or grid search)
- Define search space
- Objective function (AUC/NDCG)
- Cross-validation
- Save best parameters to YAML

---

### Phase 6: FastAPI Server

#### 6.1 API Schemas

**File**: `aml_miner/api/schemas.py`
- Pydantic models for requests/responses:
  - `BatchData` - Input batch
  - `ScoreResponse` - Single alert score
  - `RankResponse` - Ranked alerts
  - `ClusterScoreResponse` - Cluster score
  - `HealthResponse` - Health check
  - `VersionResponse` - Version info

#### 6.2 API Routes

**File**: `aml_miner/api/routes.py`
- Route handlers:
  - `POST /score/alerts` - Score all alerts
  - `POST /rank/alerts` - Rank alerts
  - `POST /score/clusters` - Score clusters
  - `GET /health` - Health check
  - `GET /version` - Version info
  - `GET /metrics` - Performance metrics

#### 6.3 Main Server

**File**: `aml_miner/api/server.py`
- FastAPI app initialization
- CORS middleware
- Startup event: Load models
- Shutdown event: Cleanup
- Error handlers
- Request logging
- Main entry point

**Features**:
- Async request handling
- Model caching (load once)
- Batch processing
- Performance metrics (latency, throughput)
- Graceful shutdown

---

### Phase 7: Scripts & Utilities

#### 7.1 Data Download Script

**File**: `scripts/download_batch.sh`
- Download SOT batch data
- Parameters: start_date, end_date
- Save to local directory
- Validate downloaded data

#### 7.2 Training Script

**File**: `scripts/train_models.py`
- Wrapper to train all models
- Download data if needed
- Train scorer, ranker, cluster_scorer
- Save all models
- Generate training report

#### 7.3 Validation Script

**File**: `scripts/validate_submission.py`
- Test API locally
- Load sample batch
- Call API endpoints
- Verify responses
- Check determinism
- Measure latency

---

### Phase 8: Testing

#### 8.1 Model Tests

**File**: `tests/test_models.py`
- Test model loading/saving
- Test predictions
- Test feature engineering
- Test explanations
- Test determinism (same input → same output)

#### 8.2 API Tests

**File**: `tests/test_api.py`
- Test all endpoints
- Test request validation
- Test error handling
- Test response formats
- Integration tests

#### 8.3 Determinism Tests

**File**: `tests/test_determinism.py`
- Critical test: Ensure reproducibility
- Same batch → same scores (every time)
- Test across restarts
- Test with different batch orders

---

### Phase 9: Documentation

#### 9.1 Quick Start Guide

**File**: `docs/quickstart.md`
- Installation steps
- Run API server
- Test with curl
- Docker deployment

#### 9.2 Training Guide

**File**: `docs/training_guide.md`
- Download training data
- Train custom models
- Hyperparameter tuning
- Model evaluation
- Best practices

#### 9.3 Customization Guide

**File**: `docs/customization.md`
- Add custom features
- Modify model architecture
- Change hyperparameters
- Extend API endpoints

#### 9.4 API Reference

**File**: `docs/api_reference.md`
- Complete API documentation
- Request/response schemas
- Examples
- Error codes

#### 9.5 README

**File**: `README.md`
- Project overview
- Quick start (5-minute setup)
- Architecture diagram
- API usage examples
- Training workflow
- License

---

## Implementation Order (Recommended)

### Sprint 1: Foundation (Day 1-2)
1. ✅ Rename `template/` → `aml_miner/`
2. ✅ Create missing directories
3. ✅ Implement `pyproject.toml`
4. ✅ Implement `.env.example`
5. ✅ Implement `requirements.txt`
6. ✅ Package init files (`__init__.py`, `version.py`)

### Sprint 2: Configuration & Utils (Day 2-3)
7. ✅ Implement `config/settings.py`
8. ✅ Implement `config/model_config.yaml`
9. ✅ Implement `utils/determinism.py`
10. ✅ Implement `utils/validators.py`
11. ✅ Implement `utils/data_loader.py`

### Sprint 3: Models (Day 3-5)
12. ✅ Implement `models/base_model.py`
13. ✅ Implement `models/alert_scorer.py`
14. ✅ Implement `models/alert_ranker.py`
15. ✅ Implement `models/cluster_scorer.py`

### Sprint 4: Features (Day 5-6)
16. ✅ Implement `features/feature_builder.py`
17. ✅ Implement `features/feature_selector.py`

### Sprint 5: API (Day 6-7)
18. ✅ Implement `api/schemas.py`
19. ✅ Implement `api/routes.py`
20. ✅ Implement `api/server.py`

### Sprint 6: Training (Day 8-9)
21. ✅ Implement `training/train_scorer.py`
22. ✅ Implement `training/train_ranker.py`
23. ✅ Implement `training/hyperparameter_tuner.py`

### Sprint 7: Scripts & Docker (Day 9-10)
24. ✅ Implement `scripts/download_batch.sh`
25. ✅ Implement `scripts/train_models.py`
26. ✅ Implement `scripts/validate_submission.py`
27. ✅ Implement `Dockerfile`
28. ✅ Implement `docker-compose.yml`

### Sprint 8: Testing & Documentation (Day 10-12)
29. ✅ Implement all tests
30. ✅ Write documentation
31. ✅ Write README.md
32. ✅ Final validation

---

## Key Design Decisions

### 1. Model Serialization Format
**Decision**: Use LightGBM's text format (`.txt`)  
**Rationale**: 
- Deterministic
- Human-readable
- Version control friendly
- Small file size
- Fast loading

### 2. Feature Engineering Approach
**Decision**: Separate feature builder from models  
**Rationale**:
- Reusable across models
- Easier to experiment
- Clear separation of concerns
- Can version features independently

### 3. API Design
**Decision**: FastAPI with async handlers  
**Rationale**:
- High performance
- Automatic OpenAPI docs
- Type safety with Pydantic
- Easy to test
- Modern Python patterns

### 4. Configuration Management
**Decision**: Pydantic Settings + YAML  
**Rationale**:
- Type-safe configuration
- Environment variable support
- Easy validation
- YAML for complex hyperparameters
- Settings for runtime config

### 5. Logging
**Decision**: Loguru  
**Rationale**:
- Simple API
- Structured logging
- No configuration needed
- Good performance
- Python 3.11+ compatible

---

## Critical Requirements

### 1. Determinism
**Requirement**: Same input MUST produce same output  
**Implementation**:
- Set all random seeds
- Use deterministic algorithms
- Test with `test_determinism.py`
- Document seed values

### 2. Performance
**Requirements**:
- Latency: < 1ms per alert
- Throughput: 1000+ alerts/second
- Memory: < 2GB for API server

**Implementation**:
- Batch processing
- Model caching
- Efficient feature computation
- Async I/O

### 3. Explainability
**Requirement**: Every score must have explanation  
**Implementation**:
- SHAP values for feature importance
- JSON format explanations
- Top-5 contributing features
- Human-readable output

---

## Testing Strategy

### Unit Tests
- Individual model methods
- Feature engineering functions
- Utility functions
- Configuration loading

### Integration Tests
- Full API endpoints
- Model loading + prediction
- Feature builder + model pipeline

### System Tests
- End-to-end batch scoring
- Docker deployment
- Performance benchmarks

### Determinism Tests
- **Most critical**
- Run same batch 100 times
- Verify identical outputs
- Test across restarts

---

## Next Steps

1. **Review this breakdown** with stakeholders
2. **Prioritize sprints** if needed
3. **Create GitHub issues** for each task (optional)
4. **Start implementation** with Sprint 1
5. **Iterate and test** continuously

---

## Success Criteria

✅ API server runs and responds to requests  
✅ Models load and make predictions  
✅ Determinism tests pass (100% consistency)  
✅ Performance meets requirements  
✅ Docker container builds and runs  
✅ Documentation is complete  
✅ All tests pass  
✅ Can train custom models  
✅ Explainability works  
✅ Ready for miners to fork and customize  

---

## Estimated Effort

**Total**: ~10-12 developer days  
**Timeline**: 2-3 weeks with testing/iteration  
**Team**: 1-2 developers

**Breakdown**:
- Foundation: 2 days
- Models & Features: 4 days
- API & Training: 3 days
- Testing & Docs: 3 days