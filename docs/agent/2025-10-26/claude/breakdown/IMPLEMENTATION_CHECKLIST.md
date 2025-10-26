# AML Miner Template - Implementation Checklist
**Date**: 2025-10-26  
**Purpose**: Track implementation progress step-by-step

---

## Phase 1: Foundation & Setup ⏳

### 1.1 Directory Restructuring
- [x] Rename `template/` → `aml_miner/`
- [x] Create `aml_miner/config/` directory
- [x] Create `aml_miner/utils/` directory
- [x] Create `trained_models/` directory
- [x] Create `scripts/` directory
- [x] Create `tests/` directory
- [x] Create `docs/` directory
- [x] Create `logs/` directory

### 1.2 Package Initialization
- [x] Implement `aml_miner/__init__.py`
- [x] Implement `aml_miner/version.py`

### 1.3 Configuration Files
- [x] Implement `pyproject.toml` (complete with all dependencies)
- [x] Generate `requirements.txt` from pyproject.toml
- [x] Create `.env.example` template
- [x] Update `.gitignore` (add logs/, .env, __pycache__, etc.)

---

## Phase 2: Configuration & Utilities ✅

### 2.1 Configuration System
- [x] Implement `aml_miner/config/__init__.py`
- [x] Implement `aml_miner/config/settings.py` (Pydantic Settings)
- [x] Implement `aml_miner/config/model_config.yaml` (hyperparameters)

### 2.2 Utilities
- [x] Implement `aml_miner/utils/__init__.py`
- [x] Implement `aml_miner/utils/determinism.py`
  - [x] `set_deterministic_mode()` function
  - [x] Seed setting for random, numpy, etc.
- [x] Implement `aml_miner/utils/validators.py`
  - [x] Input validation functions
  - [x] Schema validators
  - [x] Error classes
- [x] Implement `aml_miner/utils/data_loader.py`
  - [x] `load_batch()` function
  - [x] Parquet file reading
  - [x] Data validation

---

## Phase 3: ML Models ✅

### 3.1 Base Model
- [x] Implement `aml_miner/models/__init__.py`
- [x] Implement `aml_miner/models/base_model.py`
  - [x] `BaseModel` abstract class
  - [x] `load_model()` method
  - [x] `save_model()` method
  - [x] `predict()` abstract method
  - [x] `prepare_features()` abstract method
  - [x] `create_explanations()` method (SHAP)
  - [x] Model versioning logic
  - [x] Logging decorators

### 3.2 Alert Scorer
- [x] Implement `aml_miner/models/alert_scorer.py`
  - [x] `AlertScorerModel` class (extends BaseModel)
  - [x] `prepare_features()` implementation
  - [x] `predict()` implementation
  - [x] `create_explanations()` implementation
  - [x] Feature names mapping
  - [x] Validation logic

### 3.3 Alert Ranker
- [x] Implement `aml_miner/models/alert_ranker.py`
  - [x] `AlertRankerModel` class (extends BaseModel)
  - [x] `prepare_features()` implementation
  - [x] `rank_alerts()` method
  - [x] `predict()` implementation

### 3.4 Cluster Scorer
- [x] Implement `aml_miner/models/cluster_scorer.py`
  - [x] `ClusterScorerModel` class (extends BaseModel)
  - [x] `prepare_features()` implementation
  - [x] `predict()` implementation
  - [x] Cluster-specific features

---

## Phase 4: Feature Engineering ✅

### 4.1 Feature Builder
- [x] Implement `aml_miner/features/__init__.py`
- [x] Implement `aml_miner/features/feature_builder.py`
  - [x] `build_alert_features()` function
  - [x] `build_network_features()` function
  - [x] `build_cluster_features()` function
  - [x] `build_temporal_features()` function
  - [x] `build_statistical_features()` function
  - [x] `build_all_features()` orchestrator
  - [x] Feature name standardization

### 4.2 Feature Selector
- [x] Implement `aml_miner/features/feature_selector.py`
  - [x] `FeatureSelector` class
  - [x] Importance-based selection
  - [x] Correlation analysis
  - [x] `select_features()` method
  - [x] `save_selected_features()` method

---

## Phase 5: FastAPI Server ✅

### 5.1 API Schemas
- [x] Implement `aml_miner/api/__init__.py`
- [x] Implement `aml_miner/api/schemas.py`
  - [x] `BatchData` Pydantic model
  - [x] `AlertData` Pydantic model
  - [x] `FeatureData` Pydantic model
  - [x] `ClusterData` Pydantic model
  - [x] `MoneyFlowData` Pydantic model
  - [x] `ScoreResponse` Pydantic model
  - [x] `RankResponse` Pydantic model
  - [x] `ClusterScoreResponse` Pydantic model
  - [x] `HealthResponse` Pydantic model
  - [x] `VersionResponse` Pydantic model

### 5.2 API Routes
- [x] Implement `aml_miner/api/routes.py`
  - [x] `score_alerts()` handler
  - [x] `rank_alerts()` handler
  - [x] `score_clusters()` handler
  - [x] `health_check()` handler
  - [x] `get_version()` handler
  - [x] `get_metrics()` handler (optional)
  - [x] Error handling
  - [x] Request logging

### 5.3 Main Server
- [x] Implement `aml_miner/api/server.py`
  - [x] FastAPI app initialization
  - [x] CORS middleware
  - [x] `startup_event()` - load models
  - [x] `shutdown_event()` - cleanup
  - [x] Global exception handlers
  - [x] Request/response logging
  - [x] `main()` entry point
  - [x] Uvicorn configuration

---

## Phase 6: Training Pipelines ✅

### 6.1 Alert Scorer Training
- [x] Implement `aml_miner/training/__init__.py`
- [x] Implement `aml_miner/training/train_scorer.py`
  - [x] `prepare_training_data()` function
  - [x] `train_alert_scorer()` function
  - [x] Cross-validation logic
  - [x] Model evaluation (AUC, precision, recall)
  - [x] Model saving
  - [x] CLI interface (argparse)
  - [x] `main()` entry point
  - [x] Logging & metrics

### 6.2 Alert Ranker Training
- [x] Implement `aml_miner/training/train_ranker.py`
  - [x] `prepare_ranking_data()` function
  - [x] `train_alert_ranker()` function
  - [x] Query group creation
  - [x] NDCG evaluation
  - [x] Model saving
  - [x] CLI interface

### 6.3 Hyperparameter Tuning
- [x] Implement `aml_miner/training/hyperparameter_tuner.py`
  - [x] `HyperparameterTuner` class
  - [x] Optuna integration (or grid search)
  - [x] Search space definition
  - [x] Objective function
  - [x] Cross-validation
  - [x] Save best params to YAML
  - [x] CLI interface

---

## Phase 7: Scripts & Utilities ✅

### 7.1 Data Scripts
- [x] Implement `scripts/download_batch.sh`
  - [x] Download SOT batch data
  - [x] Command-line arguments (start_date, end_date)
  - [x] Validation
  - [x] Error handling

### 7.2 Training Scripts
- [x] Implement `scripts/train_models.py`
  - [x] Orchestrate all training
  - [x] Download data if needed
  - [x] Train all models
  - [x] Generate training report
  - [x] CLI interface

### 7.3 Validation Scripts
- [x] Implement `scripts/validate_submission.py`
  - [x] Test API locally
  - [x] Load sample batch
  - [x] Call all endpoints
  - [x] Verify responses
  - [x] Check determinism
  - [x] Measure latency
  - [x] Generate validation report

---

## Phase 8: Docker & Deployment ✅

### 8.1 Docker Configuration
- [x] Implement `Dockerfile`
  - [x] Multi-stage build
  - [x] Python 3.13 base
  - [x] Install dependencies
  - [x] Copy application
  - [x] Non-root user
  - [x] Health check
  - [x] CMD uvicorn

### 8.2 Docker Compose
- [x] Implement `docker-compose.yml`
  - [x] API service definition
  - [x] Volume mounts
  - [x] Environment variables
  - [x] Port mapping
  - [x] Health checks
  - [x] Restart policy

---

## Phase 9: Testing ⏳

### 9.1 Test Infrastructure
- [ ] Implement `tests/__init__.py`
- [ ] Implement `tests/conftest.py`
  - [ ] Pytest fixtures
  - [ ] Sample data fixtures
  - [ ] Model fixtures
  - [ ] API client fixtures

### 9.2 Model Tests
- [ ] Implement `tests/test_models.py`
  - [ ] Test BaseModel
  - [ ] Test AlertScorerModel
  - [ ] Test AlertRankerModel
  - [ ] Test ClusterScorerModel
  - [ ] Test model loading/saving
  - [ ] Test predictions
  - [ ] Test explanations

### 9.3 Feature Tests
- [ ] Implement `tests/test_features.py`
  - [ ] Test feature_builder functions
  - [ ] Test feature_selector
  - [ ] Test feature validation
  - [ ] Test edge cases

### 9.4 API Tests
- [ ] Implement `tests/test_api.py`
  - [ ] Test /score/alerts endpoint
  - [ ] Test /rank/alerts endpoint
  - [ ] Test /score/clusters endpoint
  - [ ] Test /health endpoint
  - [ ] Test /version endpoint
  - [ ] Test error handling
  - [ ] Test request validation
  - [ ] Integration tests

### 9.5 Determinism Tests
- [ ] Implement `tests/test_determinism.py`
  - [ ] Test same input → same output
  - [ ] Test across restarts
  - [ ] Test batch order independence
  - [ ] 100 iterations test
  - [ ] Critical test - must pass

---

## Phase 10: Documentation ✅

### 10.1 User Documentation
- [x] Write `docs/quickstart.md`
  - [x] Installation instructions
  - [x] Quick start (5 minutes)
  - [x] Run API server
  - [x] Test with curl
  - [x] Docker deployment

- [x] Write `docs/training_guide.md`
  - [x] Download training data
  - [x] Train custom models
  - [x] Hyperparameter tuning
  - [x] Model evaluation
  - [x] Best practices

- [x] Write `docs/customization.md`
  - [x] Add custom features
  - [x] Modify model architecture
  - [x] Change hyperparameters
  - [x] Extend API endpoints
  - [x] Advanced techniques

- [x] Write `docs/api_reference.md`
  - [x] Complete API documentation
  - [x] All endpoints
  - [x] Request/response schemas
  - [x] Examples
  - [x] Error codes

### 10.2 Main README
- [x] Write `README.md`
  - [x] Project overview
  - [x] Architecture diagram
  - [x] Quick start (copy from docs)
  - [x] API usage examples
  - [x] Training workflow
  - [x] Contributing guidelines
  - [x] License

### 10.3 Additional Docs
- [x] Create `LICENSE` file (MIT)
- [ ] Create `CONTRIBUTING.md` (optional)
- [ ] Create `CHANGELOG.md` (optional)

---

## Phase 11: Pretrained Models ⏳

### 11.1 Model Training
- [ ] Train initial alert_scorer model
- [ ] Train initial alert_ranker model
- [ ] Train initial cluster_scorer model
- [ ] Validate all models

### 11.2 Model Files
- [ ] Save `trained_models/alert_scorer_v1.0.0.txt`
- [ ] Save `trained_models/alert_ranker_v1.0.0.txt`
- [ ] Save `trained_models/cluster_scorer_v1.0.0.txt`
- [ ] Create `trained_models/model_metadata.json`
  - [ ] Model versions
  - [ ] Training date
  - [ ] Performance metrics
  - [ ] Feature lists

---

## Phase 12: Final Validation ⏳

### 12.1 Integration Testing
- [ ] Full end-to-end test
- [ ] API server startup
- [ ] Load models
- [ ] Process sample batch
- [ ] Verify responses
- [ ] Check performance

### 12.2 Performance Testing
- [ ] Latency benchmarks (< 1ms per alert)
- [ ] Throughput test (1000+ alerts/sec)
- [ ] Memory usage (< 2GB)
- [ ] Load testing

### 12.3 Determinism Validation
- [ ] Run determinism test 100 times
- [ ] 100% pass rate required
- [ ] Document results

### 12.4 Docker Validation
- [ ] Build Docker image
- [ ] Run container
- [ ] Test API endpoints
- [ ] Check health
- [ ] Verify logs

### 12.5 Documentation Review
- [ ] Review all docs for accuracy
- [ ] Test all code examples
- [ ] Check links
- [ ] Proofread

---

## Completion Criteria ✅

All items must be checked before release:

- [ ] All code implemented and tested
- [ ] All tests passing (100% pass rate)
- [ ] Determinism test passes 100 times
- [ ] Performance targets met
- [ ] Docker image builds and runs
- [ ] Documentation complete
- [ ] README has clear quick start
- [ ] Pretrained models available
- [ ] License file present
- [ ] Ready for miners to fork

---

## Progress Summary

**Phase 1**: ✅ 13/13 (100%)
**Phase 2**: ✅ 11/11 (100%)
**Phase 3**: ✅ 15/15 (100%)
**Phase 4**: ✅ 12/12 (100%)
**Phase 5**: ✅ 21/21 (100%)
**Phase 6**: ✅ 17/17 (100%)
**Phase 7**: ✅ 15/15 (100%)
**Phase 8**: ✅ 13/13 (100%)
**Phase 9**: ⬜ 0/23 (0%)
**Phase 10**: ✅ 13/14 (93%) - Core documentation complete
**Phase 11**: ⬜ 0/7 (0%)
**Phase 12**: ⬜ 0/15 (0%)

**Overall**: ✅ 130/176 tasks (73.9%)

---

## Notes

- Update progress as tasks are completed
- Mark with ✅ when phase is complete
- Add notes for any blockers or issues
- Track time spent on each phase
- Document any deviations from plan