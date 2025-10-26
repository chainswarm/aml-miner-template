# AML Miner Template - Implementation Checklist
**Date**: 2025-10-26  
**Purpose**: Track implementation progress step-by-step

---

## Phase 1: Foundation & Setup ⏳

### 1.1 Directory Restructuring
- [ ] Rename `template/` → `aml_miner/`
- [ ] Create `aml_miner/config/` directory
- [ ] Create `aml_miner/utils/` directory
- [ ] Create `trained_models/` directory
- [ ] Create `scripts/` directory
- [ ] Create `tests/` directory
- [ ] Create `docs/` directory
- [ ] Create `logs/` directory

### 1.2 Package Initialization
- [ ] Implement `aml_miner/__init__.py`
- [ ] Implement `aml_miner/version.py`

### 1.3 Configuration Files
- [ ] Implement `pyproject.toml` (complete with all dependencies)
- [ ] Generate `requirements.txt` from pyproject.toml
- [ ] Create `.env.example` template
- [ ] Update `.gitignore` (add logs/, .env, __pycache__, etc.)

---

## Phase 2: Configuration & Utilities ⏳

### 2.1 Configuration System
- [ ] Implement `aml_miner/config/__init__.py`
- [ ] Implement `aml_miner/config/settings.py` (Pydantic Settings)
- [ ] Implement `aml_miner/config/model_config.yaml` (hyperparameters)

### 2.2 Utilities
- [ ] Implement `aml_miner/utils/__init__.py`
- [ ] Implement `aml_miner/utils/determinism.py`
  - [ ] `set_deterministic_mode()` function
  - [ ] Seed setting for random, numpy, etc.
- [ ] Implement `aml_miner/utils/validators.py`
  - [ ] Input validation functions
  - [ ] Schema validators
  - [ ] Error classes
- [ ] Implement `aml_miner/utils/data_loader.py`
  - [ ] `load_batch()` function
  - [ ] Parquet file reading
  - [ ] Data validation

---

## Phase 3: ML Models ⏳

### 3.1 Base Model
- [ ] Implement `aml_miner/models/__init__.py`
- [ ] Implement `aml_miner/models/base_model.py`
  - [ ] `BaseModel` abstract class
  - [ ] `load_model()` method
  - [ ] `save_model()` method
  - [ ] `predict()` abstract method
  - [ ] `prepare_features()` abstract method
  - [ ] `create_explanations()` method (SHAP)
  - [ ] Model versioning logic
  - [ ] Logging decorators

### 3.2 Alert Scorer
- [ ] Implement `aml_miner/models/alert_scorer.py`
  - [ ] `AlertScorerModel` class (extends BaseModel)
  - [ ] `prepare_features()` implementation
  - [ ] `predict()` implementation
  - [ ] `create_explanations()` implementation
  - [ ] Feature names mapping
  - [ ] Validation logic

### 3.3 Alert Ranker
- [ ] Implement `aml_miner/models/alert_ranker.py`
  - [ ] `AlertRankerModel` class (extends BaseModel)
  - [ ] `prepare_features()` implementation
  - [ ] `rank_alerts()` method
  - [ ] `predict()` implementation

### 3.4 Cluster Scorer
- [ ] Implement `aml_miner/models/cluster_scorer.py`
  - [ ] `ClusterScorerModel` class (extends BaseModel)
  - [ ] `prepare_features()` implementation
  - [ ] `predict()` implementation
  - [ ] Cluster-specific features

---

## Phase 4: Feature Engineering ⏳

### 4.1 Feature Builder
- [ ] Implement `aml_miner/features/__init__.py`
- [ ] Implement `aml_miner/features/feature_builder.py`
  - [ ] `build_alert_features()` function
  - [ ] `build_network_features()` function
  - [ ] `build_cluster_features()` function
  - [ ] `build_temporal_features()` function
  - [ ] `build_statistical_features()` function
  - [ ] `build_all_features()` orchestrator
  - [ ] Feature name standardization

### 4.2 Feature Selector
- [ ] Implement `aml_miner/features/feature_selector.py`
  - [ ] `FeatureSelector` class
  - [ ] Importance-based selection
  - [ ] Correlation analysis
  - [ ] `select_features()` method
  - [ ] `save_selected_features()` method

---

## Phase 5: FastAPI Server ⏳

### 5.1 API Schemas
- [ ] Implement `aml_miner/api/__init__.py`
- [ ] Implement `aml_miner/api/schemas.py`
  - [ ] `BatchData` Pydantic model
  - [ ] `AlertData` Pydantic model
  - [ ] `FeatureData` Pydantic model
  - [ ] `ClusterData` Pydantic model
  - [ ] `MoneyFlowData` Pydantic model
  - [ ] `ScoreResponse` Pydantic model
  - [ ] `RankResponse` Pydantic model
  - [ ] `ClusterScoreResponse` Pydantic model
  - [ ] `HealthResponse` Pydantic model
  - [ ] `VersionResponse` Pydantic model

### 5.2 API Routes
- [ ] Implement `aml_miner/api/routes.py`
  - [ ] `score_alerts()` handler
  - [ ] `rank_alerts()` handler
  - [ ] `score_clusters()` handler
  - [ ] `health_check()` handler
  - [ ] `get_version()` handler
  - [ ] `get_metrics()` handler (optional)
  - [ ] Error handling
  - [ ] Request logging

### 5.3 Main Server
- [ ] Implement `aml_miner/api/server.py`
  - [ ] FastAPI app initialization
  - [ ] CORS middleware
  - [ ] `startup_event()` - load models
  - [ ] `shutdown_event()` - cleanup
  - [ ] Global exception handlers
  - [ ] Request/response logging
  - [ ] `main()` entry point
  - [ ] Uvicorn configuration

---

## Phase 6: Training Pipelines ⏳

### 6.1 Alert Scorer Training
- [ ] Implement `aml_miner/training/__init__.py`
- [ ] Implement `aml_miner/training/train_scorer.py`
  - [ ] `prepare_training_data()` function
  - [ ] `train_alert_scorer()` function
  - [ ] Cross-validation logic
  - [ ] Model evaluation (AUC, precision, recall)
  - [ ] Model saving
  - [ ] CLI interface (argparse)
  - [ ] `main()` entry point
  - [ ] Logging & metrics

### 6.2 Alert Ranker Training
- [ ] Implement `aml_miner/training/train_ranker.py`
  - [ ] `prepare_ranking_data()` function
  - [ ] `train_alert_ranker()` function
  - [ ] Query group creation
  - [ ] NDCG evaluation
  - [ ] Model saving
  - [ ] CLI interface

### 6.3 Hyperparameter Tuning
- [ ] Implement `aml_miner/training/hyperparameter_tuner.py`
  - [ ] `HyperparameterTuner` class
  - [ ] Optuna integration (or grid search)
  - [ ] Search space definition
  - [ ] Objective function
  - [ ] Cross-validation
  - [ ] Save best params to YAML
  - [ ] CLI interface

---

## Phase 7: Scripts & Utilities ⏳

### 7.1 Data Scripts
- [ ] Implement `scripts/download_batch.sh`
  - [ ] Download SOT batch data
  - [ ] Command-line arguments (start_date, end_date)
  - [ ] Validation
  - [ ] Error handling

### 7.2 Training Scripts
- [ ] Implement `scripts/train_models.py`
  - [ ] Orchestrate all training
  - [ ] Download data if needed
  - [ ] Train all models
  - [ ] Generate training report
  - [ ] CLI interface

### 7.3 Validation Scripts
- [ ] Implement `scripts/validate_submission.py`
  - [ ] Test API locally
  - [ ] Load sample batch
  - [ ] Call all endpoints
  - [ ] Verify responses
  - [ ] Check determinism
  - [ ] Measure latency
  - [ ] Generate validation report

---

## Phase 8: Docker & Deployment ⏳

### 8.1 Docker Configuration
- [ ] Implement `Dockerfile`
  - [ ] Multi-stage build
  - [ ] Python 3.11+ base
  - [ ] Install dependencies
  - [ ] Copy application
  - [ ] Non-root user
  - [ ] Health check
  - [ ] CMD uvicorn

### 8.2 Docker Compose
- [ ] Implement `docker-compose.yml`
  - [ ] API service definition
  - [ ] Volume mounts
  - [ ] Environment variables
  - [ ] Port mapping
  - [ ] Health checks
  - [ ] Restart policy

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

## Phase 10: Documentation ⏳

### 10.1 User Documentation
- [ ] Write `docs/quickstart.md`
  - [ ] Installation instructions
  - [ ] Quick start (5 minutes)
  - [ ] Run API server
  - [ ] Test with curl
  - [ ] Docker deployment

- [ ] Write `docs/training_guide.md`
  - [ ] Download training data
  - [ ] Train custom models
  - [ ] Hyperparameter tuning
  - [ ] Model evaluation
  - [ ] Best practices

- [ ] Write `docs/customization.md`
  - [ ] Add custom features
  - [ ] Modify model architecture
  - [ ] Change hyperparameters
  - [ ] Extend API endpoints
  - [ ] Advanced techniques

- [ ] Write `docs/api_reference.md`
  - [ ] Complete API documentation
  - [ ] All endpoints
  - [ ] Request/response schemas
  - [ ] Examples
  - [ ] Error codes

### 10.2 Main README
- [ ] Write `README.md`
  - [ ] Project overview
  - [ ] Architecture diagram
  - [ ] Quick start (copy from docs)
  - [ ] API usage examples
  - [ ] Training workflow
  - [ ] Contributing guidelines
  - [ ] License

### 10.3 Additional Docs
- [ ] Create `LICENSE` file (MIT)
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

**Phase 1**: ⬜ 0/13 (0%)  
**Phase 2**: ⬜ 0/8 (0%)  
**Phase 3**: ⬜ 0/15 (0%)  
**Phase 4**: ⬜ 0/12 (0%)  
**Phase 5**: ⬜ 0/21 (0%)  
**Phase 6**: ⬜ 0/17 (0%)  
**Phase 7**: ⬜ 0/15 (0%)  
**Phase 8**: ⬜ 0/13 (0%)  
**Phase 9**: ⬜ 0/23 (0%)  
**Phase 10**: ⬜ 0/14 (0%)  
**Phase 11**: ⬜ 0/7 (0%)  
**Phase 12**: ⬜ 0/15 (0%)  

**Overall**: ⬜ 0/173 tasks (0%)

---

## Notes

- Update progress as tasks are completed
- Mark with ✅ when phase is complete
- Add notes for any blockers or issues
- Track time spent on each phase
- Document any deviations from plan