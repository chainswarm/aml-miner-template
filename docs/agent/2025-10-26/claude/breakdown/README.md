# AML Miner Template Implementation - Breakdown Overview

**Date**: 2025-10-26  
**Project**: aml-miner-template  
**Status**: Planning Complete ✅ | Implementation Ready 🚀

---

## 📋 Document Index

This breakdown consists of four comprehensive documents:

1. **[IMPLEMENTATION_BREAKDOWN.md](./IMPLEMENTATION_BREAKDOWN.md)**
   - High-level implementation plan
   - Organized into 8 sprints (12 days)
   - Architecture decisions
   - Current state vs. target state
   - Recommended implementation order

2. **[TECHNICAL_SPECIFICATION.md](./TECHNICAL_SPECIFICATION.md)**
   - Detailed technical specifications
   - Data structures and schemas
   - Model specifications (hyperparameters, features)
   - API contracts and endpoints
   - Configuration system details
   - Performance requirements
   - Error handling strategy

3. **[DATA_SCHEMA_MAPPING.md](./DATA_SCHEMA_MAPPING.md)** ⭐ NEW
   - Complete mapping from SOT schemas to Parquet files
   - API request/response formats (Pydantic models)
   - Data loading implementation
   - Feature engineering pipeline
   - 160+ feature descriptions

4. **[IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)**
   - Granular task checklist (173 tasks)
   - Organized into 12 phases
   - Progress tracking
   - Completion criteria
   - Can be used as project management tool

5. **[README.md](./README.md)** (this file)
   - Overview and navigation
   - Quick reference guide
   - Next steps

---

## 🎯 Project Goal

Implement a **reusable ML library** for AML alert scoring that:

- ✅ Provides FastAPI server for inference
- ✅ Includes pretrained models (scorer, ranker, cluster scorer)
- ✅ Supports custom model training
- ✅ Ensures deterministic predictions
- ✅ Can be forked and customized by miners
- ✅ Separates ML code from Bittensor integration

---

## 📊 Current Status

### What Exists
- Empty directory structure: `template/`
- Placeholder files (all empty):
  - `template/api/server.py`
  - `template/models/alert_scorer.py`
  - `template/models/alert_ranker.py`
  - `template/models/cluster_scorer.py`
  - `pyproject.toml`
  - `Dockerfile`
  - `README.md`

### What's Missing
**Everything** - Complete implementation needed (173 tasks)

---

## 🏗️ Architecture Summary

### Repository Structure
```
aml-miner-template/
├── aml_miner/              # Main package (importable)
│   ├── api/                # FastAPI server
│   ├── models/             # ML models (scorer, ranker)
│   ├── features/           # Feature engineering
│   ├── training/           # Training pipelines
│   ├── utils/              # Utilities
│   └── config/             # Configuration
├── trained_models/         # Pretrained models
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### Key Components

**1. FastAPI Server** (`aml_miner/api/`)
- Endpoints: `/score/alerts`, `/rank/alerts`, `/score/clusters`
- Async handlers, < 1ms latency per alert
- Health checks, metrics, versioning

**2. ML Models** (`aml_miner/models/`)
- `AlertScorerModel` - LightGBM binary classifier
- `AlertRankerModel` - LightGBM ranker (LambdaRank)
- `ClusterScorerModel` - Cluster risk scorer
- Base class with common interface

**3. Feature Engineering** (`aml_miner/features/`)
- Alert features (~30)
- Network features (~20)
- Cluster features (~15)
- Statistical aggregations (~15)

**4. Training Pipelines** (`aml_miner/training/`)
- Train custom models
- Hyperparameter tuning
- Cross-validation
- Model evaluation

---

## 🚀 Quick Start Guide

### For Reviewing the Plan

1. **Start with** [IMPLEMENTATION_BREAKDOWN.md](./IMPLEMENTATION_BREAKDOWN.md)
   - Understand the high-level approach
   - Review sprint organization
   - Check architecture decisions

2. **Then review** [TECHNICAL_SPECIFICATION.md](./TECHNICAL_SPECIFICATION.md)
   - Understand detailed specifications
   - Review data schemas
   - Check model specifications
   - Understand API contracts

3. **Review** [DATA_SCHEMA_MAPPING.md](./DATA_SCHEMA_MAPPING.md)
   - Understand SOT schema mapping to Parquet
   - See complete API request/response formats
   - Review 160+ feature descriptions
   - Understand data loading pipeline

4. **Finally check** [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)
   - See granular task breakdown
   - Understand dependencies between tasks
   - Review completion criteria

### For Implementation

1. **Follow the sprints** in [IMPLEMENTATION_BREAKDOWN.md](./IMPLEMENTATION_BREAKDOWN.md)
2. **Check off tasks** in [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)
3. **Refer to specs** in [TECHNICAL_SPECIFICATION.md](./TECHNICAL_SPECIFICATION.md)
4. **Track progress** in the checklist

---

## 📈 Implementation Plan

### Recommended Sprint Order

| Sprint | Duration | Focus | Tasks |
|--------|----------|-------|-------|
| **Sprint 1** | Day 1-2 | Foundation | Directory structure, package config, dependencies |
| **Sprint 2** | Day 2-3 | Config & Utils | Settings, validators, data loaders, determinism |
| **Sprint 3** | Day 3-5 | Models | Base model, scorer, ranker, cluster scorer |
| **Sprint 4** | Day 5-6 | Features | Feature builder, feature selector |
| **Sprint 5** | Day 6-7 | API | Schemas, routes, server, endpoints |
| **Sprint 6** | Day 8-9 | Training | Training pipelines, hyperparameter tuning |
| **Sprint 7** | Day 9-10 | Scripts & Docker | Deployment scripts, Dockerfile, compose |
| **Sprint 8** | Day 10-12 | Testing & Docs | Tests, documentation, final validation |

**Total**: ~10-12 developer days

---

## ✅ Success Criteria

The implementation is complete when:

- [ ] API server runs and responds to all endpoints
- [ ] Models load and make deterministic predictions
- [ ] Determinism test passes 100 iterations (100% pass rate)
- [ ] Performance meets targets (< 1ms latency, 1000+ alerts/sec)
- [ ] Docker container builds and runs successfully
- [ ] All 173 tasks in checklist are complete
- [ ] All tests pass (> 80% coverage)
- [ ] Documentation is complete and accurate
- [ ] Pretrained models are available
- [ ] Ready for miners to fork and customize

---

## 🔑 Critical Requirements

### 1. Determinism (HIGHEST PRIORITY)
- Same input MUST produce same output
- Set all random seeds
- Test extensively (100 iterations)
- Document in tests

### 2. Performance
- Latency: < 1ms per alert
- Throughput: 1000+ alerts/second
- Memory: < 2GB for API server

### 3. Explainability
- Every score must have SHAP explanation
- Top-5 contributing features
- Human-readable JSON format

### 4. Reusability
- Clean, modular code
- Well-documented
- Easy to customize
- Follows Python best practices

---

## 📝 Key Design Decisions

### Model Serialization
**Decision**: LightGBM text format (`.txt`)  
**Rationale**: Deterministic, human-readable, version control friendly

### API Framework
**Decision**: FastAPI with async handlers  
**Rationale**: High performance, automatic docs, type safety

### Configuration
**Decision**: Pydantic Settings + YAML  
**Rationale**: Type-safe, env var support, easy validation

### Logging
**Decision**: Loguru  
**Rationale**: Simple API, structured logging, good performance

### Testing
**Decision**: Pytest with fixtures  
**Rationale**: Industry standard, powerful, extensible

---

## 🎓 Learning Resources

### For Understanding the Architecture
- Read: `docs/agent/2025-10-25/claude/MINER_TEMPLATE_AND_SUBNET_ARCHITECTURE.md`
- Understand the two-repository pattern
- See how miner proxies to template API

### For Understanding Validation
- Read: `docs/agent/2025-10-25/claude/MINER_CAPABILITIES_AND_VALIDATION_PROPOSAL.md`
- Understand 3-tier validation
- See Strategy 2 (Weighted Ensemble)

### For Understanding SOT Extensions
- Read: `docs/agent/2025-10-25/claude/SOT_PROPOSAL_4_5_IMPLEMENTATION_PLAN.md`
- Understand data format
- See batch structure

---

## 📦 Dependencies

### Core
- Python 3.13.5+
- FastAPI 0.104.0+
- Pydantic 2.5.0+
- LightGBM 4.3.0+
- pandas 2.1.0+
- numpy 1.26.0+

### ML
- scikit-learn 1.3.0+
- SHAP 0.44.0+

### Utils
- loguru 0.7.2+
- httpx 0.25.0+
- PyNaCl 1.5.0+

### Development
- pytest 7.4.0+
- black 23.11.0+
- ruff 0.1.6+
- mypy 1.7.0+

---

## 🚦 Next Steps

### Option 1: Review & Approve
1. Review all three documents
2. Provide feedback or approval
3. Prioritize any changes needed
4. Confirm sprint order

### Option 2: Start Implementation
1. Switch to Code mode
2. Begin with Sprint 1 (Foundation)
3. Follow checklist task-by-task
4. Track progress in checklist

### Option 3: Create Issues/Tasks
1. Create GitHub issues for each sprint (optional)
2. Assign developers (if team)
3. Set up project board (optional)
4. Begin parallel work (if multiple devs)

---

## 📞 Support

For questions about:
- **Architecture**: See [IMPLEMENTATION_BREAKDOWN.md](./IMPLEMENTATION_BREAKDOWN.md)
- **Technical details**: See [TECHNICAL_SPECIFICATION.md](./TECHNICAL_SPECIFICATION.md)
- **Specific tasks**: See [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)
- **Original design**: See `docs/agent/2025-10-25/claude/MINER_TEMPLATE_AND_SUBNET_ARCHITECTURE.md`

---

## 📊 Metrics

| Metric | Value |
|--------|-------|
| Total Tasks | 173 |
| Estimated Days | 10-12 |
| Documents | 4 main + 1 overview |
| Phases | 12 |
| Sprints | 8 |
| Success Criteria | 10 |
| Critical Requirements | 4 |
| Core Dependencies | 12 |
| Python Version | 3.13.5+ |
| SOT Schema Files | 3 (alerts, features, clusters) |
| Total Features Available | 160+ |

---

## ✨ Summary

This breakdown provides:

✅ **Complete implementation plan** (173 tasks)  
✅ **Detailed technical specifications** (data, models, API)  
✅ **Clear sprint organization** (8 sprints, 12 days)  
✅ **Progress tracking** (checklist with checkboxes)  
✅ **Success criteria** (10 clear objectives)  
✅ **Architecture decisions** (documented rationale)  
✅ **Ready for implementation** (can start immediately)

The plan is comprehensive, actionable, and ready to execute. 🚀

---

**Status**: ✅ Planning Complete
**Next**: 🚀 Ready for Implementation
**Mode**: Switch to Code mode to begin Sprint 1

---

## 📊 Data Schema Alignment

All data schemas in this implementation are **100% aligned** with SOT (Source of Truth) schema files:

- **Alerts**: [`docs/sot/schema/analyzers_alerts.sql`](../../sot/schema/analyzers_alerts.sql:1) (51 lines)
- **Features**: [`docs/sot/schema/analyzers_features.sql`](../../sot/schema/analyzers_features.sql:1) (160+ features, 236 lines)
- **Clusters**: [`docs/sot/schema/analyzers_alert_clusters.sql`](../../sot/schema/analyzers_alert_clusters.sql:1) (53 lines)

Miners receive data as **Parquet files** with schemas compatible with these SQL definitions.