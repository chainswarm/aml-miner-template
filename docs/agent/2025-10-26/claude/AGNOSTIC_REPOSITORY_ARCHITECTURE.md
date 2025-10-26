# Risk Scoring Repository - Agnostic Architecture

**Date**: 2025-10-26  
**Purpose**: Reorganize repository to support both scoring and evaluation roles without role-specific terminology

---

## Problem Statement

Current repository uses "miner" terminology (`aml_miner/`) and is organized for only one role. However, this is a **risk scoring repository** that should support:

1. **Risk Assessment** - Generating risk scores for alerts (miner role)
2. **Quality Evaluation** - Validating submissions and aggregating scores (validator role)

The code should be organized **logically** by function, not by network role.

---

## Proposed Package Structure

```
alert_scoring/  (rename from aml_miner/)
│
├── assessment/              # Risk Assessment Module
│   ├── __init__.py
│   ├── models/              # ML models for scoring
│   │   ├── alert_scorer.py
│   │   ├── alert_ranker.py
│   │   ├── cluster_scorer.py
│   │   └── base_model.py
│   ├── features/            # Feature engineering
│   │   ├── feature_builder.py
│   │   └── feature_selector.py
│   └── training/            # Model training
│       ├── train_scorer.py
│       └── hyperparameter_tuner.py
│
├── evaluation/              # Quality Evaluation Module
│   ├── __init__.py
│   ├── quality/             # Quality checks
│   │   ├── integrity.py     # Schema, completeness, determinism
│   │   ├── behavior.py      # Pattern traps, plagiarism
│   │   └── performance.py   # AUC-ROC, F1 vs ground truth
│   ├── aggregation/         # Multi-submission aggregation
│   │   ├── ensemble.py      # Weighted ensemble strategies
│   │   ├── consensus.py     # Consensus algorithms
│   │   └── strategies.py    # Aggregation strategy interface
│   └── metrics/             # Performance metrics
│       ├── score_metrics.py
│       └── comparison.py
│
├── storage/                 # Data Storage Layer (shared)
│   ├── __init__.py
│   ├── repositories/        # Data access
│   ├── schema/              # ClickHouse schemas
│   └── utils.py
│
├── api/                     # REST API (shared)
│   ├── __init__.py
│   ├── server.py
│   ├── routes.py
│   └── schemas.py
│
├── config/                  # Configuration (shared)
│   ├── __init__.py
│   └── settings.py
│
└── utils/                   # Utilities (shared)
    ├── __init__.py
    └── data_loader.py
```

---

## Module Responsibilities

### 1. Assessment Module (`assessment/`)

**Purpose**: Generate risk scores for alerts

**Components**:
- **Models**: ML models (AlertScorer, AlertRanker, ClusterScorer)
- **Features**: Feature engineering pipelines
- **Training**: Model training and hyperparameter tuning

**Used by**: 
- Miners running local scoring API
- Standalone risk assessment systems

**Key Files**:
- `assessment/models/alert_scorer.py` - Score individual alerts
- `assessment/models/alert_ranker.py` - Rank alerts by priority
- `assessment/training/train_scorer.py` - Train new models

### 2. Evaluation Module (`evaluation/`)

**Purpose**: Evaluate submission quality and aggregate multiple submissions

**Components**:
- **Quality**: Quality checks (integrity, behavior, performance)
- **Aggregation**: Combine multiple submissions into consensus
- **Metrics**: Performance measurement and comparison

**Used by**:
- Validators aggregating multiple miner submissions
- A/B testing between model versions
- Quality monitoring systems

**Key Files**:
- `evaluation/quality/integrity.py` - Schema validation, completeness
- `evaluation/quality/behavior.py` - Pattern detection, plagiarism
- `evaluation/quality/performance.py` - AUC-ROC, F1 vs ground truth
- `evaluation/aggregation/ensemble.py` - Weighted ensemble aggregation
- `evaluation/aggregation/strategies.py` - Strategy interface

### 3. Storage Module (`storage/`)

**Purpose**: Shared data layer for both assessment and evaluation

**Components**:
- **Repositories**: Data access patterns
- **Schema**: ClickHouse table definitions
- **Utils**: Database utilities

**Used by**: Both assessment and evaluation modules

**Tables**:
- `raw_alerts`, `raw_features`, `raw_clusters` - Input data
- `alert_scores`, `alert_rankings`, `cluster_scores` - Assessment outputs
- `submission_scores` (NEW) - Multiple submissions for aggregation
- `aggregated_scores` (NEW) - Consensus/aggregated results
- `quality_reports` (NEW) - Evaluation results
- `batch_metadata` - Processing metadata

### 4. API Module (`api/`)

**Purpose**: Serve both assessment and evaluation results

**Endpoints**:
```
# Assessment outputs (single submission)
GET /assessment/scores/latest
GET /assessment/rankings/latest

# Evaluation outputs (multiple submissions)
GET /evaluation/submissions/{batch_id}
GET /evaluation/aggregated/{batch_id}
GET /evaluation/quality/{submission_id}

# Shared
GET /dates/available
GET /metadata/latest
```

---

## Workflow Examples

### Workflow 1: Risk Assessment (Miner Role)

```bash
# 1. Download data
python scripts/download_from_sot.py --processing-date 2024-01-15

# 2. Run assessment
python scripts/run_assessment.py --processing-date 2024-01-15

# 3. Serve scores
python -m alert_scoring.api.server --mode assessment
```

**Code Flow**:
```python
from alert_scoring.assessment.models import AlertScorerModel
from alert_scoring.storage.repositories import AlertsRepository, ScoresRepository

# Load data
alerts = AlertsRepository().get_alerts(date, network)

# Assess
scorer = AlertScorerModel()
scores = scorer.predict(alerts)

# Store
ScoresRepository().insert_scores(scores, date, network)
```

### Workflow 2: Quality Evaluation (Validator Role)

```bash
# 1. Collect submissions from multiple miners
python scripts/collect_submissions.py --batch-id batch_001

# 2. Run evaluation
python scripts/run_evaluation.py --batch-id batch_001

# 3. Generate aggregated result
python scripts/aggregate_scores.py --batch-id batch_001
```

**Code Flow**:
```python
from alert_scoring.evaluation.quality import IntegrityChecker, BehaviorChecker
from alert_scoring.evaluation.aggregation import WeightedEnsemble
from alert_scoring.storage.repositories import SubmissionsRepository

# Load submissions
submissions = SubmissionsRepository().get_submissions(batch_id)

# Evaluate each submission
for submission in submissions:
    integrity_score = IntegrityChecker().evaluate(submission)
    behavior_score = BehaviorChecker().evaluate(submission)
    # Store quality scores

# Aggregate
weights = compute_weights(quality_scores)
aggregated = WeightedEnsemble().aggregate(submissions, weights)
```

---

## New Database Tables

### submission_scores

Stores individual submissions from multiple sources (miners):

```sql
CREATE TABLE submission_scores (
    submission_id String,
    submitter_id String,
    batch_id String,
    processing_date Date,
    network String,
    alert_id String,
    score Float64,
    model_version String,
    timestamp DateTime,
    
    PRIMARY KEY (batch_id, submitter_id, alert_id)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (batch_id, submitter_id, alert_id);
```

### aggregated_scores

Stores consensus/aggregated results:

```sql
CREATE TABLE aggregated_scores (
    batch_id String,
    processing_date Date,
    network String,
    alert_id String,
    aggregated_score Float64,
    aggregation_method String,
    num_submissions UInt32,
    variance Float64,
    timestamp DateTime,
    
    PRIMARY KEY (batch_id, alert_id)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (batch_id, alert_id);
```

### quality_reports

Stores evaluation results for each submission:

```sql
CREATE TABLE quality_reports (
    submission_id String,
    batch_id String,
    submitter_id String,
    processing_date Date,
    network String,
    
    integrity_score Float64,
    behavior_score Float64,
    performance_score Float64,
    total_score Float64,
    
    details String,  -- JSON with detailed metrics
    timestamp DateTime,
    
    PRIMARY KEY (batch_id, submitter_id)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (batch_id, submitter_id);
```

---

## Migration Path

### Phase 1: Rename Package
```bash
git mv aml_miner alert_scoring
# Update all imports
# Update pyproject.toml
```

### Phase 2: Reorganize Code
```bash
# Create new structure
mkdir -p alert_scoring/assessment/{models,features,training}
mkdir -p alert_scoring/evaluation/{quality,aggregation,metrics}

# Move files
mv alert_scoring/models/* alert_scoring/assessment/models/
mv alert_scoring/features/* alert_scoring/assessment/features/
mv alert_scoring/training/* alert_scoring/assessment/training/

mv alert_scoring/validation/integrity_validator.py alert_scoring/evaluation/quality/integrity.py
mv alert_scoring/validation/behavior_validator.py alert_scoring/evaluation/quality/behavior.py
mv alert_scoring/validation/ground_truth_validator.py alert_scoring/evaluation/quality/performance.py
```

### Phase 3: Add Aggregation Logic
```bash
# Create new files
alert_scoring/evaluation/aggregation/ensemble.py
alert_scoring/evaluation/aggregation/strategies.py
```

### Phase 4: Update API
```bash
# Add evaluation endpoints
alert_scoring/api/routes.py  # Add /evaluation/* endpoints
```

### Phase 5: Add New Schemas
```bash
alert_scoring/storage/schema/submission_scores.sql
alert_scoring/storage/schema/aggregated_scores.sql
alert_scoring/storage/schema/quality_reports.sql
```

---

## Benefits

1. **Role-Agnostic**: Code is organized by function, not network role
2. **Reusable**: Both miners and validators can use this repository
3. **Clear Separation**: Assessment vs Evaluation modules have distinct responsibilities
4. **Extensible**: Easy to add new aggregation strategies or quality checks
5. **Testable**: Each module can be tested independently

---

## Example Use Cases

### Use Case 1: Solo Miner
```python
from alert_scoring.assessment import AlertScorerModel
# Just use assessment module
```

### Use Case 2: Validator
```python
from alert_scoring.evaluation import IntegrityChecker, WeightedEnsemble
# Just use evaluation module
```

### Use Case 3: Testing/Research
```python
from alert_scoring.assessment import AlertScorerModel
from alert_scoring.evaluation import BehaviorChecker
# Use both modules
```

---

## Summary

**Old**: `aml_miner/` with miner-specific organization  
**New**: `alert_scoring/` with functional organization

**Key Modules**:
- `assessment/` - Risk scoring (miner functionality)
- `evaluation/` - Quality checking & aggregation (validator functionality)
- `storage/` - Shared data layer
- `api/` - Shared API layer

This structure supports both roles while remaining **completely agnostic** to network-specific terminology.