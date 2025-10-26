# Data Storage and Validation Architecture
## Training, Validation, and DuckDB Integration

**Date**: 2025-10-26  
**Purpose**: Define data storage strategy, training workflow, and validation framework  

---

## Directory Structure

```
aml-miner-template/
├── input/                          # Raw data from SOT
│   └── {processing_date}/          # e.g., 2025-10-26/
│       ├── alerts.parquet
│       ├── features.parquet
│       ├── clusters.parquet
│       └── metadata.json
│
├── output/                         # Processed results
│   └── {processing_date}/          # e.g., 2025-10-26/
│       ├── alert_scores.parquet
│       ├── alert_rankings.parquet
│       ├── cluster_scores.parquet
│       └── processing_metadata.json
│
├── training_data/                  # Historical data for model training
│   ├── 2025-01-01/
│   ├── 2025-01-02/
│   ├── ...
│   ├── 2025-09-30/
│   └── ground_truth.parquet        # T+τ labeled data
│
├── validation_results/             # A/B testing results
│   ├── model_v1.0.0_evaluation.json
│   ├── model_v1.1.0_evaluation.json
│   └── comparison_report.md
│
└── trained_models/                 # Model artifacts
    ├── alert_scorer_v1.0.0.txt
    ├── alert_ranker_v1.0.0.txt
    └── model_metadata.json
```

---

## Model Training Workflow

### 1. Data Collection
```bash
# Download historical batches for training
python scripts/download_batch.py \
    --network ethereum \
    --start-date 2025-01-01 \
    --end-date 2025-09-30 \
    --output-dir training_data/

# Each date gets its own directory
# training_data/2025-01-01/alerts.parquet
# training_data/2025-01-02/alerts.parquet
# ...
```

### 2. Training
```bash
# Train models on historical data
python scripts/train_models.py \
    --data-dir training_data/ \
    --start-date 2025-01-01 \
    --end-date 2025-09-30 \
    --output trained_models/alert_scorer_v1.1.0.txt

# Uses all dates in range
# Creates deterministic LightGBM models
```

### 3. Validation (A/B Testing)
```bash
# Test new model against old model
python scripts/validate_models.py \
    --model-a trained_models/alert_scorer_v1.0.0.txt \
    --model-b trained_models/alert_scorer_v1.1.0.txt \
    --test-data training_data/ \
    --start-date 2025-09-01 \
    --end-date 2025-09-30 \
    --ground-truth training_data/ground_truth.parquet \
    --output validation_results/v1.0.0_vs_v1.1.0.json

# Compares:
# - Tier1: Determinism, format, latency
# - Tier2: Pattern trap detection
# - Tier3: Ground truth accuracy (T+τ)
```

---

## Validation Framework

### Overview
Implement the **exact same validation logic** that validators will use. This allows miners to:
- ✅ Test models locally before deployment
- ✅ Avoid code duplication (can be extracted to shared package later)
- ✅ Ensure submissions will pass validator checks

### Implementation Location
`aml_miner/validation/` - Shared validation code

```
aml_miner/
└── validation/
    ├── __init__.py
    ├── tier1_validator.py      # Integrity checks
    ├── tier2_validator.py      # Pattern traps
    ├── tier3_validator.py      # Ground truth scoring
    └── validator_utils.py
```

### Tier 1: Integrity Validation

```python
# aml_miner/validation/tier1_validator.py

from typing import Dict, List
import pandas as pd
from loguru import logger

class Tier1Validator:
    """
    Integrity checks:
    - Determinism (same input → same output)
    - Format validation (schema, types)
    - Completeness (all alerts scored)
    - Latency constraints
    """
    
    def validate(self, scores_df: pd.DataFrame, alerts_df: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'score': 0.0,
            'checks': {}
        }
        
        # Check 1: All alerts scored
        if len(scores_df) != len(alerts_df):
            results['checks']['completeness'] = False
            results['passed'] = False
            return results
        results['checks']['completeness'] = True
        
        # Check 2: Schema validation
        required_cols = ['alert_id', 'score', 'model_version', 'latency_ms']
        if not all(col in scores_df.columns for col in required_cols):
            results['checks']['schema'] = False
            results['passed'] = False
            return results
        results['checks']['schema'] = True
        
        # Check 3: Score range [0, 1]
        if not scores_df['score'].between(0, 1).all():
            results['checks']['score_range'] = False
            results['passed'] = False
            return results
        results['checks']['score_range'] = True
        
        # Check 4: Latency < 100ms per alert
        if scores_df['latency_ms'].mean() > 100:
            results['checks']['latency'] = False
            results['passed'] = False
            return results
        results['checks']['latency'] = True
        
        # Compute score (0-0.2 range)
        results['score'] = 0.2
        
        return results
```

### Tier 2: Pattern Trap Detection

```python
# aml_miner/validation/tier2_validator.py

class Tier2Validator:
    """
    Pattern trap detection:
    - Constant scores (gaming)
    - Copy other miners (plagiarism)
    - Known patterns (statistical anomalies)
    """
    
    def validate(self, scores_df: pd.DataFrame, pattern_traps: List[Dict]) -> Dict:
        results = {
            'passed': True,
            'score': 0.0,
            'traps_detected': []
        }
        
        # Check 1: Constant scores
        score_variance = scores_df['score'].var()
        if score_variance < 0.001:
            results['traps_detected'].append('constant_scores')
            results['score'] -= 0.1
        
        # Check 2: Pattern matching
        for trap in pattern_traps:
            trap_alert_id = trap['alert_id']
            expected_score = trap['expected_score']
            
            actual_score = scores_df[scores_df['alert_id'] == trap_alert_id]['score'].values[0]
            
            if abs(actual_score - expected_score) < 0.01:
                results['traps_detected'].append(f"trap_{trap_alert_id}")
                results['score'] -= 0.05
        
        # Normalize score to 0-0.3 range
        results['score'] = max(0.0, min(0.3, 0.3 + results['score']))
        
        return results
```

### Tier 3: Ground Truth Validation

```python
# aml_miner/validation/tier3_validator.py

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class Tier3Validator:
    """
    Ground truth scoring:
    - Compare predictions to T+τ labels
    - Compute AUC-ROC, AUC-PR
    - Final 50% of miner score
    """
    
    def validate(self, scores_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        # Merge predictions with ground truth
        merged = scores_df.merge(
            ground_truth_df[['alert_id', 'is_sar_filed']],
            on='alert_id',
            how='inner'
        )
        
        if len(merged) == 0:
            return {'score': 0.0, 'metrics': {}}
        
        # Compute metrics
        y_true = merged['is_sar_filed'].values
        y_pred = merged['score'].values
        
        auc_roc = roc_auc_score(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        
        # Combine metrics (0-0.5 range)
        score = (0.3 * auc_roc) + (0.2 * auc_pr)
        
        return {
            'score': score,
            'metrics': {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'n_samples': len(merged)
            }
        }
```

---

## DuckDB Integration Analysis

### Option A: Parquet Files Only (Current)

#### Pros
✅ **Simple** - No additional dependencies  
✅ **Fast for batch processing** - Parquet is optimized for columnar analytics  
✅ **Portable** - Easy to copy, backup, transfer  
✅ **Standard format** - Works with Pandas, Polars, PyArrow directly  
✅ **No database maintenance** - Just files on disk  

#### Cons
❌ **No SQL queries** - Must load entire file to filter  
❌ **No indexing** - Can't efficiently query specific dates  
❌ **Multiple files** - API must load many files for date ranges  
❌ **Memory pressure** - Loading large Parquet files into Pandas  

#### Use Case
Best for:
- Simple workflows (one date at a time)
- Small to medium datasets (< 1GB per file)
- Batch processing scripts

---

### Option B: DuckDB Integration (Recommended)

#### Pros
✅ **SQL queries** - Filter, aggregate, join without loading all data  
✅ **Zero-copy Parquet reading** - DuckDB reads Parquet files directly  
✅ **Efficient indexing** - Fast queries on date, alert_id, etc.  
✅ **In-process database** - No separate server, just a file  
✅ **Analytics optimized** - Built for OLAP workloads  
✅ **Memory efficient** - Streams data, doesn't load everything  
✅ **Multi-file queries** - Query across all dates in one SQL statement  

#### Cons
❌ **Additional dependency** - Need `duckdb` package (lightweight though)  
❌ **Learning curve** - SQL queries instead of Pandas  
❌ **Database file size** - `.db` file adds overhead (minimal)  

#### Use Case
Best for:
- Multiple date queries (e.g., "last 30 days")
- Large datasets (> 1GB)
- API serving pre-computed results efficiently
- Training data management (query specific date ranges)

---

### Recommended Architecture: Hybrid Approach

```
Workflow:
1. Scripts write Parquet files → output/{processing_date}/
2. DuckDB reads Parquet files directly (no import needed)
3. API queries DuckDB for efficient data access

Benefits:
- Keep Parquet files as source of truth (portable, standard)
- Use DuckDB as query layer (fast, efficient)
- No data duplication (DuckDB reads Parquet directly)
- Best of both worlds
```

### Implementation

```python
# aml_miner/api/database.py

import duckdb
from pathlib import Path

class ResultsDatabase:
    """
    DuckDB interface for querying pre-computed results
    """
    
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.conn = duckdb.connect(":memory:")
        self._register_parquet_files()
    
    def _register_parquet_files(self):
        """
        Register all Parquet files as DuckDB tables
        DuckDB can read Parquet files directly without import
        """
        # Create view over all alert scores
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW alert_scores AS
            SELECT 
                *,
                CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
            FROM read_parquet('{self.output_dir}/*/alert_scores.parquet', filename=true)
        """)
        
        # Create view over all rankings
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW alert_rankings AS
            SELECT 
                *,
                CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
            FROM read_parquet('{self.output_dir}/*/alert_rankings.parquet', filename=true)
        """)
        
        # Create view over all cluster scores
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW cluster_scores AS
            SELECT 
                *,
                CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
            FROM read_parquet('{self.output_dir}/*/cluster_scores.parquet', filename=true)
        """)
    
    def get_alert_scores(self, processing_date: str):
        """Get alert scores for specific date"""
        return self.conn.execute(
            "SELECT * FROM alert_scores WHERE processing_date = ?",
            [processing_date]
        ).df()
    
    def get_alert_rankings(self, processing_date: str):
        """Get alert rankings for specific date"""
        return self.conn.execute(
            "SELECT * FROM alert_rankings WHERE processing_date = ?",
            [processing_date]
        ).df()
    
    def get_cluster_scores(self, processing_date: str):
        """Get cluster scores for specific date"""
        return self.conn.execute(
            "SELECT * FROM cluster_scores WHERE processing_date = ?",
            [processing_date]
        ).df()
    
    def get_available_dates(self):
        """Get list of all available processing dates"""
        return self.conn.execute(
            "SELECT DISTINCT processing_date FROM alert_scores ORDER BY processing_date DESC"
        ).df()
```

### Updated API Routes

```python
# aml_miner/api/routes.py

from fastapi import APIRouter, HTTPException
from aml_miner.api.database import ResultsDatabase

router = APIRouter()
db = ResultsDatabase()

@router.get("/scores/alerts/{processing_date}")
def get_alert_scores(processing_date: str):
    """
    Get pre-computed alert scores for specific date
    """
    try:
        scores_df = db.get_alert_scores(processing_date)
        
        if scores_df.empty:
            raise HTTPException(404, f"No scores for date {processing_date}")
        
        return scores_df.to_dict(orient='records')
    
    except Exception as e:
        raise HTTPException(500, f"Error retrieving scores: {e}")

@router.get("/dates/available")
def get_available_dates():
    """
    Get list of all available processing dates
    """
    dates_df = db.get_available_dates()
    return dates_df['processing_date'].tolist()
```

---

## Summary

### Data Storage Strategy
**Use DuckDB as query layer over Parquet files**

1. **Parquet files** - Source of truth, portable, standard format
2. **DuckDB** - Efficient query layer, no data duplication
3. **API** - Reads from DuckDB for fast, efficient access

### Training Workflow
1. Download historical batches → `training_data/`
2. Train models using `scripts/train_models.py`
3. Validate with `scripts/validate_models.py` (A/B testing)
4. Deploy best model

### Validation Framework
Implement Tier1, Tier2, Tier3 validators in `aml_miner/validation/`
- Same code that validators will use
- Can be extracted to shared package later
- Enables local testing before deployment

### Benefits
✅ **Efficient** - DuckDB enables fast queries without loading all data  
✅ **Simple** - Parquet files remain portable and easy to work with  
✅ **Testable** - Local validation using same logic as validators  
✅ **Scalable** - Works well with growing datasets  
✅ **Zero duplication** - DuckDB reads Parquet directly  