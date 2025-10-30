# Example Scripts

Python examples for running the risk scoring pipeline. Perfect for development - just click Run in PyCharm without typing CLI arguments.

## Quick Start

Run any example directly in PyCharm or from command line:

```bash
python scripts/examples/example_ingest.py
python scripts/examples/example_train.py
python scripts/examples/example_score.py
python scripts/examples/example_full_pipeline.py
```

## How It Works

Each example script:
1. Sets `sys.argv` with predefined parameters
2. Calls the `main()` function from production CLI scripts
3. Production CLI behavior preserved - you get all logs and proper error handling

### Example Structure

```python
"""Developer entry point with predefined parameters"""
import sys

if __name__ == "__main__":
    sys.argv = [
        'train_model.py',
        '--network', 'torus',
        '--start-date', '2025-08-01',
        '--end-date', '2025-08-01',
        '--model-type', 'alert_scorer',
        '--window-days', '195'
    ]
    
    from scripts.train_model import main
    main()
```

## Available Examples

### 1. Data Ingestion
**File:** [`example_ingest.py`](example_ingest.py:1)

Ingests data from SOT S3 bucket to ClickHouse.

**Predefined parameters:**
- Network: `torus`
- Processing date: `2025-08-01`
- Window days: `7`

### 2. Model Training
**File:** [`example_train.py`](example_train.py:1)

Trains XGBoost models for alert scoring.

**Predefined parameters:**
- Network: `torus`
- Start date: `2025-08-01`
- End date: `2025-08-01`
- Model type: `alert_scorer`
- Window days: `195`

### 3. Risk Scoring
**File:** [`example_score.py`](example_score.py:1)

Generates risk scores using trained models.

**Predefined parameters:**
- Network: `torus`
- Processing date: `2025-08-01`
- Window days: `7`

### 4. Full Pipeline
**File:** [`example_full_pipeline.py`](example_full_pipeline.py:1)

Runs complete pipeline: ingestion → training → scoring.

Uses `subprocess` to run each step in separate process for proper logging.

## Customization

Simply edit the `sys.argv` values in any example file:

```python
sys.argv = [
    'ingest_data.py',
    '--network', 'ethereum',      # Change network
    '--processing-date', '2025-08-15',  # Different date
    '--days', '30'                 # Different window
]
```

## Production CLI Scripts

These examples wrap production CLI scripts from [`scripts/`](../) directory:
- [`ingest_data.py`](../ingest_data.py:1) - Production data ingestion
- [`train_model.py`](../train_model.py:1) - Production model training
- [`score_batch.py`](../score_batch.py:1) - Production risk scoring

All production scripts have `main()` functions that can be called programmatically.

## CLI Parameters Reference

### Ingestion (`ingest_data.py`)
- `--network`: Network identifier (ethereum, bitcoin, torus)
- `--processing-date`: Date in YYYY-MM-DD format
- `--days`: Window days (7, 30, 90, 195)

### Training (`train_model.py`)
- `--network`: Network identifier
- `--start-date`: Training start date (YYYY-MM-DD)
- `--end-date`: Training end date (YYYY-MM-DD)
- `--model-type`: alert_scorer, alert_ranker, or cluster_scorer
- `--window-days`: Feature window (7, 30, 90, 195)
- `--output-dir`: Custom model output directory (optional)

### Scoring (`score_batch.py`)
- `--network`: Network identifier
- `--processing-date`: Date to score (YYYY-MM-DD)
- `--window-days`: Window days (7, 30, 90, 195)
- `--models-dir`: Custom models directory (optional)
- `--model-types`: List of model types to run (optional)

## Environment Setup

Configure `.env` file with ClickHouse and S3 credentials:

```bash
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=

RISK_SCORING_S3_ENDPOINT=https://s3.example.com
RISK_SCORING_S3_BUCKET=risk-scoring-data
RISK_SCORING_S3_REGION=nl-ams
```

## PyCharm Usage

1. Open any example file (e.g., `example_train.py`)
2. Right-click → Run 'example_train'
3. Logs appear in console automatically
4. Modify parameters directly in the file as needed

No CLI configuration needed in PyCharm!

## Notes

- **Full pipeline** uses `subprocess` to run steps in separate processes for proper logging
- **Individual examples** run in same process - faster but may have logger conflicts if run sequentially
- For sequential runs, use full pipeline or run examples separately
- All examples are self-contained and ready to use