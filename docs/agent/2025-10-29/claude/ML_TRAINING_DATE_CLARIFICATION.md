# ML Training Date Range Clarification

**Date**: 2025-10-29  
**Purpose**: Clarify date range semantics for training  

---

## Two Temporal Dimensions in Data

Looking at the schema, we have:

1. **`processing_date`** - When the snapshot was processed (batch date)
2. **`window_days`** - Lookback window (e.g., 7, 30, 90 days)

```sql
CREATE TABLE raw_alerts (
    window_days UInt16,        -- Lookback window (7, 30, 90 days)
    processing_date Date,      -- Batch/snapshot date
    alert_id String,
    ...
)
```

---

## Training Date Range Semantics

### ✅ Correct Interpretation

**`start_date` and `end_date` = Range of `processing_date` values**

This means:
- We train on **multiple daily snapshots**
- Each snapshot has data from its own `window_days` lookback period
- We don't need to worry about block timestamps - that's already encoded in the data

### Example

```python
# Train on 3 months of daily snapshots
start_date = "2024-01-01"    # First processing_date
end_date = "2024-03-31"      # Last processing_date
window_days = 7              # Fixed parameter (filter)
```

**SQL Query**:
```sql
SELECT *
FROM raw_alerts
WHERE processing_date >= '2024-01-01'
  AND processing_date <= '2024-03-31'
  AND window_days = 7
```

**Result**: 
- ~90 daily snapshots (one per day)
- Each snapshot contains alerts from a 7-day window
- Total: ~90 snapshots × alerts per day

---

## Why This Approach?

### 1. Multiple Training Samples
Each `processing_date` is a separate snapshot with its own alerts:
- 2024-01-01: alerts from blocks ~Dec 25 - Jan 1
- 2024-01-02: alerts from blocks ~Dec 26 - Jan 2
- 2024-01-03: alerts from blocks ~Dec 27 - Jan 3
- ...

This gives us **many more training samples** than a single date.

### 2. Temporal Patterns
Training across multiple dates captures:
- Day-of-week patterns
- Monthly trends
- Evolving attack patterns
- Market condition variations

### 3. No Block Timestamp Confusion
We don't need to specify block timestamp ranges because:
- The `window_days` parameter already defines the lookback
- Each `processing_date` snapshot contains properly windowed data
- The ingestion system handles all the block timestamp logic

---

## Training Query Examples

### Train on Single Week
```sql
-- 7 daily snapshots
SELECT *
FROM raw_alerts
WHERE processing_date >= '2024-01-01'
  AND processing_date <= '2024-01-07'
  AND window_days = 7
```

### Train on Quarter
```sql
-- ~90 daily snapshots
SELECT *
FROM raw_alerts
WHERE processing_date >= '2024-01-01'
  AND processing_date <= '2024-03-31'
  AND window_days = 7
```

### Train on Year
```sql
-- ~365 daily snapshots
SELECT *
FROM raw_alerts
WHERE processing_date >= '2023-01-01'
  AND processing_date <= '2023-12-31'
  AND window_days = 7
```

---

## Model Naming Convention

### Current Design
```
alert_scorer_ethereum_v1.0.0_20250129_103045.txt
    ↑           ↑         ↑         ↑
  model_type  network  version  timestamp
```

### Enhanced with Training Period
```
alert_scorer_ethereum_v1.0.0_20240101-20240331_20250129.txt
    ↑           ↑         ↑      ↑                  ↑
  model_type  network  version  training_period   created
```

**Metadata includes**:
```json
{
  "training_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "window_days": 7,
    "num_snapshots": 90
  }
}
```

---

## Fixed Window Days

For each model training run, we use a **fixed `window_days`** value:

```python
# Train with 7-day windows
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --window-days 7

# Train with 30-day windows (different model)
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --window-days 30
```

**Why fixed?**
- Features are computed differently for different windows
- Can't mix 7-day and 30-day windows in same model
- Each window size = separate model variant

---

## Updated Feature Extraction

```python
def _extract_alerts(
    self,
    start_date: str,
    end_date: str,
    window_days: int
) -> pd.DataFrame:
    """
    Extract alerts for training.
    
    Args:
        start_date: Start processing_date (inclusive)
        end_date: End processing_date (inclusive)
        window_days: Fixed window size (7, 30, 90, etc.)
    
    Returns:
        DataFrame with all alerts from snapshots in date range
    """
    
    query = f"""
        SELECT *
        FROM raw_alerts
        WHERE processing_date >= '{start_date}'
          AND processing_date <= '{end_date}'
          AND window_days = {window_days}
        ORDER BY processing_date, alert_id
    """
    
    result = self.client.query(query)
    
    if not result.result_rows:
        raise ValueError(
            f"No alerts found for processing_date range "
            f"{start_date} to {end_date} with window_days={window_days}"
        )
    
    df = pd.DataFrame(
        result.result_rows,
        columns=[col[0] for col in result.column_names]
    )
    
    logger.info(
        f"Extracted {len(df):,} alerts from "
        f"{df['processing_date'].nunique()} snapshots"
    )
    
    return df
```

---

## Benefits

### ✅ Clear Semantics
- `start_date`/`end_date` = processing_date range
- `window_days` = fixed lookback parameter
- No confusion about block timestamps

### ✅ More Training Data
- Multiple daily snapshots
- Better temporal coverage
- More robust models

### ✅ Consistent with Ingestion
- Uses same date concepts as ingestion
- Follows existing schema design
- No new temporal logic needed

### ✅ Flexible Training
```python
# Quick test: 1 week of data
start_date="2024-01-01", end_date="2024-01-07"

# Medium: 1 month
start_date="2024-01-01", end_date="2024-01-31"

# Full: 1 year
start_date="2023-01-01", end_date="2023-12-31"
```

---

## Summary

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `start_date` | First `processing_date` to include | "2024-01-01" |
| `end_date` | Last `processing_date` to include | "2024-03-31" |
| `window_days` | Fixed lookback window (filter) | 7 |
| **Result** | All alerts from snapshots in range | ~90 daily snapshots |

**Key Point**: We train on **processing_date ranges**, not block timestamp ranges. The block timestamps are already properly windowed by the ingestion system.
