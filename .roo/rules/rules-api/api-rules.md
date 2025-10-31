# API Rules

## Data Handling
- APIs return empty results when no data exists, NOT exceptions
- Only raise HTTPException for actual errors (connection failures, invalid parameters)
- No data = empty list/dict with appropriate metadata
- Invalid network/date = 404 HTTPException
- Database errors = 500 HTTPException

## Example
```python
# WRONG - raises ValueError
def get_scores(processing_date):
    if not data:
        raise ValueError("No data")

# CORRECT - returns empty
def get_scores(processing_date):
    if not data:
        return []