# Architecture Decision: scripts/ vs __main__ in Modules

## Current State

- `packages/training/model_training.py` has `__main__` block for CLI
- `scripts/` directory exists with some files
- Need to decide standard approach for all CLI entry points

## Option 1: scripts/ Directory (RECOMMENDED)

### Structure
```
packages/
  training/
    model_training.py          # Pure logic, no CLI
    feature_builder.py
    ...
  scoring/
    risk_scoring.py            # Pure logic, no CLI
    model_loader.py
    ...

scripts/
  train_models.py              # CLI wrapper for training
  score_batch.py               # CLI wrapper for scoring
  download_batch.py            # CLI wrapper for ingestion
  validate_models.py
  ...
```

### Pros

✅ **Clear Separation of Concerns**
- Modules = reusable logic
- Scripts = CLI entry points
- Each file has single responsibility

✅ **Better Reusability**
- Can import `ModelTraining` without CLI code
- Can create multiple CLIs for same logic:
  - `score_batch.py` - batch scoring
  - `score_realtime.py` - streaming scoring
  - Both use same `RiskScoring` class

✅ **Easier Discovery**
- All CLI entry points in one directory
- User knows where to look for executables
- Clear what can be run vs what is library code

✅ **Cleaner Testing**
- Test modules independently
- No CLI argparse logic in unit tests
- Mock CLI arguments easily in scripts

✅ **Better Package Distribution**
- Modules are pure Python library
- Scripts are optional CLI tools
- Can use modules in other projects

✅ **Follows Python Best Practices**
- Separation of concerns
- Single responsibility principle
- Library vs application code

✅ **Matches Existing Pattern**
- Already have `scripts/download_batch.py`
- Consistent with project structure

### Cons

⚠️ **Extra File Indirection**
- Need to look in `scripts/` and `packages/`
- Minimal issue with good naming

⚠️ **Slight Duplication**
- argparse code in scripts
- But scripts should be thin (10-20 lines)

## Option 2: __main__ in Modules

### Structure
```
packages/
  training/
    model_training.py          # Logic + CLI together
    feature_builder.py
    ...
  scoring/
    risk_scoring.py            # Logic + CLI together
    model_loader.py
    ...
```

### Pros

✅ **Single File**
- Everything in one place
- No indirection

✅ **Can Use python -m**
- Run as: `python -m packages.training.model_training`

### Cons

❌ **Mixed Responsibilities**
- Logic and CLI in same file
- Violates single responsibility

❌ **Less Reusable**
- Importing module brings CLI code
- Hard to use as library

❌ **Multiple CLIs Problem**
- Can't have multiple CLIs for same logic
- Would need multiple modules with same logic

❌ **Testing Complexity**
- CLI code mixed with logic
- Have to test argparse in same tests

❌ **Discovery Issues**
- Which modules have CLI? Need to check each file
- Not obvious what can be executed

❌ **Import Side Effects**
- Importing might trigger CLI code (if not guarded properly)

## Recommendation: Use scripts/ Directory

### Reasoning

1. **Aligns with Project Goals**
   - "We build new system" → clean architecture
   - "Fail fast" → clear separation helps debugging
   - "Single responsibility" → modules do one thing

2. **Scalability**
   - Easy to add new CLIs without changing modules
   - Can deprecate old CLIs without touching logic
   - Multiple scripts can share same module

3. **Matches Existing Pattern**
   - Already have `scripts/download_batch.py`
   - `scripts/train_models.py` exists
   - Just need to move CLI code there

4. **Better for Team**
   - Clear where executables are
   - Modules are pure Python
   - Easy to understand separation

## Implementation Plan

### 1. Keep Current Structure, Clean Up __main__

**Current:**
```python
# packages/training/model_training.py
class ModelTraining:
    ...

if __name__ == "__main__":
    # argparse + logic
    ...
```

**Move to:**
```python
# packages/training/model_training.py
class ModelTraining:
    ...
# No __main__ block

# scripts/train_models.py
from packages.training.model_training import ModelTraining

if __name__ == "__main__":
    # argparse
    # call ModelTraining
```

### 2. Create Thin Script Wrappers

Each script should be ~20 lines:
1. Import from packages
2. Parse arguments
3. Setup logging
4. Call module
5. Handle errors

Example:
```python
#!/usr/bin/env python3
from packages.training.model_training import ModelTraining
from packages import setup_logger
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True)
    # ... other args
    args = parser.parse_args()
    
    setup_logger(f'{args.network}-training')
    
    training = ModelTraining(
        network=args.network,
        # ... params from args
    )
    training.run()
```

### 3. Benefits for Risk Scoring

For the new risk scoring pipeline:

**Without scripts/ (bad):**
```python
# packages/scoring/risk_scoring.py
class RiskScoring:
    ...

if __name__ == "__main__":
    # CLI for batch scoring
    # What about real-time scoring?
    # What about validation scoring?
```

**With scripts/ (good):**
```python
# packages/scoring/risk_scoring.py
class RiskScoring:
    ...

# scripts/score_batch.py
from packages.scoring.risk_scoring import RiskScoring
# Batch scoring CLI

# scripts/score_realtime.py
from packages.scoring.risk_scoring import RiskScoring
# Real-time scoring CLI

# scripts/validate_scores.py
from packages.scoring.risk_scoring import RiskScoring
# Validation CLI
```

## Migration Strategy

### Phase 1: New Code
- All new pipelines use scripts/ pattern
- Risk scoring implementation uses this

### Phase 2: Existing Code (Optional)
- Can migrate `model_training.py` later if needed
- Not urgent, but would be consistent

## File Organization

```
scripts/
  # Data Pipeline
  download_batch.py          # Download from S3
  process_batch.py           # Process data
  
  # Training
  train_models.py            # Train all 3 models
  validate_models.py         # Validate trained models
  
  # Scoring
  score_batch.py             # Batch risk scoring
  score_realtime.py          # Real-time scoring (future)
  
  # Utilities
  init_database.py           # Initialize ClickHouse
  
packages/
  ingestion/
    sot_ingestion.py         # Pure logic
  training/
    model_training.py        # Pure logic
    feature_builder.py
    ...
  scoring/                   # NEW
    risk_scoring.py          # Pure logic
    model_loader.py
    ...
```

## Conclusion

**Use scripts/ directory for all CLI entry points.**

This provides:
- ✅ Clean separation
- ✅ Better reusability
- ✅ Easier testing
- ✅ Scalability
- ✅ Matches project patterns
- ✅ Follows best practices

The minimal extra indirection is worth the architectural benefits.