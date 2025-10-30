# Plan: Python Examples Using sys.argv

## Cel
Stworzyć przykłady Python które uruchamiają skrypty CLI bezpośrednio z poziomu Pythona zamiast przez bash.

## Podejście
Użyć `sys.argv` do ustawienia argumentów i zaimportować moduł skryptu.

## Przykładowa struktura

### example_ingest.py
```python
"""Example: Data Ingestion - uruchamia scripts/ingest_data.py"""
import sys

sys.argv = [
    'ingest_data.py',
    '--network', 'torus',
    '--processing-date', '2025-08-01',
    '--days', '7'
]

import scripts.ingest_data
```

### example_train.py
```python
"""Example: Model Training - uruchamia scripts/train_model.py"""
import sys

sys.argv = [
    'train_model.py',
    '--network', 'torus',
    '--start-date', '2025-08-01',
    '--end-date', '2025-08-01',
    '--model-type', 'alert_scorer',
    '--window-days', '195'
]

import scripts.train_model
```

### example_score.py
```python
"""Example: Risk Scoring - uruchamia scripts/score_batch.py"""
import sys

sys.argv = [
    'score_batch.py',
    '--network', 'torus',
    '--processing-date', '2025-08-01',
    '--window-days', '7'
]

import scripts.score_batch
```

### example_full_pipeline.py
```python
"""Example: Full Pipeline - uruchamia wszystkie kroki"""
import sys

# 1. Ingestion
sys.argv = [
    'ingest_data.py',
    '--network', 'torus',
    '--processing-date', '2025-08-01',
    '--days', '7'
]
import scripts.ingest_data

# 2. Training
sys.argv = [
    'train_model.py',
    '--network', 'torus',
    '--start-date', '2025-08-01',
    '--end-date', '2025-08-01',
    '--model-type', 'alert_scorer',
    '--window-days', '195'
]
import scripts.train_model

# 3. Scoring
sys.argv = [
    'score_batch.py',
    '--network', 'torus',
    '--processing-date', '2025-08-01',
    '--window-days', '7'
]
import scripts.score_batch
```

## Uruchomienie
```bash
python3 scripts/examples/example_ingest.py
python3 scripts/examples/example_train.py
python3 scripts/examples/example_score.py
python3 scripts/examples/example_full_pipeline.py
```

## Uwagi
- Proste i bezpośrednie
- Używa istniejących skryptów CLI
- Łatwo modyfikować parametry
- Nie wymaga bash/shell scripting