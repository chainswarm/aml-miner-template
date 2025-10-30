# Plan: Dual Entry Points dla Scripts

## Cel
Każdy skrypt powinien mieć dwa sposoby uruchamiania:
1. CLI via argparse (obecne `if __name__ == "__main__":`)
2. Programmatic via funkcja `run()` z parametrami

## Struktura

### Przed (tylko CLI):
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', ...)
    args = parser.parse_args()
    
    # Logika
    training = ModelTraining(...)
    training.run()
```

### Po (dual entry points):
```python
def run(network: str, start_date: str, end_date: str, 
        model_type: str = 'alert_scorer', window_days: int = 7,
        output_dir: Path = None):
    """Programmatic entry point"""
    service_name = f'{network}-{model_type}-training'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info("Initializing model training", extra={...})
    
    connection_params = get_connection_params(network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        training = ModelTraining(
            network=network,
            start_date=start_date,
            end_date=end_date,
            client=client,
            model_type=model_type,
            window_days=window_days,
            output_dir=output_dir
        )
        training.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='alert_scorer')
    parser.add_argument('--window-days', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    
    run(
        network=args.network,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        window_days=args.window_days,
        output_dir=args.output_dir
    )
```

## Użycie

### CLI (jak dotychczas):
```bash
python scripts/train_model.py --network torus --start-date 2025-08-01 --end-date 2025-08-01 --model-type alert_scorer --window-days 195
```

### Programmatic (nowy sposób):
```python
from scripts.train_model import run

run(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    model_type='alert_scorer',
    window_days=195
)
```

## Przykłady po refaktoryzacji

### example_train.py (nowy sposób):
```python
"""Example: Model Training - używa funkcji run()"""
from scripts.train_model import run

run(
    network='torus',
    start_date='2025-08-01',
    end_date='2025-08-01',
    model_type='alert_scorer',
    window_days=195
)
```

### example_ingest.py (nowy sposób):
```python
"""Example: Data Ingestion - używa funkcji run()"""
from scripts.ingest_data import run

run(
    network='torus',
    processing_date='2025-08-01',
    days=7
)
```

### example_score.py (nowy sposób):
```python
"""Example: Risk Scoring - używa funkcji run()"""
from scripts.score_batch import run

run(
    network='torus',
    processing_date='2025-08-01',
    window_days=7
)
```

## Pliki do zmodyfikowania

1. **scripts/train_model.py** - dodać funkcję `run()`
2. **scripts/ingest_data.py** - dodać funkcję `run()`
3. **scripts/score_batch.py** - dodać funkcję `run()`
4. **scripts/examples/example_train.py** - użyć nowej funkcji
5. **scripts/examples/example_ingest.py** - użyć nowej funkcji
6. **scripts/examples/example_score.py** - użyć nowej funkcji
7. **scripts/examples/example_full_pipeline.py** - użyć nowych funkcji
8. **scripts/examples/README.md** - zaktualizować dokumentację

## Zalety

- ✅ Zachowana kompatybilność wsteczna (CLI działa jak dotychczas)
- ✅ Łatwiejsze użycie programmatic (bez manipulacji sys.argv)
- ✅ Lepsze type hints
- ✅ Czystszy kod w przykładach
- ✅ IDE autocomplete działa lepiej