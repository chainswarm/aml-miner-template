# Plan: Developer Entry Points z Predefiniowanymi Parametrami

## Problem
Developer chce szybko uruchomić skrypty w PyCharm bez wpisywania długiej listy parametrów za każdym razem.

## Rozwiązanie
Każdy skrypt ma funkcję `main()` którą można wywołać z ustawionym `sys.argv`.

## Struktura

### Skrypt produkcyjny (scripts/train_model.py)
```python
def main():
    """Main entry point with argparse"""
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='alert_scorer')
    parser.add_argument('--window-days', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    
    service_name = f'{args.network}-{args.model_type}-training'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info("Initializing model training", extra={...})
    
    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        training = ModelTraining(
            network=args.network,
            start_date=args.start_date,
            end_date=args.end_date,
            client=client,
            model_type=args.model_type,
            window_days=args.window_days,
            output_dir=args.output_dir
        )
        training.run()


if __name__ == "__main__":
    main()
```

### Skrypt developerski (scripts/examples/example_train.py)
```python
"""Developer entry point - predefiniowane parametry dla wygody"""
import sys

if __name__ == "__main__":
    # Predefiniowane parametry - zmień jak potrzebujesz
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

## Użycie

### Produkcja (CLI z parametrami):
```bash
python scripts/train_model.py --network torus --start-date 2025-08-01 --end-date 2025-08-01 --model-type alert_scorer --window-days 195
```

### Development (PyCharm - po prostu Run):
```
Right click → Run 'example_train'
```
Albo:
```bash
python scripts/examples/example_train.py
```

## Zalety

✅ **Produkcja**: Pełna elastyczność przez CLI  
✅ **Development**: Jeden klik w PyCharm, zero wpisywania parametrów  
✅ **Predefiniowane wartości**: Łatwo zmienić w kodzie jak potrzebujesz  
✅ **Zachowana kompatybilność**: CLI działa jak dotychczas

## Pliki do modyfikacji

1. **scripts/train_model.py** - wydzielić `main()`
2. **scripts/ingest_data.py** - wydzielić `main()`
3. **scripts/score_batch.py** - wydzielić `main()`
4. **scripts/examples/example_train.py** - import main() i wywołanie
5. **scripts/examples/example_ingest.py** - import main() i wywołanie
6. **scripts/examples/example_score.py** - import main() i wywołanie
7. **scripts/examples/example_full_pipeline.py** - import main() dla wszystkich

## Przykład full pipeline (developerski)

```python
"""Full pipeline - wszystkie kroki z predefiniowanymi wartościami"""
import sys

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 1/3: Data Ingestion")
    print("=" * 80)
    sys.argv = ['ingest_data.py', '--network', 'torus', '--processing-date', '2025-08-01', '--days', '7']
    from scripts.ingest_data import main as ingest_main
    ingest_main()
    
    print("\n" + "=" * 80)
    print("STEP 2/3: Model Training")
    print("=" * 80)
    sys.argv = ['train_model.py', '--network', 'torus', '--start-date', '2025-08-01', '--end-date', '2025-08-01', '--model-type', 'alert_scorer', '--window-days', '195']
    from scripts.train_model import main as train_main
    train_main()
    
    print("\n" + "=" * 80)
    print("STEP 3/3: Risk Scoring")
    print("=" * 80)
    sys.argv = ['score_batch.py', '--network', 'torus', '--processing-date', '2025-08-01', '--window-days', '7']
    from scripts.score_batch import main as score_main
    score_main()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)