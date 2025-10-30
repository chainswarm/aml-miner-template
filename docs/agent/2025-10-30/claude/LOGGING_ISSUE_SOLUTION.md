# Problem z Logami w example_full_pipeline.py

## Problem
Gdy uruchamiamy wiele skryptów w jednym procesie Pythona (`example_full_pipeline.py`), logi nie są wyświetlane poprawnie bo każdy skrypt wywołuje `setup_logger()` który może nadpisywać poprzednią konfigurację.

## Rozwiązania

### Opcja 1: Uruchamiaj skrypty osobno (ZALECANE dla developmentu)
```bash
python scripts/examples/example_ingest.py
python scripts/examples/example_train.py
python scripts/examples/example_score.py
```
✅ Każdy proces ma własną konfigurację loggera  
✅ Logi wyświetlane poprawnie

### Opcja 2: Użyj subprocess w full_pipeline
Zmodyfikować `example_full_pipeline.py` aby uruchamiał skrypty w osobnych procesach:

```python
import subprocess
import sys

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 1/3: Data Ingestion")
    print("=" * 80)
    subprocess.run([sys.executable, 'scripts/examples/example_ingest.py'], check=True)
    
    print("\n" + "=" * 80)
    print("STEP 2/3: Model Training")
    print("=" * 80)
    subprocess.run([sys.executable, 'scripts/examples/example_train.py'], check=True)
    
    print("\n" + "=" * 80)
    print("STEP 3/3: Risk Scoring")
    print("=" * 80)
    subprocess.run([sys.executable, 'scripts/examples/example_score.py'], check=True)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
```

✅ Każdy skrypt w osobnym procesie  
✅ Logi wyświetlane poprawnie  
❌ Wolniejsze (overhead tworzenia procesów)

### Opcja 3: Reset loggera między wywołaniami
```python
import sys
import importlib
from loguru import logger

if __name__ == "__main__":
    # Step 1
    sys.argv = [...]
    from scripts.ingest_data import main as ingest_main
    ingest_main()
    logger.remove()  # Wyczyść handlery
    
    # Step 2
    sys.argv = [...]
    from scripts.train_model import main as train_main
    train_main()
    logger.remove()  # Wyczyść handlery
    
    # ...
```

⚠️ Może nie zadziałać ze wszystkimi konfiguracjami loggera

## Rekomendacja

**Dla developmentu**: Używaj osobnych skryptów (Opcja 1)  
**Dla automatyzacji**: Użyj subprocess (Opcja 2)