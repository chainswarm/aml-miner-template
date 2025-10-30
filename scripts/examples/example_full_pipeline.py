"""Developer entry point for full pipeline with predefined parameters"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get absolute path to examples directory
    examples_dir = Path(__file__).parent.absolute()
    
    print("=" * 80)
    print("STEP 1/3: Data Ingestion")
    print("=" * 80)
    result = subprocess.run(
        [sys.executable, str(examples_dir / 'example_ingest.py')], 
        check=True
    )
    
    print("\n" + "=" * 80)
    print("STEP 2/3: Model Training")
    print("=" * 80)
    result = subprocess.run(
        [sys.executable, str(examples_dir / 'example_train.py')], 
        check=True
    )
    
    print("\n" + "=" * 80)
    print("STEP 3/3: Risk Scoring")
    print("=" * 80)
    result = subprocess.run(
        [sys.executable, str(examples_dir / 'example_score.py')], 
        check=True
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)