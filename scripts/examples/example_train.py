"""Developer entry point for model training with predefined parameters"""
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